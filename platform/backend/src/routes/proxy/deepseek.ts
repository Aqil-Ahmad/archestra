import fastifyHttpProxy from "@fastify/http-proxy";
import { RouteId } from "@shared";
import type { FastifyReply } from "fastify";
import type { FastifyPluginAsyncZod } from "fastify-type-provider-zod";
import OpenAIProvider from "openai";
import { z } from "zod";
import config from "@/config";
import getDefaultPricing from "@/default-model-prices";
import {
  getObservableFetch,
  reportBlockedTools,
  reportLLMCost,
  reportLLMTokens,
  reportTimeToFirstToken,
  reportTokensPerSecond,
} from "@/llm-metrics";
import {
  AgentModel,
  InteractionModel,
  LimitValidationService,
  TokenPriceModel,
} from "@/models";
import {
  type Agent,
  ApiError,
  constructResponseSchema,
  DeepSeek,
  UuidIdSchema,
} from "@/types";
import { PROXY_API_PREFIX, PROXY_BODY_LIMIT } from "./common";
import * as utils from "./utils";

const deepSeekProxyRoutes: FastifyPluginAsyncZod = async (fastify) => {
  const API_PREFIX = `${PROXY_API_PREFIX}/deepseek`;
  const CHAT_COMPLETIONS_SUFFIX = "chat/completions";

  await fastify.register(fastifyHttpProxy, {
    upstream: config.llm.deepseek.baseUrl,
    prefix: `${API_PREFIX}`,
    rewritePrefix: "",
    preHandler: (request, _reply, next) => {
      if (
        request.method === "POST" &&
        request.url.includes(CHAT_COMPLETIONS_SUFFIX)
      ) {
        fastify.log.info(
          {
            method: request.method,
            url: request.url,
            action: "skip-proxy",
            reason: "handled-by-custom-handler",
          },
          "DeepSeek proxy preHandler: skipping chat/completions route",
        );
        next(new Error("skip"));
        return;
      }

      const pathAfterPrefix = request.url.replace(API_PREFIX, "");
      const uuidMatch = pathAfterPrefix.match(
        /^\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(\/.*)?$/i,
      );

      if (uuidMatch) {
        const remainingPath = uuidMatch[2] || "";
        const originalUrl = request.raw.url;
        request.raw.url = `${API_PREFIX}${remainingPath}`;

        fastify.log.info(
          {
            method: request.method,
            originalUrl,
            rewrittenUrl: request.raw.url,
            upstream: config.llm.deepseek.baseUrl,
            finalProxyUrl: `${config.llm.deepseek.baseUrl}/v1${remainingPath}`,
          },
          "DeepSeek proxy preHandler: URL rewritten (UUID stripped)",
        );
      } else {
        fastify.log.info(
          {
            method: request.method,
            url: request.url,
            upstream: config.llm.deepseek.baseUrl,
            finalProxyUrl: `${config.llm.deepseek.baseUrl}/v1${pathAfterPrefix}`,
          },
          "DeepSeek proxy preHandler: proxying request",
        );
      }

      next();
    },
  });

  const handleChatCompletion = async (
    body: DeepSeek.Types.ChatCompletionsRequest,
    headers: DeepSeek.Types.ChatCompletionsHeaders,
    reply: FastifyReply,
    _organizationId: string,
    agentId?: string,
    externalAgentId?: string,
    userId?: string,
  ) => {
    const { messages, tools, stream } = body;

    let resolvedAgent: Agent;
    if (agentId) {
      const agent = await AgentModel.findById(agentId);
      if (!agent) {
        return reply.status(404).send({
          error: {
            message: `Agent with ID ${agentId} not found`,
            type: "not_found",
          },
        });
      }
      resolvedAgent = agent;
    } else {
      resolvedAgent = await AgentModel.getAgentOrCreateDefault(
        headers["user-agent"],
      );
    }

    const resolvedAgentId = resolvedAgent.id;

    fastify.log.info(
      {
        resolvedAgentId,
        agentName: resolvedAgent.name,
        wasExplicit: !!agentId,
      },
      "Agent resolved",
    );

    const { authorization } = headers;
    // Strip "Bearer " prefix - the OpenAI SDK adds its own
    const deepSeekApiKey = authorization?.replace(/^Bearer\s+/i, "") ?? "";

    // Ensure baseUrl includes /v1 for OpenAI SDK compatibility
    const baseUrl = config.llm.deepseek.baseUrl.endsWith("/v1")
      ? config.llm.deepseek.baseUrl
      : `${config.llm.deepseek.baseUrl}/v1`;

    const deepSeekClient = new OpenAIProvider({
      apiKey: deepSeekApiKey,
      baseURL: baseUrl,
      fetch: getObservableFetch("deepseek", resolvedAgent, externalAgentId),
    });

    try {
      const limitViolation =
        await LimitValidationService.checkLimitsBeforeRequest(resolvedAgentId);

      if (limitViolation) {
        const [_refusalMessage, contentMessage] = limitViolation;

        fastify.log.info(
          {
            resolvedAgentId,
            reason: "token_cost_limit_exceeded",
            contentMessage,
          },
          "DeepSeek request blocked due to token cost limit",
        );

        return reply.status(429).send({
          error: {
            message: contentMessage,
            type: "rate_limit_exceeded",
            code: "token_cost_limit_exceeded",
          },
        });
      }

      await utils.tools.persistTools(
        (tools || []).map((tool) => {
          if (tool.type === "function") {
            return {
              toolName: tool.function.name,
              toolParameters: tool.function.parameters || {},
              toolDescription: tool.function.description || "",
            };
          } else {
            return {
              toolName: tool.custom.name,
              toolParameters: tool.custom.format || {},
              toolDescription: tool.custom.description || "",
            };
          }
        }),
        resolvedAgentId,
      );

      const mergedTools = tools || [];

      const enabledToolNames = new Set(
        mergedTools.map((tool) =>
          tool.type === "function" ? tool.function.name : tool.custom.name,
        ),
      );

      const baselineModel = body.model;
      let model = baselineModel;
      const hasTools = (tools?.length ?? 0) > 0;
      const optimizedModel = await utils.costOptimization.getOptimizedModel(
        resolvedAgent,
        messages,
        "deepseek",
        hasTools,
      );

      if (optimizedModel) {
        model = optimizedModel;
        fastify.log.info(
          { resolvedAgentId, optimizedModel },
          "Optimized model selected",
        );
      } else {
        fastify.log.info(
          { resolvedAgentId, baselineModel },
          "No matching optimized model found, proceeding with baseline model",
        );
      }

      const baselinePricing = getDefaultPricing(baselineModel);
      await TokenPriceModel.createIfNotExists(baselineModel, {
        provider: "deepseek",
        ...baselinePricing,
      });

      if (model !== baselineModel) {
        const optimizedPricing = getDefaultPricing(model);
        await TokenPriceModel.createIfNotExists(model, {
          provider: "deepseek",
          ...optimizedPricing,
        });
      }

      const commonMessages = utils.adapters.deepseek.toCommonFormat(messages);

      const { toolResultUpdates, contextIsTrusted } =
        await utils.trustedData.evaluateIfContextIsTrusted(
          commonMessages,
          resolvedAgentId,
          deepSeekApiKey,
          "deepseek",
          resolvedAgent.considerContextUntrusted,
          stream
            ? () => {
                const startChunk = {
                  id: "chatcmpl-sanitizing",
                  object: "chat.completion.chunk" as const,
                  created: Date.now() / 1000,
                  model: model,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        role: "assistant" as const,
                        content: "Analyzing with Dual LLM:\n\n",
                      },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                reply.raw.write(`data: ${JSON.stringify(startChunk)}\n\n`);
              }
            : undefined,
          stream
            ? (progress) => {
                const optionsText = progress.options
                  .map((opt, idx) => `  ${idx}: ${opt}`)
                  .join("\n");
                const progressChunk = {
                  id: "chatcmpl-sanitizing",
                  object: "chat.completion.chunk" as const,
                  created: Date.now() / 1000,
                  model: model,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        content: `Question: ${progress.question}\nOptions:\n${optionsText}\nAnswer: ${progress.answer}\n\n`,
                      },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                reply.raw.write(`data: ${JSON.stringify(progressChunk)}\n\n`);
              }
            : undefined,
        );

      let filteredMessages = utils.adapters.deepseek.applyUpdates(
        messages,
        toolResultUpdates,
      );

      let toonTokensBefore: number | null = null;
      let toonTokensAfter: number | null = null;
      let toonCostSavings: number | null = null;
      const shouldApplyToonCompression =
        await utils.toonConversion.shouldApplyToonCompression(resolvedAgentId);

      if (shouldApplyToonCompression) {
        const { messages: convertedMessages, stats } =
          await utils.adapters.deepseek.convertToolResultsToToon(
            filteredMessages,
            model,
          );
        filteredMessages = convertedMessages;
        toonTokensBefore = stats.toonTokensBefore;
        toonTokensAfter = stats.toonTokensAfter;
        toonCostSavings = stats.toonCostSavings;
      }

      fastify.log.info(
        {
          shouldApplyToonCompression,
          toonTokensBefore,
          toonTokensAfter,
          toonCostSavings,
        },
        "deepseek proxy routes: handle chat completions: tool results compression completed",
      );

      fastify.log.info(
        {
          resolvedAgentId,
          originalMessagesCount: messages.length,
          filteredMessagesCount: filteredMessages.length,
          toolResultUpdatesCount: Object.keys(toolResultUpdates).length,
          contextIsTrusted,
        },
        "Messages filtered after trusted data evaluation",
      );

      if (stream) {
        const streamStartTime = Date.now();
        let firstChunkTime: number | undefined;

        const streamingResponse = await utils.tracing.startActiveLlmSpan(
          "deepseek.chat.completions",
          "deepseek",
          model,
          true,
          resolvedAgent,
          async (llmSpan) => {
            const response = await deepSeekClient.chat.completions.create({
              ...body,
              model,
              messages:
                filteredMessages as OpenAIProvider.Chat.Completions.ChatCompletionMessageParam[],
              tools:
                mergedTools.length > 0
                  ? (mergedTools as OpenAIProvider.Chat.Completions.ChatCompletionTool[])
                  : undefined,
              stream: true,
              stream_options: { include_usage: true },
            });
            llmSpan.end();
            return response;
          },
        );

        reply.raw.writeHead(200, {
          "Content-Type": "text/event-stream; charset=utf-8",
        });

        let accumulatedContent = "";
        let accumulatedRefusal = "";
        const accumulatedToolCalls: OpenAIProvider.Chat.Completions.ChatCompletionMessageFunctionToolCall[] =
          [];
        const chunks: OpenAIProvider.Chat.Completions.ChatCompletionChunk[] =
          [];
        let tokenUsage: { input?: number; output?: number } | undefined;

        let assistantMessage:
          | OpenAIProvider.Chat.Completions.ChatCompletionMessage
          | undefined;

        try {
          for await (const chunk of streamingResponse) {
            if (!firstChunkTime) {
              firstChunkTime = Date.now();
              const ttftSeconds = (firstChunkTime - streamStartTime) / 1000;
              reportTimeToFirstToken(
                "deepseek",
                resolvedAgent,
                model,
                ttftSeconds,
                externalAgentId,
              );
            }

            chunks.push(chunk);

            if (chunk.usage) {
              tokenUsage = utils.adapters.deepseek.getUsageTokens(chunk.usage);
            }
            const delta = chunk.choices[0]?.delta;
            const finishReason = chunk.choices[0]?.finish_reason;

            if (
              !delta?.tool_calls &&
              (delta?.content !== undefined ||
                delta?.refusal !== undefined ||
                delta?.role ||
                finishReason)
            ) {
              reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);

              if (delta?.content) {
                accumulatedContent += delta.content;
              }
              if (delta?.refusal) {
                accumulatedRefusal += delta.refusal;
              }
            }

            if (delta?.tool_calls) {
              for (const toolCallDelta of delta.tool_calls) {
                const index = toolCallDelta.index;

                if (!accumulatedToolCalls[index]) {
                  accumulatedToolCalls[index] = {
                    id: toolCallDelta.id || "",
                    type: "function",
                    function: {
                      name: "",
                      arguments: "",
                    },
                  };
                }

                if (toolCallDelta.id) {
                  accumulatedToolCalls[index].id = toolCallDelta.id;
                }
                if (toolCallDelta.function?.name) {
                  accumulatedToolCalls[index].function.name =
                    toolCallDelta.function.name;
                }
                if (toolCallDelta.function?.arguments) {
                  accumulatedToolCalls[index].function.arguments +=
                    toolCallDelta.function.arguments;
                }
              }
            }
          }

          assistantMessage = {
            role: "assistant",
            content: accumulatedContent || null,
            refusal: accumulatedRefusal || null,
            tool_calls:
              accumulatedToolCalls.length > 0
                ? accumulatedToolCalls
                : undefined,
          };

          const toolInvocationRefusal =
            await utils.toolInvocation.evaluatePolicies(
              (assistantMessage.tool_calls || []).map((toolCall) => {
                if (toolCall.type === "function") {
                  return {
                    toolCallName: toolCall.function.name,
                    toolCallArgs: toolCall.function.arguments,
                  };
                } else {
                  return {
                    toolCallName: toolCall.custom.name,
                    toolCallArgs: toolCall.custom.input,
                  };
                }
              }),
              resolvedAgentId,
              contextIsTrusted,
              enabledToolNames,
            );

          if (accumulatedToolCalls.length > 0) {
            if (toolInvocationRefusal) {
              const [refusalMessage, contentMessage] = toolInvocationRefusal;
              assistantMessage = {
                role: "assistant",
                refusal: refusalMessage,
                content: contentMessage,
              };

              const refusalChunk = {
                id: "chatcmpl-blocked",
                object: "chat.completion.chunk" as const,
                created: Date.now() / 1000,
                model: model,
                choices: [
                  {
                    index: 0,
                    delta:
                      assistantMessage as OpenAIProvider.Chat.Completions.ChatCompletionChunk.Choice.Delta,
                    finish_reason: "stop" as const,
                    logprobs: null,
                  },
                ],
              };
              reply.raw.write(`data: ${JSON.stringify(refusalChunk)}\n\n`);
              reportBlockedTools(
                "deepseek",
                resolvedAgent,
                accumulatedToolCalls.length,
                model,
                externalAgentId,
              );
            } else {
              for (const [index, toolCall] of accumulatedToolCalls.entries()) {
                const baseChunk = {
                  id: chunks[0]?.id || "chatcmpl-unknown",
                  object: "chat.completion.chunk" as const,
                  created: chunks[0]?.created || Date.now() / 1000,
                  model: model,
                };

                const idChunk = {
                  ...baseChunk,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        tool_calls: [
                          {
                            index,
                            id: toolCall.id,
                            type: "function" as const,
                            function: {
                              name: toolCall.function.name,
                              arguments: "",
                            },
                          },
                        ],
                      },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                reply.raw.write(`data: ${JSON.stringify(idChunk)}\n\n`);

                const argsChunk = {
                  ...baseChunk,
                  choices: [
                    {
                      index: 0,
                      delta: {
                        tool_calls: [
                          {
                            index,
                            id: toolCall.id,
                            function: {
                              arguments: toolCall.function.arguments,
                            },
                          },
                        ],
                      },
                      finish_reason: null,
                      logprobs: null,
                    },
                  ],
                };
                reply.raw.write(`data: ${JSON.stringify(argsChunk)}\n\n`);
              }
            }
          }

          reply.raw.write("data: [DONE]\n\n");
          reply.raw.end();
          return reply;
        } finally {
          if (!assistantMessage) {
            fastify.log.info(
              "Stream was aborted before completion, building partial response",
            );

            for (const toolCall of accumulatedToolCalls) {
              try {
                toolCall.function.arguments = JSON.parse(
                  toolCall.function.arguments,
                );
              } catch {
                // If parsing fails, leave as string
              }
            }

            assistantMessage = {
              role: "assistant",
              content: accumulatedContent || null,
              refusal: accumulatedRefusal || null,
              tool_calls:
                accumulatedToolCalls.length > 0
                  ? accumulatedToolCalls
                  : undefined,
            };
          }

          if (tokenUsage) {
            reportLLMTokens(
              "deepseek",
              resolvedAgent,
              tokenUsage,
              model,
              externalAgentId,
            );

            if (tokenUsage.output && firstChunkTime) {
              const totalDurationSeconds =
                (Date.now() - streamStartTime) / 1000;
              reportTokensPerSecond(
                "deepseek",
                resolvedAgent,
                model,
                tokenUsage.output,
                totalDurationSeconds,
                externalAgentId,
              );
            }
          }

          let baselineCost: number | null = null;
          let costAfterOptimization: number | null = null;

          if (tokenUsage) {
            baselineCost =
              (await utils.costOptimization.calculateCost(
                body.model,
                tokenUsage.input || 0,
                tokenUsage.output || 0,
              )) ?? null;
            costAfterOptimization =
              (await utils.costOptimization.calculateCost(
                model,
                tokenUsage.input || 0,
                tokenUsage.output || 0,
              )) ?? null;

            fastify.log.info(
              {
                baselineCost,
                costAfterModelOptimization: costAfterOptimization,
                inputTokens: tokenUsage.input,
                outputTokens: tokenUsage.output,
              },
              "deepseek proxy routes: handle chat completions: costs",
            );
          } else {
            fastify.log.warn(
              "No token usage available for streaming request - recording interaction without usage data",
            );
          }
          reportLLMCost(
            "deepseek",
            resolvedAgent,
            model,
            costAfterOptimization,
            externalAgentId,
          );

          await InteractionModel.create({
            profileId: resolvedAgentId,
            externalAgentId,
            userId,
            type: "deepseek:chatCompletions",
            request: body,
            processedRequest: {
              ...body,
              messages: filteredMessages,
            },
            response: {
              id: chunks[0]?.id || "chatcmpl-unknown",
              object: "chat.completion",
              created: chunks[0]?.created || Date.now() / 1000,
              model: model,
              choices: [
                {
                  index: 0,
                  message: assistantMessage,
                  finish_reason: "stop",
                  logprobs: null,
                },
              ],
            },
            model: model,
            inputTokens: tokenUsage?.input || null,
            outputTokens: tokenUsage?.output || null,
            cost: costAfterOptimization?.toFixed(10) ?? null,
            baselineCost: baselineCost?.toFixed(10) ?? null,
            toonTokensBefore,
            toonTokensAfter,
            toonCostSavings: toonCostSavings?.toFixed(10) ?? null,
          });
        }
      } else {
        const response = await utils.tracing.startActiveLlmSpan(
          "deepseek.chat.completions",
          "deepseek",
          model,
          false,
          resolvedAgent,
          async (llmSpan) => {
            const response = await deepSeekClient.chat.completions.create({
              ...body,
              model,
              messages:
                filteredMessages as OpenAIProvider.Chat.Completions.ChatCompletionMessageParam[],
              tools:
                mergedTools.length > 0
                  ? (mergedTools as OpenAIProvider.Chat.Completions.ChatCompletionTool[])
                  : undefined,
              stream: false,
            });
            llmSpan.end();
            return response;
          },
        );

        let assistantMessage = response.choices[0].message;

        const toolInvocationRefusal =
          await utils.toolInvocation.evaluatePolicies(
            (assistantMessage.tool_calls || []).map((toolCall) => {
              if (toolCall.type === "function") {
                return {
                  toolCallName: toolCall.function.name,
                  toolCallArgs: toolCall.function.arguments,
                };
              } else {
                return {
                  toolCallName: toolCall.custom.name,
                  toolCallArgs: toolCall.custom.input,
                };
              }
            }),
            resolvedAgentId,
            contextIsTrusted,
            enabledToolNames,
          );

        if (toolInvocationRefusal) {
          const [refusalMessage, contentMessage] = toolInvocationRefusal;
          const blockedCount = assistantMessage.tool_calls?.length || 0;

          assistantMessage = {
            role: "assistant",
            refusal: refusalMessage,
            content: contentMessage,
          };
          response.choices = [
            {
              index: 0,
              message: assistantMessage,
              finish_reason: "stop",
              logprobs: null,
            },
          ];

          reportBlockedTools(
            "deepseek",
            resolvedAgent,
            blockedCount,
            model,
            externalAgentId,
          );
        }

        const tokenUsage = response.usage
          ? utils.adapters.deepseek.getUsageTokens(response.usage)
          : { input: null, output: null };

        const baselineCost = await utils.costOptimization.calculateCost(
          body.model,
          tokenUsage.input,
          tokenUsage.output,
        );

        const costAfterOptimization =
          await utils.costOptimization.calculateCost(
            model,
            tokenUsage.input,
            tokenUsage.output,
          );
        reportLLMCost(
          "deepseek",
          resolvedAgent,
          model,
          costAfterOptimization,
          externalAgentId,
        );

        await InteractionModel.create({
          profileId: resolvedAgentId,
          externalAgentId,
          userId,
          type: "deepseek:chatCompletions",
          request: body,
          processedRequest: {
            ...body,
            messages: filteredMessages,
          },
          response,
          model: model,
          inputTokens: tokenUsage.input,
          outputTokens: tokenUsage.output,
          cost: costAfterOptimization?.toFixed(10) ?? null,
          baselineCost: baselineCost?.toFixed(10) ?? null,
          toonTokensBefore,
          toonTokensAfter,
          toonCostSavings: toonCostSavings?.toFixed(10) ?? null,
        });

        return reply.send(response);
      }
    } catch (error) {
      fastify.log.error(error);

      const statusCode =
        error instanceof Error && "status" in error
          ? (error.status as 400 | 404 | 403 | 500)
          : 500;

      const message =
        error instanceof Error ? error.message : "Internal server error";

      throw new ApiError(statusCode, message);
    }
  };

  fastify.post(
    `${API_PREFIX}/${CHAT_COMPLETIONS_SUFFIX}`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.DeepSeekChatCompletionsWithDefaultAgent,
        description:
          "Create a chat completion with DeepSeek (uses default agent)",
        tags: ["llm-proxy"],
        body: DeepSeek.API.ChatCompletionRequestSchema,
        headers: DeepSeek.API.ChatCompletionsHeadersSchema,
        response: constructResponseSchema(
          DeepSeek.API.ChatCompletionResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const externalAgentId = utils.externalAgentId.getExternalAgentId(
        request.headers,
      );
      const userId = await utils.userId.getUserId(request.headers);
      return handleChatCompletion(
        request.body,
        request.headers,
        reply,
        request.organizationId,
        undefined,
        externalAgentId,
        userId,
      );
    },
  );

  fastify.post(
    `${API_PREFIX}/:agentId/${CHAT_COMPLETIONS_SUFFIX}`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.DeepSeekChatCompletionsWithAgent,
        description:
          "Create a chat completion with DeepSeek for a specific agent",
        tags: ["llm-proxy"],
        params: z.object({
          agentId: UuidIdSchema,
        }),
        body: DeepSeek.API.ChatCompletionRequestSchema,
        headers: DeepSeek.API.ChatCompletionsHeadersSchema,
        response: constructResponseSchema(
          DeepSeek.API.ChatCompletionResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const externalAgentId = utils.externalAgentId.getExternalAgentId(
        request.headers,
      );
      const userId = await utils.userId.getUserId(request.headers);
      return handleChatCompletion(
        request.body,
        request.headers,
        reply,
        request.organizationId,
        request.params.agentId,
        externalAgentId,
        userId,
      );
    },
  );
};

export default deepSeekProxyRoutes;
