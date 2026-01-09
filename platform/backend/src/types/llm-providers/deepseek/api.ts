import { z } from "zod";

import { MessageParamSchema, ToolCallSchema } from "./messages";
import { ToolChoiceOptionSchema, ToolSchema } from "./tools";

export const ChatCompletionUsageSchema = z
  .object({
    completion_tokens: z.number(),
    prompt_tokens: z.number(),
    total_tokens: z.number(),
    completion_tokens_details: z
      .any()
      .optional()
      .describe("Details about completion tokens"),
    prompt_tokens_details: z
      .any()
      .optional()
      .describe("Details about prompt tokens"),
    prompt_cache_hit_tokens: z
      .number()
      .optional()
      .describe("DeepSeek: Number of prompt tokens that hit the cache"),
    prompt_cache_miss_tokens: z
      .number()
      .optional()
      .describe("DeepSeek: Number of prompt tokens that missed the cache"),
  })
  .describe("DeepSeek API usage statistics");

export const FinishReasonSchema = z.enum([
  "stop",
  "length",
  "tool_calls",
  "content_filter",
  "function_call",
]);

const ChoiceSchema = z
  .object({
    finish_reason: FinishReasonSchema,
    index: z.number(),
    logprobs: z.any().nullable(),
    message: z
      .object({
        content: z.string().nullable(),
        refusal: z.string().nullable().optional(),
        role: z.enum(["assistant"]),
        reasoning_content: z
          .string()
          .nullable()
          .optional()
          .describe("DeepSeek-Reasoner: Chain of thought content"),
        function_call: z
          .object({
            arguments: z.string(),
            name: z.string(),
          })
          .nullable()
          .optional(),
        tool_calls: z.array(ToolCallSchema).optional(),
      })
      .describe("DeepSeek chat completion message"),
  })
  .describe("DeepSeek chat completion choice");

export const ChatCompletionRequestSchema = z
  .object({
    model: z.string(),
    messages: z.array(MessageParamSchema),
    tools: z.array(ToolSchema).optional(),
    tool_choice: ToolChoiceOptionSchema.optional(),
    temperature: z.number().nullable().optional(),
    max_tokens: z.number().nullable().optional(),
    stream: z.boolean().nullable().optional(),
  })
  .describe("DeepSeek chat completion request (OpenAI-compatible)");

export const ChatCompletionResponseSchema = z
  .object({
    id: z.string(),
    choices: z.array(ChoiceSchema),
    created: z.number(),
    model: z.string(),
    object: z.enum(["chat.completion"]),
    system_fingerprint: z.string().nullable().optional(),
    usage: ChatCompletionUsageSchema.optional(),
  })
  .describe("DeepSeek chat completion response (OpenAI-compatible)");

export const ChatCompletionsHeadersSchema = z.object({
  "user-agent": z.string().optional().describe("The user agent of the client"),
  authorization: z
    .string()
    .describe("Bearer token for DeepSeek")
    .transform((authorization) => authorization.replace("Bearer ", "")),
});
