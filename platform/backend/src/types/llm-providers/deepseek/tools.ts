import { z } from "zod";

export const FunctionDefinitionParametersSchema = z
  .record(z.string(), z.unknown())
  .optional()
  .describe(`
    The parameters the functions accepts, described as a JSON Schema object.
    Omitting parameters defines a function with an empty parameter list.
  `);

const FunctionDefinitionSchema = z
  .object({
    name: z.string(),
    description: z.string().optional(),
    parameters: FunctionDefinitionParametersSchema,
    strict: z.boolean().nullable().optional(),
  })
  .describe("Function definition");

const FunctionToolSchema = z
  .object({
    type: z.enum(["function"]),
    function: FunctionDefinitionSchema,
  })
  .describe("Function tool");

const CustomToolSchema = z
  .object({
    type: z.enum(["custom"]),
    custom: z.object({
      name: z
        .string()
        .describe(
          "The name of the custom tool, used to identify it in tool calls",
        ),
      description: z
        .string()
        .optional()
        .describe(
          "Optional description of the custom tool, used to provide more context",
        ),
      format: z
        .union([
          z
            .object({
              type: z
                .enum(["text"])
                .describe("Unconstrained text format. Always `text`"),
            })
            .describe("Unconstrained free-form text"),
          z.object({
            type: z.enum(["grammar"]),
            grammar: z
              .object({
                definition: z.string().describe("The grammar definition"),
                syntax: z
                  .enum(["lark", "regex"])
                  .describe("The syntax of the grammar definition"),
              })
              .describe("Your chosen grammar"),
          }),
        ])
        .optional()
        .describe(
          "The input format for the custom tool. Default is unconstrained text.",
        ),
    }),
  })
  .describe("Custom tool definition");

const AllowedToolsSchema = z
  .object({
    mode: z.enum(["auto", "required"]).describe(`
    Constrains the tools available to the model to a pre-defined set.

    auto allows the model to pick from among the allowed tools and generate a
    message.

    required requires the model to call one or more of the allowed tools.
    `),
    tools: z
      .array(z.record(z.string(), FunctionToolSchema))
      .describe(
        "A list of tool definitions that the model should be allowed to call",
      ),
  })
  .describe("Allowed tools configuration");

const AllowedToolChoiceSchema = z
  .object({
    type: z.enum(["allowed_tools"]),
    allowed_tools: AllowedToolsSchema,
  })
  .describe("Allowed tool choice");

const NamedToolChoiceSchema = z
  .object({
    type: z.enum(["function"]),
    function: z.object({
      name: z.string(),
    }),
  })
  .describe("Named tool choice");

export const ToolSchema = z
  .union([FunctionToolSchema, CustomToolSchema])
  .describe("Tool definition (function or custom)");

export const ToolChoiceOptionSchema = z
  .union([
    z.enum(["none", "auto", "required"]),
    AllowedToolChoiceSchema,
    NamedToolChoiceSchema,
    CustomToolSchema,
  ])
  .describe("Tool choice option");
