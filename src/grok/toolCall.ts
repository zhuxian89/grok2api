/**
 * Tool call utilities for OpenAI-compatible function calling.
 *
 * Provides prompt-based emulation of tool calls by injecting tool definitions
 * into the system prompt and parsing structured responses.
 */

export interface ToolDefinition {
  type?: string;
  function?: {
    name?: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface OpenAIMessage {
  role: string;
  content?: string | unknown;
  tool_calls?: Array<{
    function?: { name?: string; arguments?: string };
  }>;
  tool_call_id?: string;
  name?: string;
}

const TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

function generateCallId(): string {
  const hex = crypto.randomUUID().replace(/-/g, "").slice(0, 24);
  return `call_${hex}`;
}

export function buildToolPrompt(
  tools: ToolDefinition[],
  toolChoice?: unknown,
  parallelToolCalls: boolean = true,
): string {
  if (!tools || !tools.length) return "";
  if (toolChoice === "none") return "";

  const lines: string[] = [
    "# Available Tools",
    "",
    'You have access to the following tools. To call a tool, output a <tool_call> block with a JSON object containing "name" and "arguments".',
    "",
    "Format:",
    "<tool_call>",
    '{"name": "function_name", "arguments": {"param": "value"}}',
    "</tool_call>",
    "",
  ];

  if (parallelToolCalls) {
    lines.push("You may make multiple tool calls in a single response by using multiple <tool_call> blocks.");
    lines.push("");
  }

  lines.push("## Tool Definitions");
  lines.push("");
  for (const tool of tools) {
    if (tool.type !== "function") continue;
    const func = tool.function ?? {};
    const name = func.name ?? "";
    const desc = func.description ?? "";
    const params = func.parameters;

    lines.push(`### ${name}`);
    if (desc) lines.push(desc);
    if (params) lines.push(`Parameters: ${JSON.stringify(params)}`);
    lines.push("");
  }

  if (toolChoice === "required") {
    lines.push("IMPORTANT: You MUST call at least one tool in your response. Do not respond with only text.");
  } else if (typeof toolChoice === "object" && toolChoice !== null) {
    const funcInfo = (toolChoice as Record<string, any>).function ?? {};
    const forcedName = funcInfo.name ?? "";
    if (forcedName) {
      lines.push(`IMPORTANT: You MUST call the tool "${forcedName}" in your response.`);
    }
  } else {
    lines.push("Decide whether to call a tool based on the user's request. If you don't need a tool, respond normally with text only.");
  }

  lines.push("");
  lines.push("When you call a tool, you may include text before or after the <tool_call> blocks, but the tool call blocks must be valid JSON.");

  return lines.join("\n");
}

// --- JSON repair utilities ---

function stripCodeFences(text: string): string {
  if (!text) return text;
  let cleaned = text.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```[a-zA-Z0-9_-]*\s*/, "");
    cleaned = cleaned.replace(/\s*```$/, "");
  }
  return cleaned.trim();
}

function extractJsonObject(text: string): string {
  if (!text) return text;
  const start = text.indexOf("{");
  if (start === -1) return text;
  const end = text.lastIndexOf("}");
  if (end === -1) return text.slice(start);
  if (end < start) return text;
  return text.slice(start, end + 1);
}

function removeTrailingCommas(text: string): string {
  if (!text) return text;
  return text.replace(/,\s*([}\]])/g, "$1");
}

function balanceBraces(text: string): string {
  if (!text) return text;
  let openCount = 0;
  let closeCount = 0;
  let inString = false;
  let escape = false;
  for (const ch of text) {
    if (escape) { escape = false; continue; }
    if (ch === "\\" && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === "{") openCount++;
    else if (ch === "}") closeCount++;
  }
  if (openCount > closeCount) {
    return text + "}".repeat(openCount - closeCount);
  }
  return text;
}

function repairJson(text: string): unknown | null {
  if (!text) return null;
  let cleaned = stripCodeFences(text);
  cleaned = extractJsonObject(cleaned);
  cleaned = cleaned.replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/\n/g, " ");
  cleaned = removeTrailingCommas(cleaned);
  cleaned = balanceBraces(cleaned);
  try {
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

// --- Parsing ---

export function parseToolCallBlock(
  rawJson: string,
  tools?: ToolDefinition[],
): ToolCall | null {
  if (!rawJson) return null;

  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch {
    parsed = repairJson(rawJson);
  }
  if (!parsed || typeof parsed !== "object") return null;

  const obj = parsed as Record<string, unknown>;
  const name = obj.name;
  if (typeof name !== "string" || !name) return null;

  // Validate against known tool names
  if (tools && tools.length) {
    const validNames = new Set<string>();
    for (const tool of tools) {
      const tn = tool.function?.name;
      if (tn) validNames.add(tn);
    }
    if (validNames.size && !validNames.has(name)) return null;
  }

  const args = obj.arguments ?? {};
  let argumentsStr: string;
  if (typeof args === "string") {
    argumentsStr = args;
  } else {
    argumentsStr = JSON.stringify(args);
  }

  return {
    id: generateCallId(),
    type: "function",
    function: { name, arguments: argumentsStr },
  };
}

export function parseToolCalls(
  content: string,
  tools?: ToolDefinition[],
): { text: string | null; toolCalls: ToolCall[] | null } {
  if (!content) return { text: content, toolCalls: null };

  const matches: Array<{ start: number; end: number; json: string }> = [];
  const re = new RegExp(TOOL_CALL_RE.source, TOOL_CALL_RE.flags);
  let m: RegExpExecArray | null;
  while ((m = re.exec(content)) !== null) {
    matches.push({ start: m.index, end: m.index + m[0].length, json: (m[1] ?? "").trim() });
  }

  if (!matches.length) return { text: content, toolCalls: null };

  const toolCalls: ToolCall[] = [];
  for (const match of matches) {
    const tc = parseToolCallBlock(match.json, tools);
    if (tc) toolCalls.push(tc);
  }

  if (!toolCalls.length) return { text: content, toolCalls: null };

  // Extract text outside of tool_call blocks
  const textParts: string[] = [];
  let lastEnd = 0;
  for (const match of matches) {
    const before = content.slice(lastEnd, match.start);
    if (before.trim()) textParts.push(before.trim());
    lastEnd = match.end;
  }
  const trailing = content.slice(lastEnd);
  if (trailing.trim()) textParts.push(trailing.trim());

  const textContent = textParts.length ? textParts.join("\n") : null;
  return { text: textContent, toolCalls };
}

export function formatToolHistory(messages: OpenAIMessage[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];
  for (const msg of messages) {
    const role = msg.role ?? "";
    const content = msg.content;
    const toolCalls = msg.tool_calls;
    const toolCallId = msg.tool_call_id;
    const name = msg.name;

    if (role === "assistant" && toolCalls && toolCalls.length) {
      const parts: string[] = [];
      if (content) {
        parts.push(typeof content === "string" ? content : String(content));
      }
      for (const tc of toolCalls) {
        const func = tc.function ?? {};
        const tcName = func.name ?? "";
        const tcArgs = func.arguments ?? "{}";
        parts.push(`<tool_call>{"name":"${tcName}","arguments":${tcArgs}}</tool_call>`);
      }
      result.push({ role: "assistant", content: parts.join("\n") });
    } else if (role === "tool") {
      const toolName = name || "unknown";
      const callId = toolCallId || "";
      const contentStr = typeof content === "string" ? content : content ? JSON.stringify(content) : "";
      result.push({ role: "user", content: `tool (${toolName}, ${callId}): ${contentStr}` });
    } else {
      result.push(msg);
    }
  }
  return result;
}
