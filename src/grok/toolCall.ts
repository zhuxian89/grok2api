/**
 * Enhanced tool call utilities for OpenAI-compatible function calling.
 *
 * Ported from cc-proxy approach: random Unicode delimiters, state machine parser,
 * comprehensive JSON repair, and fuzzy matching fallback.
 */

// ==================== Types ====================

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

export interface DelimiterMarkers {
  TC_START: string;
  TC_END: string;
  NAME_START: string;
  NAME_END: string;
  ARGS_START: string;
  ARGS_END: string;
  RESULT_START: string;
  RESULT_END: string;
}

// ==================== Random Delimiter Generation ====================

const DELIMITER_SETS = [
  { open: "\u0F12", close: "\u0F12", mid: "\u0FC7" },
  { open: "\uA9C1", close: "\uA9C2", mid: "\u0FD4" },
  { open: "\u1392", close: "\u1392", mid: "\u1393" },
  { open: "\uA188", close: "\uA188", mid: "\uA2B0" },
  { open: "\uAA5C", close: "\uAA5C", mid: "\uAA5F" },
  { open: "\uA4F8", close: "\uA4F8", mid: "\uA4F9" },
];

const SUFFIX_POOL = [
  "\u9F98", "\u9750", "\u9F49", "\u9EA4", "\u7228",
  "\u99EB", "\u9C7B", "\u7FB4", "\u7287", "\u9A89",
  "\u98DD", "\u5375", "\u9747", "\u98CD", "\u99AB",
  "\u7065", "\u53BD", "\u53D2", "\u53D5", "\u82B4",
];

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]!;
}

export function createDelimiter(): DelimiterMarkers {
  const set = pickRandom(DELIMITER_SETS);
  const suffix1 = pickRandom(SUFFIX_POOL);
  let suffix2 = pickRandom(SUFFIX_POOL);
  while (suffix2 === suffix1 && SUFFIX_POOL.length > 1) {
    suffix2 = pickRandom(SUFFIX_POOL);
  }
  return {
    TC_START: `${set.open}${suffix1}\u1405`,
    TC_END: `\u140A${suffix1}${set.close}`,
    NAME_START: `${set.mid}\u25B8`,
    NAME_END: `\u25C2${set.mid}`,
    ARGS_START: `${set.mid}\u25B9`,
    ARGS_END: `\u25C3${set.mid}`,
    RESULT_START: `${set.open}${suffix2}\u27EB`,
    RESULT_END: `\u27EA${suffix2}${set.close}`,
  };
}

// ==================== Tool Prompt Builder ====================

function escapeXml(text: string): string {
  return text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function buildToolsXml(tools: ToolDefinition[]): string {
  if (!tools.length) return "<function_list>None</function_list>";
  const items = tools.map((tool, index) => {
    if (tool.type !== "function") return "";
    const func = tool.function ?? {};
    const name = func.name ?? "";
    const desc = func.description ?? "None";
    const params = func.parameters ?? {};
    const props = (params.properties ?? {}) as Record<string, Record<string, unknown>>;
    const required = (params.required ?? []) as string[];

    const paramLines = Object.entries(props).map(([pName, pInfo]) => {
      const type = pInfo.type ?? "any";
      const pDesc = pInfo.description ?? "";
      const req = required.includes(pName);
      const enumVals = pInfo.enum ? JSON.stringify(pInfo.enum) : "";
      return [
        `    <parameter name="${pName}">`,
        `      <type>${type}</type>`,
        `      <required>${req}</required>`,
        pDesc ? `      <description>${escapeXml(String(pDesc))}</description>` : "",
        enumVals ? `      <enum>${escapeXml(enumVals)}</enum>` : "",
        "    </parameter>",
      ].filter(Boolean).join("\n");
    }).join("\n");

    const reqXml = required.length
      ? required.map((r) => `    <param>${r}</param>`).join("\n")
      : "    <param>None</param>";

    return [
      `  <tool id="${index + 1}">`,
      `    <name>${name}</name>`,
      `    <description>${escapeXml(desc)}</description>`,
      "    <required>",
      reqXml,
      "    </required>",
      paramLines ? `    <parameters>\n${paramLines}\n    </parameters>` : "    <parameters>None</parameters>",
      "  </tool>",
    ].join("\n");
  }).filter(Boolean).join("\n");
  return `<function_list>\n${items}\n</function_list>`;
}

// ==================== buildToolPrompt ====================

export function buildToolPrompt(
  tools: ToolDefinition[],
  toolChoice?: unknown,
  parallelToolCalls: boolean = true,
): { prompt: string; delimiter: DelimiterMarkers | null } {
  if (!tools || !tools.length) return { prompt: "", delimiter: null };
  if (toolChoice === "none") return { prompt: "", delimiter: null };

  const delimiter = createDelimiter();
  const m = delimiter;
  const toolsXml = buildToolsXml(tools);

  let template = `You are an intelligent assistant equipped with specific tools.
In this environment you have access to a set of tools you can use to answer the user's question.
When you need to use a tool, you MUST strictly follow the format below.

### 1. Available Tools
<tools>
${toolsXml}
</tools>

### 2. Response Strategy (Execute vs. Chat)
**MODE A: TOOL EXECUTION (Prioritize this for functionality)**
- **Trigger Condition:** If the request requires data fetching, file manipulation, calculation, or any action supported by your tools.
- **Behavior:** **BE SILENT AND ACT.** Do NOT explain what you are going to do.
- **Output:** Start immediately with the tool call block using the exact delimiters provided.
**MODE B: CONVERSATION (Only when tools are useless)**
- **Trigger Condition:** If the user is greeting, asking for general advice, or asking a question that tools cannot solve.
- **Behavior:** Respond naturally and helpfully in plain text.
- **Constraint:** Do NOT output any tool call delimiters or formatting in this mode.

### 3. How to call tools
${m.TC_START}
${m.NAME_START}function_name${m.NAME_END}
${m.ARGS_START}{"param": "value"}${m.ARGS_END}
${m.TC_END}

### 4. Strict Tool Implementation Rules
1. Tool calls MUST be at the END of your response.
2. Copy the delimiters EXACTLY as shown.
3. **Arguments must be valid JSON (PERFECT SYNTAX IS MANDATORY)**
4. One tool per block.
5. You may provide explanations or reasoning BEFORE the tool call block.
6. Once the tool call block starts, no other text may be added until the closing delimiter.
7. After the closing delimiter, NO additional text may be added.

### 5. Delimiters to use for this session:
- TC_START: ${m.TC_START}
- TC_END: ${m.TC_END}
- NAME_START: ${m.NAME_START}
- NAME_END: ${m.NAME_END}
- ARGS_START: ${m.ARGS_START}
- ARGS_END: ${m.ARGS_END}`;

  if (toolChoice === "required") {
    template += "\n\nIMPORTANT: You MUST call at least one tool in your response. Do not respond with only text.";
  } else if (typeof toolChoice === "object" && toolChoice !== null) {
    const funcInfo = (toolChoice as Record<string, any>).function ?? {};
    const forcedName = funcInfo.name ?? "";
    if (forcedName) {
      template += `\n\nIMPORTANT: You MUST call the tool "${forcedName}" in your response.`;
    }
  }

  if (parallelToolCalls) {
    template += "\n\nYou may make multiple tool calls by using multiple tool call blocks.";
  }

  return { prompt: template, delimiter };
}

// ==================== Enhanced JSON Repair ====================

function repairJson(str: string): string {
  let fixed = str.trim();

  // 1. Extract first { to last }
  const firstBrace = fixed.indexOf("{");
  const lastBrace = fixed.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    fixed = fixed.slice(firstBrace, lastBrace + 1);
  }

  // 2. Remove trailing commas
  fixed = fixed.replace(/,\s*([}\]])/g, "$1");

  // 3. Escape unescaped newlines inside strings
  fixed = fixed.replace(/(".*?[^\\]")|(\n)/g, (_match, group1, group2) => {
    if (group2) return "\\n";
    return group1;
  });

  // 4. Fix quoted booleans/null: "true" -> true, "false" -> false, "null" -> null
  fixed = fixed.replace(/:[ \t]*"(true|false|null)"/gi, (_match, val) => {
    return `: ${val.toLowerCase()}`;
  });

  // 5. Auto-complete brackets
  const stack: ("{" | "[")[] = [];
  for (let i = 0; i < fixed.length; i++) {
    const ch = fixed[i];
    if (ch === "{") stack.push("{");
    else if (ch === "[") stack.push("[");
    else if (ch === "}") { if (stack[stack.length - 1] === "{") stack.pop(); }
    else if (ch === "]") { if (stack[stack.length - 1] === "[") stack.pop(); }
  }
  while (stack.length > 0) {
    const open = stack.pop();
    fixed += open === "{" ? "}" : "]";
  }

  return fixed;
}

function tryParseJson(str: string): unknown | null {
  if (!str) return {};
  try {
    return JSON.parse(str);
  } catch {
    const repaired = repairJson(str);
    try {
      return JSON.parse(repaired);
    } catch {
      // Last resort: replace all control chars
      try {
        const lastResort = repaired.replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
        return JSON.parse(lastResort);
      } catch {
        return null;
      }
    }
  }
}

function generateCallId(): string {
  const hex = crypto.randomUUID().replace(/-/g, "").slice(0, 24);
  return `call_${hex}`;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ==================== State Machine Parser ====================

export type ParserEventType = "text" | "tool_call" | "tool_call_failed" | "end";

export interface ParserEvent {
  type: ParserEventType;
  content?: string;
  call?: { name: string; arguments: unknown };
  reason?: string;
}

type ParserState = "TEXT" | "TOOL";

export class ToolifyParser {
  private readonly markers: DelimiterMarkers;
  private state: ParserState = "TEXT";
  private buffer = "";
  private toolBuffer = "";
  private readonly events: ParserEvent[] = [];

  constructor(markers: DelimiterMarkers) {
    this.markers = markers;
  }

  feed(chunk: string): void {
    this.buffer += chunk;
    this.processBuffer();
  }

  finish(): void {
    if (this.state === "TOOL") {
      this.toolBuffer += this.buffer;
      this.parseAndEmitToolCall();
    } else if (this.buffer) {
      this.events.push({ type: "text", content: this.buffer });
    }
    this.events.push({ type: "end" });
    this.state = "TEXT";
    this.buffer = "";
    this.toolBuffer = "";
  }

  consumeEvents(): ParserEvent[] {
    return this.events.splice(0, this.events.length);
  }

  private processBuffer(): void {
    const m = this.markers;

    if (this.state === "TOOL") {
      if (this.buffer.includes(m.TC_END)) {
        const idx = this.buffer.indexOf(m.TC_END) + m.TC_END.length;
        this.toolBuffer += this.buffer.slice(0, idx);
        this.parseAndEmitToolCall();
        this.state = "TEXT";
        const remaining = this.buffer.slice(idx);
        this.buffer = "";
        if (remaining) {
          this.buffer = remaining;
          this.processBuffer();
        }
      }
      return;
    }

    // TEXT state: look for TC_START
    if (this.buffer.includes(m.TC_START)) {
      const idx = this.buffer.indexOf(m.TC_START);
      const textBefore = this.buffer.slice(0, idx);
      if (textBefore) {
        this.events.push({ type: "text", content: textBefore });
      }
      this.state = "TOOL";
      this.toolBuffer = "";
      const remaining = this.buffer.slice(idx);
      this.buffer = "";
      if (remaining) {
        this.buffer = remaining;
        this.processBuffer();
      }
      return;
    }

    // Flush safe text if buffer is large (keep tail for partial marker match)
    const maxMarkerLen = m.TC_START.length;
    if (this.buffer.length > 512) {
      const safeLen = this.buffer.length - maxMarkerLen;
      const safeText = this.buffer.slice(0, safeLen);
      this.events.push({ type: "text", content: safeText });
      this.buffer = this.buffer.slice(safeLen);
    }
  }

  private parseAndEmitToolCall(): void {
    const m = this.markers;
    const content = this.toolBuffer;

    // 1. Try regex matching
    const regex = new RegExp(
      `${escapeRegex(m.TC_START)}[\\s\\S]*?` +
      `${escapeRegex(m.NAME_START)}\\s*([\\s\\S]*?)\\s*${escapeRegex(m.NAME_END)}[\\s\\S]*?` +
      `${escapeRegex(m.ARGS_START)}\\s*([\\s\\S]*?)\\s*${escapeRegex(m.ARGS_END)}[\\s\\S]*?` +
      `${escapeRegex(m.TC_END)}`,
      "g",
    );

    let name = "";
    let argsStr = "";
    const match = regex.exec(content);

    if (match) {
      name = (match[1] ?? "").trim();
      argsStr = (match[2] ?? "").trim();
    } else {
      // 2. Fuzzy marker matching fallback
      const nStart = content.indexOf(m.NAME_START);
      const nEnd = nStart >= 0 ? content.indexOf(m.NAME_END, nStart + m.NAME_START.length) : -1;
      const aStart = nEnd >= 0 ? content.indexOf(m.ARGS_START, nEnd + m.NAME_END.length) : -1;
      const aEnd = aStart >= 0 ? content.indexOf(m.ARGS_END, aStart + m.ARGS_START.length) : -1;

      if (nStart !== -1 && nEnd !== -1 && aStart !== -1 && aEnd !== -1) {
        name = content.slice(nStart + m.NAME_START.length, nEnd).trim();
        argsStr = content.slice(aStart + m.ARGS_START.length, aEnd).trim();
      }
    }

    if (name) {
      const args = tryParseJson(argsStr);
      if (args !== null) {
        this.events.push({ type: "tool_call", call: { name, arguments: args } });
        this.toolBuffer = "";
        return;
      }
    }

    // Failed to parse
    const reason = content.includes(m.TC_END) ? "malformed_json" : "incomplete_delimiter";
    this.events.push({ type: "tool_call_failed", content, reason });
    this.toolBuffer = "";
  }
}

// ==================== Non-stream parsing (batch) ====================

export function parseToolCalls(
  content: string,
  delimiter: DelimiterMarkers,
  tools?: ToolDefinition[],
): { text: string | null; toolCalls: ToolCall[] | null } {
  if (!content) return { text: content, toolCalls: null };

  const parser = new ToolifyParser(delimiter);
  parser.feed(content);
  parser.finish();

  const events = parser.consumeEvents();
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];

  const validNames = new Set<string>();
  if (tools?.length) {
    for (const t of tools) {
      const n = t.function?.name;
      if (n) validNames.add(n);
    }
  }

  for (const ev of events) {
    if (ev.type === "text" && ev.content) {
      textParts.push(ev.content);
    } else if (ev.type === "tool_call" && ev.call) {
      const name = ev.call.name;
      if (validNames.size && !validNames.has(name)) continue;
      const argsStr = typeof ev.call.arguments === "string"
        ? ev.call.arguments
        : JSON.stringify(ev.call.arguments);
      toolCalls.push({
        id: generateCallId(),
        type: "function",
        function: { name, arguments: argsStr },
      });
    } else if (ev.type === "tool_call_failed" && ev.content) {
      // Treat failed tool call content as text
      textParts.push(ev.content);
    }
  }

  if (!toolCalls.length) return { text: content, toolCalls: null };

  const textContent = textParts.join("").trim() || null;
  return { text: textContent, toolCalls };
}

// ==================== Tool History Formatting ====================

export function formatToolHistory(
  messages: OpenAIMessage[],
  delimiter?: DelimiterMarkers,
): OpenAIMessage[] {
  const m = delimiter;
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
        if (m) {
          parts.push(`${m.TC_START}\n${m.NAME_START}${tcName}${m.NAME_END}\n${m.ARGS_START}${tcArgs}${m.ARGS_END}\n${m.TC_END}`);
        } else {
          parts.push(`<tool_call>{"name":"${tcName}","arguments":${tcArgs}}</tool_call>`);
        }
      }
      result.push({ role: "assistant", content: parts.join("\n") });
    } else if (role === "tool") {
      const toolName = name || "unknown";
      const callId = toolCallId || "";
      const contentStr = typeof content === "string" ? content : content ? JSON.stringify(content) : "";
      if (m) {
        result.push({ role: "user", content: `${m.RESULT_START}[ID: ${callId}] (${toolName})\n${contentStr}${m.RESULT_END}` });
      } else {
        result.push({ role: "user", content: `tool (${toolName}, ${callId}): ${contentStr}` });
      }
    } else {
      result.push(msg);
    }
  }
  return result;
}
