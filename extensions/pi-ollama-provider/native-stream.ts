/**
 * Ollama native /api/chat streaming implementation.
 *
 * Uses NDJSON (newline-delimited JSON) instead of SSE.
 * This fixes the tool-calling bug (ollama/ollama#12557) where the
 * OpenAI compat shim silently drops tool_calls in streaming mode.
 *
 * Key features:
 * - Correct tool call delivery in streaming
 * - Always sets num_ctx from model data (avoids 4096 silent truncation)
 * - Ghost-token retry (empty stream with eval_count > 0)
 * - Ollama-specific options (temperature, top_p, top_k, keep_alive, etc.)
 * - Overflow detection (Ollama 400 errors)
 * - Fallback to OpenAI compat /v1/chat/completions if native fails
 */

import type { AssistantMessageEventStream } from "@mariozechner/pi-ai";
import { Readable } from "node:stream";
import { logWarn } from "./log-buffer.js";

// ── types ──

/** Parsed NDJSON chunk from /api/chat */
export interface OllamaChatChunk {
  model: string;
  created_at: string;
  message?: {
    role: string;
    content?: string;
    tool_calls?: OllamaToolCall[];
    thinking?: string;
  };
  done?: boolean;
  done_reason?: string;
  total_duration?: number;
  eval_count?: number;
  prompt_eval_count?: number;
  eval_duration?: number;
  prompt_eval_duration?: number;
  error?: string;
}

export interface OllamaToolCall {
  function: {
    name: string;
    arguments: Record<string, unknown>;
  };
}

/** Request body for /api/chat */
export interface OllamaChatRequest {
  model: string;
  messages: OllamaChatMessage[];
  stream?: boolean;
  tools?: OllamaToolDefinition[];
  options?: OllamaOptions;
  keep_alive?: string;
  format?: string | object;
}

export interface OllamaChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  images?: string[];
  tool_calls?: OllamaToolCall[];
  tool_call_id?: string;
}

export interface OllamaToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

export interface OllamaOptions {
  num_ctx?: number;
  num_predict?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  stop?: string[];
  seed?: number;
  [key: string]: unknown;
}

// ── NDJSON parser ──

/**
 * Parse an NDJSON stream from a ReadableStream<Uint8Array>.
 * Yields parsed JSON objects line by line.
 */
export async function* parseNDJSON(
  body: ReadableStream<Uint8Array> | NodeJS.ReadableStream | null,
): AsyncGenerator<OllamaChatChunk> {
  if (!body) return;

  const decoder = new TextDecoder();
  let buffer = "";

  // Handle both browser ReadableStream and Node.js Readable
  if (body instanceof Readable || ("read" in body && typeof (body as any).read === "function")) {
    // Node.js Readable
    const nodeStream = body as NodeJS.ReadableStream;
    for await (const chunk of nodeStream) {
      buffer += decoder.decode(chunk as Uint8Array, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          yield JSON.parse(line);
        } catch {
          // Skip malformed lines
        }
      }
    }
  } else {
    // Browser ReadableStream
    const reader = (body as ReadableStream<Uint8Array>).getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop()!;
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            yield JSON.parse(line);
          } catch {
            // Skip malformed lines
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer);
    } catch {
      // Ignore trailing malformed data
    }
  }
}

// ── message conversion ──

/**
 * Convert pi's message format to Ollama's native /api/chat format.
 * - Maps `developer` role → `system` (Ollama doesn't support developer)
 * - Converts tool results to Ollama's format
 * - Strips images for non-vision messages
 */
export function convertMessages(
  messages: Array<Record<string, unknown>>,
  modelSupportsVision: boolean = false,
): OllamaChatMessage[] {
  return messages.map((msg) => {
    const role = String(msg.role ?? "user");

    // Convert developer → system (Ollama compat)
    const ollamaRole = role === "developer" ? "system" : role as OllamaChatMessage["role"];

    // Handle content
    let content: string | null = null;
    const images: string[] = [];

    if (typeof msg.content === "string") {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      // Multi-part content (text + images)
      const textParts: string[] = [];
      for (const part of msg.content as Array<Record<string, unknown>>) {
        if (part.type === "text" && typeof part.text === "string") {
          textParts.push(part.text);
        } else if (
          part.type === "image_url" &&
          modelSupportsVision &&
          typeof part.image_url === "object" &&
          part.image_url !== null
        ) {
          const url = String((part.image_url as Record<string, unknown>).url ?? "");
          // Extract base64 from data URI
          if (url.startsWith("data:")) {
            const base64 = url.split(",")[1];
            if (base64) images.push(base64);
          }
        }
      }
      content = textParts.join("\n") || null;
    }

    // Handle tool_calls from assistant messages
    const toolCalls: OllamaToolCall[] | undefined = Array.isArray(msg.tool_calls)
      ? (msg.tool_calls as OllamaToolCall[])
      : undefined;

    // Handle tool results
    if (role === "tool" || role === "toolResult" || role === "ipython") {
      const toolCallId = String(msg.tool_call_id ?? msg.toolCallId ?? "");
      // Extract clean text from pi's content block format [{type:"text", text:"..."}, ...]
      let resultContent: string;
      if (typeof msg.content === "string") {
        resultContent = msg.content;
      } else if (Array.isArray(msg.content)) {
        resultContent = (msg.content as Array<Record<string, unknown>>)
          .filter((b) => b.type === "text" && typeof b.text === "string")
          .map((b) => b.text as string)
          .join("\n");
      } else {
        resultContent = "";
      }
      return {
        role: "tool" as const,
        content: resultContent,
        tool_call_id: toolCallId,
      };
    }

    return {
      role: ollamaRole,
      content,
      ...(images.length > 0 ? { images } : {}),
      ...(toolCalls ? { tool_calls: toolCalls } : {}),
    };
  });
}

// ── tool conversion ──

/**
 * Convert pi's tool definitions to Ollama's native format.
 */
export function convertTools(
  tools: Array<Record<string, unknown>> | undefined,
): OllamaToolDefinition[] {
  if (!tools || tools.length === 0) return [];

  return tools.map((t) => {
    // Handle Ollama/OpenAI format: { type: "function", function: { name, description, parameters } }
    if (t.type === "function" && t.function) {
      return {
        type: "function" as const,
        function: {
          name: String((t.function as Record<string, unknown>).name ?? ""),
          description: String((t.function as Record<string, unknown>).description ?? ""),
          parameters: ((t.function as Record<string, unknown>).parameters ?? {}) as Record<string, unknown>,
        },
      };
    }
    // Handle pi's agent-level format: { name, description, parameters } (built-in tools)
    return {
      type: "function" as const,
      function: {
        name: String((t as Record<string, unknown>).name ?? ""),
        description: String((t as Record<string, unknown>).description ?? ""),
        parameters: ((t as Record<string, unknown>).parameters ?? {}) as Record<string, unknown>,
      },
    };
  });
}

// ── block merging ──

/**
 * Merge adjacent same-type content blocks and handle interleaved
 * thinking blocks between text blocks.
 *
 * GLM outputs thinking tokens in a long contiguous block, then text.
 * But occasionally it emits a trailing thinking fragment after text
 * has already started (e.g. text "I appreciate", thinking ".",
 * text " the request!"). This creates fragmented multi-block responses.
 *
 * Three passes:
 * 1. Merge adjacent same-type blocks (text+text, thinking+thinking)
 * 2. Merge sandwiched thinking blocks into an adjacent thinking block
 *    or keep them as-is (never drop thinking content)
 * 3. Re-merge any adjacent same-type blocks created by pass 2
 */
export function mergeAdjacentBlocks(blocks: any[]): void {
  // Pass 1: merge adjacent same-type blocks
  for (let i = blocks.length - 1; i > 0; i--) {
    const curr = blocks[i];
    const prev = blocks[i - 1];
    if (!curr || !prev) continue;
    if (prev.type === "text" && curr.type === "text") {
      prev.text = (prev.text || "") + (curr.text || "");
      blocks.splice(i, 1);
    } else if (prev.type === "thinking" && curr.type === "thinking") {
      prev.thinking = (prev.thinking || "") + (curr.thinking || "");
      blocks.splice(i, 1);
    }
  }

  // Pass 2: handle thinking blocks sandwiched between text blocks
  // Instead of dropping thinking content, merge it into an existing
  // thinking block or keep it as-is.
  for (let i = blocks.length - 1; i > 0; i--) {
    const curr = blocks[i];
    const prev = blocks[i - 1];
    if (!curr || !prev) continue;
    if (prev.type === "text" && curr.type === "thinking") {
      const next = blocks[i + 1];
      if (next && next.type === "text") {
        // GLM sometimes emits: text "I appreciate" + thinking "." + text " the request!"
        // This is a fragmented sentence — merge into one text block,
        // but PRESERVE the thinking content by appending to an adjacent thinking block.
        const thinkingContent = curr.thinking || "";

        // Look for a nearby thinking block to merge into
        const prevThinking = i >= 2 && blocks[i - 2]?.type === "thinking" ? blocks[i - 2] : null;
        if (prevThinking && thinkingContent) {
          // Merge thinking into the previous thinking block
          prevThinking.thinking = (prevThinking.thinking || "") + thinkingContent;
        } else {
          // No adjacent thinking block — convert the sandwiched thinking into
          // a newline + content within the merged text (better than dropping it)
          if (thinkingContent.trim()) {
            prev.text = (prev.text || "") + "\n" + thinkingContent + "\n";
          }
        }

        // Merge the two text blocks, removing the thinking block
        prev.text = (prev.text || "") + (next.text || "");
        blocks.splice(i, 2); // remove thinking + next text
      }
    }
  }

  // Pass 3: re-merge any adjacent same-type blocks created by pass 2
  for (let i = blocks.length - 1; i > 0; i--) {
    const curr = blocks[i];
    const prev = blocks[i - 1];
    if (!curr || !prev) continue;
    if (prev.type === "thinking" && curr.type === "thinking") {
      prev.thinking = (prev.thinking || "") + (curr.thinking || "");
      blocks.splice(i, 1);
    }
  }
}

// ── thinking-text redirection ──

/**
 * Patterns that indicate a text block is actually internal thinking/reasoning
 * that the model incorrectly emitted as regular content.
 *
 * GLM-5.x models with thinking enabled sometimes output brief planning
 * statements ("Let me search...", "Now I'll check...", "Good, the build passed.")
 * as `message.content` instead of `message.thinking`. These appear as visible
 * text in the TUI, creating noisy "thinking literals".
 *
 * This heuristic redirects such text blocks into thinking blocks when:
 * 1. The model has a thinking block in the same response
 * 2. The text block is short (< threshold characters)
 * 3. The text matches common thinking-literal patterns
 */
const THINKING_LITERAL_PATTERNS = [
  /^Let me\b/i,
  /^Now let\b/i,
  /^Now I\b/i,
  /^Good[,—\s]/i,
  /^OK[,—\.\s]/i,
  /^Alright[,—\s]/i,
  /^Hmm[,\s]/i,
  /^I (?:need|should|will|can|'ll|would)\b/i,
  /^I'll\b/i,
  /^I've\b/i,
  /^Let's\b/i,
  /^First,?\s/i,
  /^Next,?\s/i,
  /^The user\b/i,
  /^Wait[,\s]/i,
  /^Actually[,\s]/i,
  /^I see[—,\s\.]/i,
  /^I (?:also|already|still)\b/i,
  /^This (?:looks|seems|appears)\b/i,
];

/** Maximum text length (in chars) to consider for thinking redirection. */
const THINKING_LITERAL_MAX_LENGTH = 200;

/**
 * Check if a text block looks like internal thinking/reasoning content.
 * Returns true if the text matches common thinking-literal patterns and is short.
 */
function looksLikeThinkingLiteral(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed || trimmed.length > THINKING_LITERAL_MAX_LENGTH) return false;
  return THINKING_LITERAL_PATTERNS.some((p) => p.test(trimmed));
}

/**
 * Redirect text blocks that look like internal thinking into thinking blocks.
 *
 * When a model is in thinking mode and emits short planning statements as
 * regular text (instead of in the `thinking` field), this function detects
 * those statements and moves their content into thinking blocks.
 *
 * Strategy:
 * - If there's an existing thinking block in the response, merge into it
 * - If there's no existing thinking block, convert the text block to a thinking block
 * - Only redirects text blocks that match thinking-literal patterns
 * - Preserves text blocks that don't match patterns (likely real user-facing content)
 *
 * @param blocks The content blocks array (mutated in place)
 * @param hasThinkingMode Whether the model was called with thinking enabled
 */
export function redirectThinkingText(blocks: any[], hasThinkingMode: boolean): void {
  if (!hasThinkingMode || blocks.length === 0) return;

  // Find if there are any existing thinking blocks
  const hasThinkingBlock = blocks.some((b) => b.type === "thinking");

  // Check which text blocks look like thinking literals
  const textIndicesToRedirect: number[] = [];
  for (let i = 0; i < blocks.length; i++) {
    if (blocks[i].type === "text" && looksLikeThinkingLiteral(blocks[i].text || "")) {
      textIndicesToRedirect.push(i);
    }
  }

  if (textIndicesToRedirect.length === 0) return;

  // Redirect each thinking-literal text block
  // Process in reverse to keep indices stable
  for (let i = textIndicesToRedirect.length - 1; i >= 0; i--) {
    const idx = textIndicesToRedirect[i];
    const block = blocks[idx];
    const text = (block.text || "").trim();

    if (hasThinkingBlock) {
      // Merge into the nearest thinking block
      // Prefer the one just before this text block, otherwise the first thinking block
      let targetIdx = -1;
      for (let j = idx - 1; j >= 0; j--) {
        if (blocks[j].type === "thinking") {
          targetIdx = j;
          break;
        }
      }
      if (targetIdx === -1) {
        // No thinking block before — find the first one
        for (let j = 0; j < blocks.length; j++) {
          if (blocks[j].type === "thinking") {
            targetIdx = j;
            break;
          }
        }
      }

      if (targetIdx >= 0) {
        // Append to the target thinking block
        blocks[targetIdx].thinking = (blocks[targetIdx].thinking || "") + "\n" + text;
        // Remove the text block
        blocks.splice(idx, 1);
      }
    } else {
      // No thinking block exists — convert this text block into a thinking block
      block.type = "thinking";
      block.thinking = text;
      delete block.text;
    }
  }

  // Final pass: merge adjacent thinking blocks that may have been created
  for (let i = blocks.length - 1; i > 0; i--) {
    const curr = blocks[i];
    const prev = blocks[i - 1];
    if (prev.type === "thinking" && curr.type === "thinking") {
      prev.thinking = (prev.thinking || "") + "\n" + (curr.thinking || "");
      blocks.splice(i, 1);
    }
  }
}

// ── ghost-token retry ──

/**
 * Detect a "ghost token" scenario: the stream completed without
 * producing any content, but eval_count > 0 means the model did
 * generate tokens. This happens when Ollama's streaming drops the
 * tool call content (ollama#12557 variant).
 */
export function isGhostTokenStream(chunks: OllamaChatChunk[]): boolean {
  if (chunks.length === 0) return false;

  const finalChunk = chunks[chunks.length - 1];
  if (!finalChunk.done) return false;

  // Model ran inference but we got no content or tool calls
  const hasContent = chunks.some(
    (c) => c.message?.content && c.message.content.trim().length > 0,
  );
  const hasToolCalls = chunks.some(
    (c) => c.message?.tool_calls && c.message.tool_calls.length > 0,
  );
  const hasEvalCount = finalChunk.eval_count != null && finalChunk.eval_count > 0;

  return !hasContent && !hasToolCalls && hasEvalCount;
}

// ── overflow detection ──

/** Ollama context overflow error patterns */
export const OLLAMA_OVERFLOW_PATTERNS = [
  /exceeded max context length/i,
  /prompt too long/i,
  /context window exceeded/i,
  /maximum context length exceeded/i,
] as const;

/**
 * Check if an Ollama error indicates context overflow.
 * Ollama returns 400 with messages like:
 *   "prompt too long; exceeded max context length by X tokens"
 */
export function isOllamaContextOverflow(error: unknown): boolean {
  if (!error) return false;
  const message =
    error instanceof Error ? error.message : String(error);
  return OLLAMA_OVERFLOW_PATTERNS.some((p) => p.test(message));
}

// ── main stream function ──

export interface StreamNativeOptions {
  baseUrl: string;
  apiKey?: string;
  model: string;
  contextWindow: number;
  messages: Array<Record<string, unknown>>;
  tools?: Array<Record<string, unknown>>;
  modelSupportsVision?: boolean;
  ollamaOptions?: OllamaOptions;
  signal?: AbortSignal;
  keepAlive?: string;
  /** Whether the model has reasoning/thinking capability and is in thinking mode. */
  hasThinkingMode?: boolean;
}

/**
 * Stream from Ollama's native /api/chat endpoint.
 *
 * Returns an AssistantMessageEventStream that pi's agent loop can consume.
 * Handles: text deltas, thinking deltas, tool calls, usage data, overflow errors,
 * ghost-token retry, and truncation detection.
 */
export async function streamNativeChat(
  stream: AssistantMessageEventStream,
  options: StreamNativeOptions,
): Promise<void> {
  const {
    baseUrl,
    apiKey,
    model,
    contextWindow,
    messages,
    tools,
    modelSupportsVision,
    ollamaOptions,
    signal,
    keepAlive,
  } = options;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey && apiKey !== "ollama") {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  // Always set num_ctx — prevents Ollama's 4096 silent truncation
  const resolvedOptions: OllamaOptions = {
    num_ctx: contextWindow,
    ...ollamaOptions,
  };

  const body: OllamaChatRequest = {
    model,
    messages: convertMessages(messages, modelSupportsVision),
    stream: true,
    ...(tools && tools.length > 0 ? { tools: convertTools(tools) } : {}),
    options: resolvedOptions,
    ...(keepAlive ? { keep_alive: keepAlive } : {}),
  };

  // Build the output message that will be updated
  const output: any = {
    role: "assistant",
    content: [],
    api: "ollama-native",
    provider: "ollama",
    model: model,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: Date.now(),
  };

  // Track content indices for proper delta ordering
  let toolCallIndex = 0;
  let hasToolCalls = false;

  // Track open blocks for proper delta handling (keep blocks open for incremental updates)
  let activeTextBlock: { index: number; content: string } | null = null;
  let activeThinkingBlock: { index: number; content: string } | null = null;

  // Collect chunks for ghost-token detection
  const chunks: OllamaChatChunk[] = [];

  try {
    const response = await fetch(`${baseUrl}/api/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => "");

      // Check for context overflow
      if (response.status === 400 && isOllamaContextOverflow(errorText)) {
        throw new Error(
          `[ollama] Context overflow: ${errorText}. ` +
          `num_ctx=${resolvedOptions.num_ctx}. Consider using a model with a larger context window.`,
        );
      }

      throw new Error(
        `[ollama] /api/chat HTTP ${response.status}: ${errorText.slice(0, 500)}`,
      );
    }

    if (!response.body) {
      throw new Error("[ollama] /api/chat returned empty body");
    }

    // Push start event
    stream.push({ type: "start", partial: output });

    for await (const chunk of parseNDJSON(response.body as ReadableStream<Uint8Array>)) {
      if (signal?.aborted) break;

      // Check for error in chunk
      if (chunk.error) {
        if (isOllamaContextOverflow(chunk.error)) {
          throw new Error(
            `[ollama] Context overflow: ${chunk.error}. ` +
            `num_ctx=${resolvedOptions.num_ctx}.`,
          );
        }
        throw new Error(`[ollama] API error: ${chunk.error}`);
      }

      chunks.push(chunk);

      // Process message content
      if (chunk.message) {
        // Text content
        if (chunk.message.content) {
          // Close any open thinking block before text
          if (activeThinkingBlock) {
            stream.push({ type: "thinking_end", contentIndex: activeThinkingBlock.index, content: activeThinkingBlock.content, partial: output });
            activeThinkingBlock = null;
          }

          // Start text block if not already active
          if (!activeTextBlock) {
            const idx = output.content.length;
            activeTextBlock = { index: idx, content: "" };
            output.content.push({ type: "text", text: "" });
            stream.push({ type: "text_start", contentIndex: idx, partial: output });
          }

          // Accumulate and emit delta
          activeTextBlock.content += chunk.message.content;
          output.content[activeTextBlock.index].text = activeTextBlock.content;
          stream.push({ type: "text_delta", contentIndex: activeTextBlock.index, delta: chunk.message.content, partial: output });
        }

        // Thinking content
        if (chunk.message.thinking) {
          // Close any open text block before thinking
          if (activeTextBlock) {
            stream.push({ type: "text_end", contentIndex: activeTextBlock.index, content: activeTextBlock.content, partial: output });
            activeTextBlock = null;
          }

          // Start thinking block if not already active
          if (!activeThinkingBlock) {
            const idx = output.content.length;
            activeThinkingBlock = { index: idx, content: "" };
            output.content.push({ type: "thinking", thinking: "" });
            stream.push({ type: "thinking_start", contentIndex: idx, partial: output });
          }

          // Accumulate and emit delta
          activeThinkingBlock.content += chunk.message.thinking;
          output.content[activeThinkingBlock.index].thinking = activeThinkingBlock.content;
          stream.push({ type: "thinking_delta", contentIndex: activeThinkingBlock.index, delta: chunk.message.thinking, partial: output });
        }

        // Tool calls — emit as a complete burst
        if (chunk.message.tool_calls && chunk.message.tool_calls.length > 0) {
          // Close any open text/thinking block
          if (activeTextBlock) {
            stream.push({ type: "text_end", contentIndex: activeTextBlock.index, content: activeTextBlock.content, partial: output });
            activeTextBlock = null;
          }
          if (activeThinkingBlock) {
            stream.push({ type: "thinking_end", contentIndex: activeThinkingBlock.index, content: activeThinkingBlock.content, partial: output });
            activeThinkingBlock = null;
          }

          for (const tc of chunk.message.tool_calls) {
            // Ollama may return arguments as a JSON string or as an object
            let args = tc.function.arguments;
            if (typeof args === "string") {
              try {
                args = JSON.parse(args);
              } catch {
                args = {};
              }
            }

            const toolCallIdx = output.content.length;
            const toolCallBlock = {
              type: "toolCall",
              id: `tool_${toolCallIndex}`,
              name: tc.function.name,
              arguments: args,
            };
            output.content.push(toolCallBlock);
            stream.push({ type: "toolcall_start", contentIndex: toolCallIdx, partial: output });

            // Ollama sends complete tool calls, so single delta + end
            const argsStr = JSON.stringify(args);
            stream.push({ type: "toolcall_delta", contentIndex: toolCallIdx, delta: argsStr, partial: output });
            stream.push({ type: "toolcall_end", contentIndex: toolCallIdx, toolCall: toolCallBlock, partial: output });

            hasToolCalls = true;
            toolCallIndex++;
          }
        }
      }

      // Final chunk — close blocks and emit usage
      if (chunk.done) {
        if (activeTextBlock) {
          stream.push({ type: "text_end", contentIndex: activeTextBlock.index, content: activeTextBlock.content, partial: output });
          activeTextBlock = null;
        }
        if (activeThinkingBlock) {
          stream.push({ type: "thinking_end", contentIndex: activeThinkingBlock.index, content: activeThinkingBlock.content, partial: output });
          activeThinkingBlock = null;
        }

        output.usage.input = chunk.prompt_eval_count ?? 0;
        output.usage.output = chunk.eval_count ?? 0;
        output.usage.totalTokens = output.usage.input + output.usage.output;
        output.stopReason = hasToolCalls ? "toolUse" : (chunk.done_reason === "length" ? "length" : "stop");

        // Merge adjacent same-type blocks (GLM interleaves tiny thinking fragments)
        mergeAdjacentBlocks(output.content);

        // Redirect thinking-literal text into thinking blocks when model is in thinking mode
        redirectThinkingText(output.content, !!options.hasThinkingMode);

        stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
        stream.end();
      }
    }

    // Ghost-token detection: model generated tokens but we received nothing
    if (output.content.length === 0 && chunks.length > 0) {
      if (isGhostTokenStream(chunks)) {
        // Retry with stream: false
        logWarn("stream", "Ghost token detected — retrying with stream: false");
        await retryNonStreaming(baseUrl, headers, body, stream, output);
        return;
      }

      // Connection closed without done:true — truncation
      const lastChunk = chunks[chunks.length - 1];
      if (!lastChunk?.done) {
        throw new Error(
          "[ollama] Stream ended without done:true — possible truncation. " +
          "The model may have exceeded its context window.",
        );
      }
    }
  } catch (err) {
    if (signal?.aborted) {
      // Close any open blocks
      if (activeTextBlock) {
        output.content[activeTextBlock.index].text = activeTextBlock.content;
        activeTextBlock = null;
      }
      if (activeThinkingBlock) {
        output.content[activeThinkingBlock.index].thinking = activeThinkingBlock.content;
        activeThinkingBlock = null;
      }
      output.stopReason = "aborted";
      (output as any).errorMessage = "Request was aborted";
      stream.push({ type: "error", reason: "aborted", error: output });
      stream.end();
      return;
    }

    for (const block of output.content) {
      delete (block as any).partial;
    }
    output.stopReason = "error";
    (output as any).errorMessage = err instanceof Error ? err.message : String(err);
    stream.push({ type: "error", reason: "error", error: output });
    stream.end();
    throw err;
  }
}

/**
 * Retry a request with stream: false (workaround for ghost tokens).
 * Parses the complete response and pushes events to the stream.
 */
async function retryNonStreaming(
  baseUrl: string,
  headers: Record<string, string>,
  body: OllamaChatRequest,
  stream: AssistantMessageEventStream,
  output: any,
): Promise<void> {
  const nonStreamBody = { ...body, stream: false };

  const response = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(nonStreamBody),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => "");
    throw new Error(`[ollama] Non-stream retry failed: HTTP ${response.status}: ${errorText.slice(0, 500)}`);
  }

  const result: OllamaChatChunk = await response.json();

  if (result.error) {
    throw new Error(`[ollama] Non-stream retry error: ${result.error}`);
  }

  stream.push({ type: "start", partial: output });

  let toolCallIndex = 0;

  if (result.message) {
    // Text content
    if (result.message.content) {
      const idx = output.content.length;
      const textBlock = { type: "text", text: result.message.content };
      output.content.push(textBlock);
      stream.push({ type: "text_start", contentIndex: idx, partial: output });
      stream.push({ type: "text_delta", contentIndex: idx, delta: result.message.content, partial: output });
      stream.push({ type: "text_end", contentIndex: idx, content: result.message.content, partial: output });
    }

    // Thinking content
    if (result.message.thinking) {
      const idx = output.content.length;
      const thinkingBlock = { type: "thinking", thinking: result.message.thinking };
      output.content.push(thinkingBlock);
      stream.push({ type: "thinking_start", contentIndex: idx, partial: output });
      stream.push({ type: "thinking_delta", contentIndex: idx, delta: result.message.thinking, partial: output });
      stream.push({ type: "thinking_end", contentIndex: idx, content: result.message.thinking, partial: output });
    }

    // Tool calls
    if (result.message.tool_calls && result.message.tool_calls.length > 0) {
      for (const tc of result.message.tool_calls) {
        const idx = output.content.length;
        // Ollama may return arguments as a JSON string or as an object
        let args = tc.function.arguments;
        if (typeof args === "string") {
          try {
            args = JSON.parse(args);
          } catch {
            args = {};
          }
        }

        const toolCallBlock = {
          type: "toolCall",
          id: `tool_${toolCallIndex}`,
          name: tc.function.name,
          arguments: args,
        };
        output.content.push(toolCallBlock);
        stream.push({ type: "toolcall_start", contentIndex: idx, partial: output });
        stream.push({ type: "toolcall_delta", contentIndex: idx, delta: JSON.stringify(args), partial: output });
        stream.push({ type: "toolcall_end", contentIndex: idx, toolCall: toolCallBlock, partial: output });
        toolCallIndex++;
      }
    }
  }

  output.usage.input = result.prompt_eval_count ?? 0;
  output.usage.output = result.eval_count ?? 0;
  output.usage.totalTokens = output.usage.input + output.usage.output;

  const hasToolCalls = result.message?.tool_calls && result.message.tool_calls.length > 0;
  output.stopReason = hasToolCalls ? "toolUse" : "stop";

  // Merge adjacent same-type blocks
  mergeAdjacentBlocks(output.content);

  // Redirect thinking-literal text into thinking blocks when model is in thinking mode
  redirectThinkingText(output.content, !!options.hasThinkingMode);

  stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
  stream.end();
}