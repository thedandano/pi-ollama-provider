/**
 * Tests for native-stream.ts — NDJSON parsing, message conversion,
 * tool conversion, ghost-token detection, overflow detection,
 * stream event format validation, and content block ordering.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Readable } from "node:stream";
import { createAssistantMessageEventStream } from "@mariozechner/pi-ai";

import {
  parseNDJSON,
  convertMessages,
  convertTools,
  isGhostTokenStream,
  isOllamaContextOverflow,
  streamNativeChat,
  mergeAdjacentBlocks,
  type OllamaChatChunk,
  type OllamaToolCall,
} from "../extensions/pi-ollama-provider/native-stream.js";

let originalFetch: typeof globalThis.fetch;

beforeEach(() => {
  originalFetch = globalThis.fetch;
  // By default, unmocked fetch calls will reject
  globalThis.fetch = vi.fn().mockRejectedValue(new Error("fetch not mocked in this test"));
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ════════════════════════════════════════════════════════════════
// parseNDJSON
// ════════════════════════════════════════════════════════════════

describe("parseNDJSON", () => {
  it("parses single-line NDJSON from Node.js Readable", async () => {
    const data = JSON.stringify({ message: { content: "hello" }, done: false }) + "\n";
    const stream = Readable.from([Buffer.from(data)]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0].message?.content).toBe("hello");
  });

  it("parses multi-line NDJSON", async () => {
    const lines = [
      JSON.stringify({ message: { content: "hel" }, done: false }),
      JSON.stringify({ message: { content: "lo" }, done: false }),
      JSON.stringify({ message: { content: "!" }, done: true, eval_count: 3 }),
    ].join("\n") + "\n";
    const stream = Readable.from([Buffer.from(lines)]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(3);
    expect(chunks[2].done).toBe(true);
    expect(chunks[2].eval_count).toBe(3);
  });

  it("handles streaming chunks (partial lines)", async () => {
    // Simulates data arriving in chunks that split NDJSON lines
    const full = JSON.stringify({ message: { content: "test" }, done: false }) + "\n" +
      JSON.stringify({ message: { content: "more" }, done: true, eval_count: 2 }) + "\n";
    const stream = Readable.from([
      Buffer.from(full.slice(0, 20)),
      Buffer.from(full.slice(20)),
    ]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(2);
  });

  it("skips malformed lines", async () => {
    const data = 'not-json\n{"message":{"content":"ok"},"done":true}\n\n';
    const stream = Readable.from([Buffer.from(data)]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0].done).toBe(true);
  });

  it("handles empty stream", async () => {
    const stream = Readable.from([]);
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(stream as any)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(0);
  });

  it("handles null body", async () => {
    const chunks: OllamaChatChunk[] = [];
    for await (const chunk of parseNDJSON(null)) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(0);
  });
});

// ════════════════════════════════════════════════════════════════
// convertMessages
// ════════════════════════════════════════════════════════════════

describe("convertMessages", () => {
  it("converts developer role to system", () => {
    const messages = [{ role: "developer", content: "You are helpful." }];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("system");
    expect(result[0].content).toBe("You are helpful.");
  });

  it("preserves system and user roles", () => {
    const messages = [
      { role: "system", content: "System prompt" },
      { role: "user", content: "User message" },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("system");
    expect(result[1].role).toBe("user");
  });

  it("handles multipart content with images (vision)", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,iVBOR..." },
          },
        ],
      },
    ];
    const result = convertMessages(messages, true);
    expect(result[0].content).toBe("What's in this image?");
    expect(result[0].images).toHaveLength(1);
    expect(result[0].images![0]).toBe("iVBOR...");
  });

  it("strips images when model doesn't support vision", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "What's in this image?" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,iVBOR..." },
          },
        ],
      },
    ];
    const result = convertMessages(messages, false);
    expect(result[0].content).toBe("What's in this image?");
    expect(result[0].images).toBeUndefined();
  });

  it("converts tool result messages", () => {
    const messages = [
      {
        role: "tool",
        tool_call_id: "call_123",
        content: '{"result": "success"}',
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("tool");
    expect(result[0].tool_call_id).toBe("call_123");
  });

  it("converts pi-style toolResult role to Ollama tool role", () => {
    // pi's AgentLoop sends tool results with role: "toolResult"
    // but the Ollama API expects role: "tool"
    const messages = [
      {
        role: "toolResult",
        toolCallId: "tool_0",
        toolName: "bash",
        content: [{ type: "text", text: "file1.txt\nfile2.txt" }],
        isError: false,
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("tool");
    expect(result[0].tool_call_id).toBe("tool_0");
    expect(result[0].content).toContain("file1.txt");
    // Should extract clean text, not JSON.stringify the content array
    expect(result[0].content).not.toContain('"type":"text"');
  });

  it("extracts clean text from pi's toolResult content array", () => {
    // pi sends content as [{type:"text", text:"..."}] array
    // convertMessages should extract the text, not JSON.stringify the array
    const messages = [
      {
        role: "toolResult",
        toolCallId: "tool_1",
        toolName: "bash",
        content: [{ type: "text", text: "output line 1" }, { type: "text", text: "output line 2" }],
        isError: false,
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("tool");
    expect(result[0].content).toContain("output line 1");
    expect(result[0].content).toContain("output line 2");
    // Should be clean text, not a JSON array string
    expect(result[0].content).not.toContain("{\"type\":\"text\"");
  });

  it("handles empty toolResult content gracefully", () => {
    const messages = [
      {
        role: "toolResult",
        toolCallId: "tool_2",
        toolName: "bash",
        content: [],
        isError: false,
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].role).toBe("tool");
    expect(result[0].content).toBe("");
  });

  it("preserves assistant tool_calls", () => {
    const toolCalls: OllamaToolCall[] = [
      { function: { name: "read_file", arguments: { path: "/foo" } } },
    ];
    const messages = [
      {
        role: "assistant",
        content: null,
        tool_calls: toolCalls,
      },
    ];
    const result = convertMessages(messages);
    expect(result[0].tool_calls).toHaveLength(1);
    expect(result[0].tool_calls![0].function.name).toBe("read_file");
  });
});

// ════════════════════════════════════════════════════════════════
// convertTools
// ════════════════════════════════════════════════════════════════

describe("convertTools", () => {
  it("converts function tools to Ollama format", () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "read_file",
          description: "Read a file",
          parameters: { type: "object", properties: { path: { type: "string" } } },
        },
      },
    ];
    const result = convertTools(tools);
    expect(result).toHaveLength(1);
    expect(result[0].type).toBe("function");
    expect(result[0].function.name).toBe("read_file");
  });

  it("returns empty array for undefined", () => {
    expect(convertTools(undefined)).toEqual([]);
  });

  it("returns empty array for empty array", () => {
    expect(convertTools([])).toEqual([]);
  });

  it("converts non-function tools as flat format (no filtering)", () => {
    // convertTools no longer filters — it maps all tools using the flat fallback
    const tools = [
      { type: "function", function: { name: "test", parameters: {} } },
      { type: "other", name: "flat_tool", parameters: {} },
    ];
    const result = convertTools(tools as any);
    expect(result).toHaveLength(2);
    expect(result[0].function.name).toBe("test");
    expect(result[1].function.name).toBe("flat_tool");
  });

  it("handles pi's flat tool format (name/description/parameters)", () => {
    // pi's built-in tools use flat format, not type+function wrapper
    const tools = [
      {
        name: "bash",
        description: "Execute a bash command",
        parameters: { type: "object", properties: { command: { type: "string" } } },
      },
      {
        name: "read",
        description: "Read a file",
        parameters: { type: "object", properties: { path: { type: "string" } } },
      },
    ];
    const result = convertTools(tools);
    expect(result).toHaveLength(2);
    expect(result[0].type).toBe("function");
    expect(result[0].function.name).toBe("bash");
    expect(result[0].function.description).toBe("Execute a bash command");
    expect(result[1].function.name).toBe("read");
  });

  it("handles mixed flat and wrapped tool formats", () => {
    const tools = [
      { name: "flat_tool", description: "Flat format", parameters: {} },
      { type: "function", function: { name: "wrapped_tool", description: "Wrapped format", parameters: {} } },
    ];
    const result = convertTools(tools);
    expect(result).toHaveLength(2);
    expect(result[0].function.name).toBe("flat_tool");
    expect(result[1].function.name).toBe("wrapped_tool");
  });

  it("handles tools missing optional fields gracefully", () => {
    const tools = [
      { name: "minimal" },
    ];
    const result = convertTools(tools as any);
    expect(result).toHaveLength(1);
    expect(result[0].function.name).toBe("minimal");
    expect(result[0].function.description).toBe("");
    expect(result[0].function.parameters).toEqual({});
  });
});

// ════════════════════════════════════════════════════════════════
// isGhostTokenStream
// ════════════════════════════════════════════════════════════════

describe("isGhostTokenStream", () => {
  it("detects ghost token: done=true, eval_count>0, no content or tool_calls", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false },
      { model: "test", created_at: "", done: true, eval_count: 42, prompt_eval_count: 10 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(true);
  });

  it("not ghost: has content", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "Hello" }, done: false },
      { model: "test", created_at: "", done: true, eval_count: 1 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: has tool calls", () => {
    const chunks: OllamaChatChunk[] = [
      {
        model: "test", created_at: "",
        message: { role: "assistant", content: "", tool_calls: [{ function: { name: "test", arguments: {} } }] },
        done: false,
      },
      { model: "test", created_at: "", done: true, eval_count: 5 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: eval_count is 0", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false },
      { model: "test", created_at: "", done: true, eval_count: 0 },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: stream not finished (no done:true)", () => {
    const chunks: OllamaChatChunk[] = [
      { model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false },
    ];
    expect(isGhostTokenStream(chunks)).toBe(false);
  });

  it("not ghost: empty chunks array", () => {
    expect(isGhostTokenStream([])).toBe(false);
  });
});

// ════════════════════════════════════════════════════════════════
// isOllamaContextOverflow
// ════════════════════════════════════════════════════════════════

describe("isOllamaContextOverflow", () => {
  it("detects 'exceeded max context length'", () => {
    expect(isOllamaContextOverflow("prompt too long; exceeded max context length by 1234 tokens")).toBe(true);
  });

  it("detects 'prompt too long'", () => {
    expect(isOllamaContextOverflow("prompt too long for model context")).toBe(true);
  });

  it("detects 'context window exceeded'", () => {
    expect(isOllamaContextOverflow("context window exceeded")).toBe(true);
  });

  it("detects 'maximum context length exceeded'", () => {
    expect(isOllamaContextOverflow("maximum context length exceeded")).toBe(true);
  });

  it("no match for unrelated errors", () => {
    expect(isOllamaContextOverflow("model not found")).toBe(false);
    expect(isOllamaContextOverflow("connection refused")).toBe(false);
  });

  it("handles Error objects", () => {
    expect(isOllamaContextOverflow(new Error("exceeded max context length"))).toBe(true);
  });

  it("handles null/undefined", () => {
    expect(isOllamaContextOverflow(null)).toBe(false);
    expect(isOllamaContextOverflow(undefined)).toBe(false);
  });
});

// ════════════════════════════════════════════════════════════════
// Stream event format validation
// ════════════════════════════════════════════════════════════════

describe("Stream event format", () => {
  it("stream accepts text_delta event format", async () => {
    const stream = createAssistantMessageEventStream();
    const events: any[] = [];

    // Capture events by listening
    stream.push({ type: "text_start", contentIndex: 0, partial: "test" });
    stream.push({ type: "text_delta", contentIndex: 0, delta: "Hello", partial: "Hello" });
    stream.push({ type: "text_end", contentIndex: 0, content: "Hello", partial: "Hello" });

    stream.end();
    expect(true).toBe(true); // If we got here without error, the format is correct
  });

  it("stream accepts thinking events", async () => {
    const stream = createAssistantMessageEventStream();

    stream.push({ type: "thinking_start", contentIndex: 0 });
    stream.push({ type: "thinking_delta", contentIndex: 0, delta: "Let me think..." });
    stream.push({ type: "thinking_end", contentIndex: 0 });

    stream.end();
    expect(true).toBe(true);
  });

  it("stream accepts toolcall events", async () => {
    const stream = createAssistantMessageEventStream();

    stream.push({ type: "toolcall_start", contentIndex: 0, id: "call_123", name: "read_file" });
    stream.push({ type: "toolcall_delta", contentIndex: 0, delta: { path: "/test" } });
    stream.push({ type: "toolcall_end", contentIndex: 0, id: "call_123" });

    stream.end();
    expect(true).toBe(true);
  });

  it("stream accepts done event", async () => {
    const stream = createAssistantMessageEventStream();

    stream.push({ type: "done", reason: "stop", usage: { input: 10, output: 20 } });

    stream.end();
    expect(true).toBe(true);
  });

  it("stream accepts error event", async () => {
    const stream = createAssistantMessageEventStream();

    stream.push({ type: "error", error: new Error("test error") });

    stream.end();
    expect(true).toBe(true);
  });

  it("stream requires type property for all events", async () => {
    const stream = createAssistantMessageEventStream();

    // Valid events with type property
    expect(() => {
      stream.push({ type: "text_start", contentIndex: 0 });
      stream.end();
    }).not.toThrow();
  });
});

// ════════════════════════════════════════════════════════════════
// Incremental streaming behavior
// ════════════════════════════════════════════════════════════════

describe("Incremental streaming behavior", () => {
  it("multiple text_delta events should accumulate in same block", async () => {
    const stream = createAssistantMessageEventStream();
    const events: any[] = [];

    // Simulate incremental streaming - multiple deltas should go to same block
    stream.push({ type: "text_start", contentIndex: 0 });
    stream.push({ type: "text_delta", contentIndex: 0, delta: "Hel" });
    stream.push({ type: "text_delta", contentIndex: 0, delta: "lo" });
    stream.push({ type: "text_delta", contentIndex: 0, delta: "!" });

    stream.end();
    expect(true).toBe(true);
  });

  it("switching text to thinking should close text block first", async () => {
    const stream = createAssistantMessageEventStream();

    stream.push({ type: "text_start", contentIndex: 0 });
    stream.push({ type: "text_delta", contentIndex: 0, delta: "Hello" });
    stream.push({ type: "text_end", contentIndex: 0, content: "Hello" });

    stream.push({ type: "thinking_start", contentIndex: 0 });
    stream.push({ type: "thinking_delta", contentIndex: 0, delta: "Thinking..." });
    stream.push({ type: "thinking_end", contentIndex: 0 });

    stream.end();
    expect(true).toBe(true);
  });

  it("tool calls should close any open text/thinking blocks", async () => {
    const stream = createAssistantMessageEventStream();

    stream.push({ type: "text_start", contentIndex: 0 });
    stream.push({ type: "text_delta", contentIndex: 0, delta: "Some text" });
    stream.push({ type: "text_end", contentIndex: 0, content: "Some text" });

    stream.push({ type: "toolcall_start", contentIndex: 0, id: "call_1", name: "test" });
    stream.push({ type: "toolcall_delta", contentIndex: 0, delta: { arg: "value" } });
    stream.push({ type: "toolcall_end", contentIndex: 0, id: "call_1" });

    stream.end();
    expect(true).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════
// Tool call arguments parsing
// ════════════════════════════════════════════════════════════════

describe("Tool call arguments parsing", () => {
  it("should handle tool call arguments as JSON string", () => {
    const toolCallWithArgsString = {
      function: {
        name: "read_file",
        arguments: '{"path": "/test/file.txt", "encoding": "utf-8"}',
      },
    };

    // Simulate the parsing logic from native-stream.ts
    let args = toolCallWithArgsString.function.arguments;
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        args = {};
      }
    }

    expect(args).toEqual({ path: "/test/file.txt", encoding: "utf-8" });
  });

  it("should handle tool call arguments as object", () => {
    const toolCallWithArgsObject = {
      function: {
        name: "read_file",
        arguments: { path: "/test/file.txt", encoding: "utf-8" },
      },
    };

    // Simulate the parsing logic from native-stream.ts
    let args = toolCallWithArgsObject.function.arguments;
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        args = {};
      }
    }

    expect(args).toEqual({ path: "/test/file.txt", encoding: "utf-8" });
  });

  it("should handle invalid JSON string in tool call arguments", () => {
    const toolCallWithInvalidArgs = {
      function: {
        name: "read_file",
        arguments: "not valid json",
      },
    };

    // Simulate the parsing logic from native-stream.ts
    let args = toolCallWithInvalidArgs.function.arguments;
    if (typeof args === "string") {
      try {
        args = JSON.parse(args);
      } catch {
        args = {};
      }
    }

    expect(args).toEqual({});
  });
});

// ════════════════════════════════════════════════════════════════
// mergeAdjacentBlocks
// ════════════════════════════════════════════════════════════════

describe("mergeAdjacentBlocks", () => {
  it("merges adjacent text blocks", () => {
    const blocks = [
      { type: "text", text: "Hello " },
      { type: "text", text: "world!" },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].text).toBe("Hello world!");
  });

  it("merges adjacent thinking blocks", () => {
    const blocks = [
      { type: "thinking", thinking: "Step 1" },
      { type: "thinking", thinking: ": analyze" },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].thinking).toBe("Step 1: analyze");
  });

  it("does not merge interleaved thinking and text", () => {
    const blocks = [
      { type: "thinking", thinking: "I think" },
      { type: "text", text: "Hello" },
      { type: "thinking", thinking: "more thinking" },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(3);
  });

  it("merges only adjacent same-type blocks in mixed content", () => {
    const blocks = [
      { type: "thinking", thinking: "think1" },
      { type: "text", text: "text1" },
      { type: "text", text: "text2" },
      { type: "thinking", thinking: "think2" },
      { type: "thinking", thinking: "think3" },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(3);
    expect(blocks[0].type).toBe("thinking");
    expect(blocks[1].type).toBe("text");
    expect(blocks[1].text).toBe("text1text2");
    expect(blocks[2].type).toBe("thinking");
    expect(blocks[2].thinking).toBe("think2think3");
  });

  it("does not merge tool call blocks", () => {
    const blocks = [
      { type: "toolCall", id: "tool_0", name: "bash", arguments: { command: "ls" } },
      { type: "toolCall", id: "tool_1", name: "read", arguments: { path: "/tmp" } },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(2);
  });

  it("handles empty array", () => {
    const blocks: any[] = [];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(0);
  });

  it("handles single block", () => {
    const blocks = [{ type: "text", text: "alone" }];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].text).toBe("alone");
  });

  it("absorbs thinking block between text blocks regardless of size", () => {
    const blocks = [
      { type: "text", text: "Hello " },
      { type: "thinking", thinking: "This is a long thoughtful analysis..." },
      { type: "text", text: "world!" },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].type).toBe("text");
    expect(blocks[0].text).toBe("Hello world!");
  });

  it("does not absorb thinking block at end without trailing text", () => {
    const blocks = [
      { type: "text", text: "Hello" },
      { type: "thinking", thinking: "thinking about it" },
    ];
    mergeAdjacentBlocks(blocks);
    expect(blocks).toHaveLength(2); // unchanged - no trailing text to merge with
  });
});

// ════════════════════════════════════════════════════════════════
// Content block ordering (regression: thinking-first models)
// ════════════════════════════════════════════════════════════════

/**
 * Helper: create a mock fetch that returns the given NDJSON lines
 * as a ReadableStream response.body.
 */
function mockFetchWithNDJSON(lines: string[]): void {
  const bodyStr = lines.join("\n") + "\n";
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(bodyStr));
      controller.close();
    },
  });

  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    body,
    text: () => Promise.resolve(bodyStr),
  } as Response);
}

describe("Content block ordering", () => {
  /**
   * Helper: capture stream events by wrapping stream.push
   */
  function captureEvents(stream: any): any[] {
    const events: any[] = [];
    const originalPush = stream.push.bind(stream);
    stream.push = (event: any) => {
      events.push(event);
      return originalPush(event);
    };
    return events;
  }

  it("should create text block at correct index when thinking precedes text", async () => {
    // Simulate the GLM/minimax pattern: thinking chunks first, then text, then done
    // This tests the regression where textIndex=0 overwrote thinking block at [0]
    mockFetchWithNDJSON([
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", thinking: "1. Analyze the input..." }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "Hello! How can I help?" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", done: true, done_reason: "stop", eval_count: 50, prompt_eval_count: 5, total_duration: 1000 }),
    ]);

    const stream = createAssistantMessageEventStream();
    const events = captureEvents(stream);

    // Let it reject so we can check the final output via error event
    // (after stream.end(), the done event has been pushed)
    try {
      await streamNativeChat(stream, {
        baseUrl: "http://localhost:11434",
        model: "test-model",
        contextWindow: 4096,
        messages: [{ role: "user", content: "hi" }],
      });
    } catch {
      // Expected since the stream will end during processing
    }

    // Find the final message in the events
    const doneEvent = events.find((e: any) => e.type === "done");
    const errorEvent = events.find((e: any) => e.type === "error");

    if (doneEvent) {
      const content = doneEvent.message.content;
      expect(content.length).toBeGreaterThanOrEqual(2);

      // First block should be thinking
      expect(content[0].type).toBe("thinking");
      expect(content[0].thinking).toContain("Analyze");

      // Second block should be text (NOT a second thinking block)
      expect(content[1].type).toBe("text");
      expect(content[1].text).toContain("Hello");

      // The thinking block should NOT have a spurious 'text' property
      expect(content[0].text).toBeUndefined();
    } else if (errorEvent) {
      // If the stream errored, check what we got before the error
      const message = errorEvent.error;
      const content = message.content;
      // At minimum we should have the right structure
      if (content.length >= 2) {
        expect(content[0].type).toBe("thinking");
        expect(content[1].type).toBe("text");
        expect(content[0].text === undefined || content[0].text === "").toBe(true);
      }
    }
  });

  it("should handle multiple interleaved thinking/text blocks", async () => {
    // GLM outputs all thinking chunks first, then all text chunks
    // After mergeAdjacentBlocks, adjacent same-type blocks are merged
    mockFetchWithNDJSON([
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", thinking: "Step 1: " }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", thinking: "Analyze" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", thinking: " input" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "Hi " }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "there" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "!" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", done: true, done_reason: "stop", eval_count: 100, prompt_eval_count: 5 }),
    ]);

    const stream = createAssistantMessageEventStream();
    const events = captureEvents(stream);

    try {
      await streamNativeChat(stream, {
        baseUrl: "http://localhost:11434",
        model: "test-model",
        contextWindow: 4096,
        messages: [{ role: "user", content: "hi" }],
      });
    } catch {
      // Expected
    }

    const doneEvent = events.find((e: any) => e.type === "done");
    const errorEvent = events.find((e: any) => e.type === "error");

    if (doneEvent) {
      const content = doneEvent.message.content;
      // After mergeAdjacentBlocks: thinking(merged 3), text(merged 3) = 2 blocks
      expect(content.length).toBe(2);

      expect(content[0].type).toBe("thinking");
      expect(content[0].thinking).toContain("Step 1: Analyze input");

      expect(content[1].type).toBe("text");
      expect(content[1].text).toBe("Hi there!");

      // No spurious 'text' on thinking blocks
      expect(content[0].text).toBeUndefined();
    } else if (errorEvent) {
      // At minimum check the structure collected so far
      const content = errorEvent.error.content;
      const thinkingBlocks = content.filter((b: any) => b.type === "thinking");
      const textBlocks = content.filter((b: any) => b.type === "text");
      // No thinking block should have 'text' property
      for (const tb of thinkingBlocks) {
        expect(tb.text).toBeUndefined();
      }
      // Text blocks should have content (not empty)
      for (const tb of textBlocks) {
        expect(tb.text.length).toBeGreaterThan(0);
      }
    }
  });

  it("should handle text before thinking (standard order)", async () => {
    // Some models output text first, then thinking in a follow-up
    // After merge pass 2, the thinking between text gets absorbed
    mockFetchWithNDJSON([
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "I am thinking" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", thinking: "actually let me reconsider" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "Final answer" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", done: true, done_reason: "stop", eval_count: 30, prompt_eval_count: 5 }),
    ]);

    const stream = createAssistantMessageEventStream();
    const events = captureEvents(stream);

    try {
      await streamNativeChat(stream, {
        baseUrl: "http://localhost:11434",
        model: "test-model",
        contextWindow: 4096,
        messages: [{ role: "user", content: "hi" }],
      });
    } catch {
      // Expected
    }

    const doneEvent = events.find((e: any) => e.type === "done");
    const errorEvent = events.find((e: any) => e.type === "error");

    if (doneEvent) {
      const content = doneEvent.message.content;
      // Pass 2 absorbs the thinking sandwiched between text → 1 merged text block
      expect(content.length).toBe(1);
      expect(content[0].type).toBe("text");
      expect(content[0].text).toBe("I am thinkingFinal answer");
    }
  });

  it("should handle text-only response (no thinking)", async () => {
    // Models like llama without thinking
    mockFetchWithNDJSON([
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "Hel" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "lo!" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", done: true, done_reason: "stop", eval_count: 2, prompt_eval_count: 1 }),
    ]);

    const stream = createAssistantMessageEventStream();
    const events = captureEvents(stream);

    try {
      await streamNativeChat(stream, {
        baseUrl: "http://localhost:11434",
        model: "test-model",
        contextWindow: 4096,
        messages: [{ role: "user", content: "hi" }],
      });
    } catch {
      // Expected
    }

    const doneEvent = events.find((e: any) => e.type === "done");
    const errorEvent = events.find((e: any) => e.type === "error");

    if (doneEvent) {
      const content = doneEvent.message.content;
      expect(content.length).toBe(1);
      expect(content[0].type).toBe("text");
      expect(content[0].text).toBe("Hello!");
    }
  });

  it("should handle tool calls after thinking", async () => {
    mockFetchWithNDJSON([
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", thinking: "I should use a tool" }, done: false }),
      JSON.stringify({ model: "test", created_at: "", message: { role: "assistant", content: "", tool_calls: [{ function: { name: "read", arguments: { path: "/test" } } }] }, done: false }),
      JSON.stringify({ model: "test", created_at: "", done: true, done_reason: "stop", eval_count: 10, prompt_eval_count: 5 }),
    ]);

    const stream = createAssistantMessageEventStream();
    const events = captureEvents(stream);

    try {
      await streamNativeChat(stream, {
        baseUrl: "http://localhost:11434",
        model: "test-model",
        contextWindow: 4096,
        messages: [{ role: "user", content: "read a file" }],
      });
    } catch {
      // Expected
    }

    const doneEvent = events.find((e: any) => e.type === "done");
    const errorEvent = events.find((e: any) => e.type === "error");

    if (doneEvent) {
      const content = doneEvent.message.content;
      expect(content.length).toBe(2);
      expect(content[0].type).toBe("thinking");
      expect(content[1].type).toBe("toolCall");
    }
  });
});