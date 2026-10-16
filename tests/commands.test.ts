/**
 * Tests for commands.ts — settings, settings validation, setup wizard flow.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  readSettings,
  writeSettings,
  validateSettings,
  type OllamaSettings,
  type SettingsValidationIssue,
} from "../extensions/pi-ollama-provider/commands.js";

import { runSetupWizard } from "../extensions/pi-ollama-provider/commands.js";

import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

let tempDir: string;
let settingsPath: string;

beforeEach(() => {
  tempDir = join(tmpdir(), `pi-ollama-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tempDir, { recursive: true });
  settingsPath = join(tempDir, "settings.json");
});

afterEach(() => {
  if (existsSync(tempDir)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

// ════════════════════════════════════════════════════════════════
// validateSettings
// ════════════════════════════════════════════════════════════════

describe("validateSettings", () => {
  it("validates valid settings with no issues", () => {
    const { validated, issues } = validateSettings({
      streamingMode: "native",
      keepAlive: "30m",
      autoPull: true,
    });
    expect(issues).toHaveLength(0);
    expect(validated.streamingMode).toBe("native");
    expect(validated.keepAlive).toBe("30m");
    expect(validated.autoPull).toBe(true);
  });

  it("rejects invalid streamingMode", () => {
    const { validated, issues } = validateSettings({
      streamingMode: "fast" as any,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(issues[0].key).toBe("streamingMode");
    expect(validated.streamingMode).toBe("native"); // reset to default
  });

  it("accepts 'openai-compat' streamingMode", () => {
    const { validated, issues } = validateSettings({
      streamingMode: "openai-compat",
    });
    expect(issues).toHaveLength(0);
    expect(validated.streamingMode).toBe("openai-compat");
  });

  it("rejects negative defaultNumCtx", () => {
    const { validated, issues } = validateSettings({
      defaultNumCtx: -1,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.defaultNumCtx).toBeUndefined();
  });

  it("rejects zero defaultNumCtx", () => {
    const { validated, issues } = validateSettings({
      defaultNumCtx: 0,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.defaultNumCtx).toBeUndefined();
  });

  it("rejects NaN defaultNumCtx", () => {
    const { validated, issues } = validateSettings({
      defaultNumCtx: NaN,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.defaultNumCtx).toBeUndefined();
  });

  it("rejects string defaultNumCtx", () => {
    const { validated, issues } = validateSettings({
      defaultNumCtx: "big" as any,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.defaultNumCtx).toBeUndefined();
  });

  it("caps defaultNumCtx at 131072", () => {
    const { validated, issues } = validateSettings({
      defaultNumCtx: 999999,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.defaultNumCtx).toBe(131072);
  });

  it("accepts valid defaultNumCtx", () => {
    const { validated, issues } = validateSettings({
      defaultNumCtx: 32768,
    });
    expect(issues).toHaveLength(0);
    expect(validated.defaultNumCtx).toBe(32768);
  });

  it("rejects temperature > 2", () => {
    const { validated, issues } = validateSettings({
      options: { temperature: 5 },
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.options!.temperature).toBeUndefined();
  });

  it("rejects negative temperature", () => {
    const { validated, issues } = validateSettings({
      options: { temperature: -0.5 },
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.options!.temperature).toBeUndefined();
  });

  it("accepts valid temperature between 0 and 2", () => {
    const { validated, issues } = validateSettings({
      options: { temperature: 0.7 },
    });
    expect(issues).toHaveLength(0);
    expect(validated.options!.temperature).toBe(0.7);
  });

  it("rejects top_p > 1", () => {
    const { validated, issues } = validateSettings({
      options: { top_p: 1.5 },
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.options!.top_p).toBeUndefined();
  });

  it("rejects negative top_p", () => {
    const { validated, issues } = validateSettings({
      options: { top_p: -0.1 },
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.options!.top_p).toBeUndefined();
  });

  it("rejects non-integer top_k", () => {
    const { validated, issues } = validateSettings({
      options: { top_k: 3.7 },
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.options!.top_k).toBeUndefined();
  });

  it("rejects negative top_k", () => {
    const { validated, issues } = validateSettings({
      options: { top_k: -1 },
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.options!.top_k).toBeUndefined();
  });

  it("rejects invalid keepAlive format", () => {
    const { validated, issues } = validateSettings({
      keepAlive: "invalid" as any,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.keepAlive).toBe("30m");
  });

  it("accepts valid keepAlive formats", () => {
    const formats = ["30m", "1h", "3600s", "5", "1:30"];
    for (const ka of formats) {
      const { validated, issues } = validateSettings({ keepAlive: ka } as any);
      expect(issues).toHaveLength(0);
      expect(validated.keepAlive).toBe(ka);
    }
  });

  it("rejects non-boolean autoPull", () => {
    const { validated, issues } = validateSettings({
      autoPull: "yes" as any,
    });
    expect(issues.length).toBeGreaterThan(0);
    expect(validated.autoPull).toBe(true);
  });

  it("merges empty settings with defaults", () => {
    const { validated } = validateSettings({});
    expect(validated.streamingMode).toBe("native");
  });

  it("handles empty settings object", () => {
    const { validated, issues } = validateSettings({});
    expect(issues).toHaveLength(0);
    expect(validated.streamingMode).toBe("native");
    expect(validated.keepAlive).toBe("30m");
  });

  it("normalizes a saved localBaseUrl", () => {
    const { validated, issues } = validateSettings({ localBaseUrl: "10.0.0.8" });
    expect(issues).toHaveLength(0);
    expect(validated.localBaseUrl).toBe("http://10.0.0.8:11434");
    expect(validated.mode).toBeUndefined();
  });

  it("rejects a non-string localBaseUrl", () => {
    const { validated, issues } = validateSettings({ localBaseUrl: 11434 as any });
    expect(issues.some((issue) => issue.key === "localBaseUrl")).toBe(true);
    expect(validated.localBaseUrl).toBeUndefined();
  });

  it("rejects an invalid mode", () => {
    const { validated, issues } = validateSettings({ mode: "lan" as any });
    expect(issues.some((issue) => issue.key === "mode")).toBe(true);
    expect(validated.mode).toBeUndefined();
  });

  it("accepts local and cloud modes", () => {
    expect(validateSettings({ mode: "local" }).validated.mode).toBe("local");
    expect(validateSettings({ mode: "cloud" }).issues).toHaveLength(0);
    expect(validateSettings({ mode: "cloud" }).validated.mode).toBe("cloud");
  });
});

// ════════════════════════════════════════════════════════════════
// Settings defaults
// ════════════════════════════════════════════════════════════════

describe("settings defaults", () => {
  it("default streamingMode is 'native'", () => {
    const expected: Partial<OllamaSettings> = {
      streamingMode: "native",
      keepAlive: "30m",
      autoPull: true,
    };
    expect(expected.streamingMode).toBe("native");
    expect(expected.keepAlive).toBe("30m");
    expect(expected.autoPull).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════
// Setup wizard flow tests (integration)
// ════════════════════════════════════════════════════════════════

describe("runSetupWizard", () => {
  let fetchSpy: any;
  const originalHost = process.env.OLLAMA_HOST;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
    delete process.env.OLLAMA_HOST;
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    if (originalHost !== undefined) {
      process.env.OLLAMA_HOST = originalHost;
    } else {
      delete process.env.OLLAMA_HOST;
    }
  });

  function mockAuthStore() {
    const store: Record<string, any> = {};
    return {
      has: vi.fn((key: string) => key in store),
      get: vi.fn((key: string) => store[key]),
      set: vi.fn((key: string, value: any) => {
        store[key] = value;
      }),
      remove: vi.fn((key: string) => {
        delete store[key];
      }),
    };
  }

  function mockUI(responses: {
    selects?: Array<string | undefined>;
    confirms?: boolean[];
    inputs?: Array<string | undefined>;
    editors?: Array<string | undefined>;
  }) {
    let selectIdx = 0;
    let confirmIdx = 0;
    let inputIdx = 0;
    let editorIdx = 0;
    const notifications: { message: string; type?: string }[] = [];
    return {
      select: vi.fn(async () => responses.selects?.[selectIdx++] ?? undefined),
      confirm: vi.fn(async () => responses.confirms?.[confirmIdx++] ?? false),
      input: vi.fn(async () => responses.inputs?.[inputIdx++] ?? undefined),
      editor: vi.fn(async () => responses.editors?.[editorIdx++] ?? undefined),
      notify: vi.fn((message: string, type?: string) => notifications.push({ message, type })),
      setStatus: vi.fn(),
      notifications,
    };
  }

  function cliInstallOrStartCalls(pi: { exec: { mock: { calls: unknown[][] } } }): unknown[][] {
    return pi.exec.mock.calls.filter((call) => {
      const cmd = call[0];
      const args = call[1] as string[] | undefined;
      if (cmd === "ollama" && args?.[0] === "--version") return true;
      if (cmd === "bash" && args?.some((arg) => String(arg).includes("ollama"))) return true;
      return false;
    });
  }

  function mockPi(opts?: { ollamaInstalled?: boolean; ollamaSigninCode?: number }) {
    return {
      exec: vi.fn(async (cmd: string, args: string[]) => {
        if (cmd === "ollama" && args[0] === "--version") {
          return { code: opts?.ollamaInstalled === false ? 1 : 0 };
        }
        if (cmd === "ollama" && args[0] === "signin") {
          return { code: opts?.ollamaSigninCode ?? 0, stderr: "", stdout: "" };
        }
        return { code: 0, stderr: "", stdout: "" };
      }),
      registerProvider: vi.fn(),
      unregisterProvider: vi.fn(),
    } as unknown as ExtensionAPI;
  }

  it("Local: clears stored cloud credential and calls onConfigChange", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    store.set("ollama", { type: "api_key", key: "old-cloud-key" });
    store.set("ollama-cloud", { type: "api_key", key: "old-cloud-key" });

    const ui = mockUI({
      selects: ["Local"],
      confirms: [true],
      editors: ["http://localhost:11434"],
    });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    let configChanged = false;
    let newMode: string | undefined;
    let newBaseUrl: string | undefined;
    let newApiKey: string | undefined;

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async (mode, baseUrl, apiKey) => {
      configChanged = true;
      newMode = mode;
      newBaseUrl = baseUrl;
      newApiKey = apiKey;
    });

    expect(store.remove).toHaveBeenCalledWith("ollama");
    expect(store.remove).toHaveBeenCalledWith("ollama-cloud");
    expect(configChanged).toBe(true);
    expect(newMode).toBe("local");
    expect(newBaseUrl).toBe("http://localhost:11434");
    expect(newApiKey).toBe("ollama");
    expect(ui.editor).toHaveBeenCalledTimes(1);
    expect(ui.editor).toHaveBeenCalledWith(expect.any(String), "http://localhost:11434");
    expect(ui.notifications.some((n) => n.message.includes("OLLAMA_HOST"))).toBe(false);
    expect(cliInstallOrStartCalls(pi).length).toBeGreaterThan(0);
    const saved = JSON.parse(readFileSync(settingsPath, "utf-8"));
    expect(saved.ollama.mode).toBe("local");
    expect(saved.ollama.localBaseUrl).toBe("http://localhost:11434");
  });

  it("Cloud + API key: saves to authStorage under 'ollama-cloud' key", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["my-api-key"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: true, status: 200, json: async () => ({ models: [{ name: "qwen3:8b", size: 4e9 }] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).toHaveBeenCalledWith("ollama-cloud", { type: "api_key", key: "my-api-key" });
  });

  it("Cloud + API key: does NOT save on HTTP failure", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["bad-key"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url.includes("localhost")) return { ok: true, json: async () => ({ models: [] }) } as Response;
      return { ok: false, status: 401, text: async () => "Unauthorized" } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalledWith(
      "ollama-cloud",
      expect.objectContaining({ key: "bad-key" }),
    );
  });

  it("Cloud + API key: cancelled when user enters nothing", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: [""] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalled();
  });

  it("Cancel at mode selection: no changes", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: [undefined] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalled();
    expect(store.remove).not.toHaveBeenCalled();
  });

  it("Cancel at auth method selection: no changes", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", undefined] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
    }, async () => {});

    expect(store.set).not.toHaveBeenCalled();
  });

  it("notifies when OLLAMA_HOST is set and skips CLI install/start", async () => {
    process.env.OLLAMA_HOST = "10.1.2.3";
    writeFileSync(settingsPath, JSON.stringify({
      theme: "dark",
      ollama: { keepAlive: "1h" },
    }));
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Local"], editors: ["http://10.1.2.3:11434"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;
    const fetched: string[] = [];

    fetchSpy.mockImplementation(async (url: string) => {
      fetched.push(String(url));
      if (url === "http://10.1.2.3:11434/api/tags") {
        return { ok: true, json: async () => ({ models: [{ name: "llama3.1:8b" }] }) } as Response;
      }
      return { ok: false, status: 500, text: async () => "nope" } as Response;
    });

    let savedUrl: string | undefined;
    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async (mode, baseUrl) => {
      savedUrl = `${mode}:${baseUrl}`;
    });

    expect(ui.notify).toHaveBeenCalledWith("OLLAMA_HOST set to http://10.1.2.3:11434", "info");
    expect(ui.editor).toHaveBeenCalledWith(expect.any(String), "http://10.1.2.3:11434");
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
    expect(fetched).toContain("http://10.1.2.3:11434/api/tags");
    expect(fetched.some((url) => url.includes("localhost"))).toBe(false);
    expect(savedUrl).toBe("local:http://10.1.2.3:11434");
    const saved = JSON.parse(readFileSync(settingsPath, "utf-8"));
    expect(saved.theme).toBe("dark");
    expect(saved.ollama.keepAlive).toBe("1h");
    expect(saved.ollama.mode).toBe("local");
    expect(saved.ollama.localBaseUrl).toBe("http://10.1.2.3:11434");
  });

  it("skips CLI when OLLAMA_HOST is set even if the accepted URL is localhost", async () => {
    process.env.OLLAMA_HOST = "http://127.0.0.1:11434";
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Local"], editors: ["http://localhost:11434"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      if (url === "http://localhost:11434/api/tags") {
        return { ok: true, json: async () => ({ models: [] }) } as Response;
      }
      throw new Error(`unexpected ${url}`);
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async () => {});

    expect(ui.notify).toHaveBeenCalledWith("OLLAMA_HOST set to http://127.0.0.1:11434", "info");
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
    expect(ui.confirm).not.toHaveBeenCalled();
  });

  it("skips CLI for a non-loopback URL chosen without OLLAMA_HOST", async () => {
    delete process.env.OLLAMA_HOST;
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Local"], editors: ["10.0.0.8"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;
    const fetched: string[] = [];

    fetchSpy.mockImplementation(async (url: string) => {
      fetched.push(String(url));
      return { ok: true, json: async () => ({ models: [] }) } as Response;
    });

    let newBaseUrl: string | undefined;
    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async (_mode, baseUrl) => {
      newBaseUrl = baseUrl;
    });

    expect(ui.notifications.some((n) => n.message.includes("OLLAMA_HOST"))).toBe(false);
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
    expect(fetched).toEqual(["http://10.0.0.8:11434/api/tags"]);
    expect(newBaseUrl).toBe("http://10.0.0.8:11434");
    expect(ui.confirm).not.toHaveBeenCalled();
  });

  it("re-prompts after a remote probe failure and lets the user cancel", async () => {
    delete process.env.OLLAMA_HOST;
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({
      selects: ["Local"],
      editors: ["http://10.0.0.4:11434", undefined],
    });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: false, status: 503, text: async () => "unavailable" } as Response;
    });

    let changed = false;
    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async () => {
      changed = true;
    });

    expect(changed).toBe(false);
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
    expect(ui.confirm).not.toHaveBeenCalled();
    expect(ui.editor).toHaveBeenCalledTimes(2);
    expect(ui.editor).toHaveBeenNthCalledWith(2, expect.any(String), "http://10.0.0.4:11434");
    expect(ui.notifications.some((n) => n.type === "error" && n.message.includes("http://10.0.0.4:11434") && n.message.includes("503"))).toBe(true);
    expect(ui.notifications.some((n) => n.type === "warning" && /cancel/i.test(n.message))).toBe(true);
    expect(existsSync(settingsPath)).toBe(false);
    expect(store.remove).not.toHaveBeenCalled();
  });

  it("saves a remote URL after a failed probe when the retry succeeds", async () => {
    delete process.env.OLLAMA_HOST;
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({
      selects: ["Local"],
      editors: ["http://10.0.0.6:11434", "http://10.0.0.7:11434"],
    });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;
    let calls = 0;

    fetchSpy.mockImplementation(async (url: string) => {
      calls++;
      if (calls === 1) return { ok: false, status: 500, text: async () => "down" } as Response;
      if (url === "http://10.0.0.7:11434/api/tags") {
        return { ok: true, json: async () => ({ models: [] }) } as Response;
      }
      throw new Error(`unexpected ${url}`);
    });

    let newBaseUrl: string | undefined;
    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async (_mode, baseUrl) => {
      newBaseUrl = baseUrl;
    });

    expect(newBaseUrl).toBe("http://10.0.0.7:11434");
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
  });

  it("prefills a previously saved local URL ahead of OLLAMA_HOST", async () => {
    process.env.OLLAMA_HOST = "10.9.9.9";
    writeFileSync(settingsPath, JSON.stringify({
      theme: "dark",
      ollama: { keepAlive: "1h", localBaseUrl: "http://10.0.0.5:11434" },
    }));
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Local"], editors: [undefined] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async () => {});

    expect(ui.notify).toHaveBeenCalledWith("OLLAMA_HOST set to http://10.9.9.9:11434", "info");
    expect(ui.editor).toHaveBeenCalledWith(expect.any(String), "http://10.0.0.5:11434");
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
    const saved = JSON.parse(readFileSync(settingsPath, "utf-8"));
    expect(saved.theme).toBe("dark");
    expect(saved.ollama.localBaseUrl).toBe("http://10.0.0.5:11434");
    expect(saved.ollama.keepAlive).toBe("1h");
  });

  it("treats 127.0.0.1 and ::1 as this machine and still checks the CLI", async () => {
    delete process.env.OLLAMA_HOST;
    for (const url of ["http://127.0.0.1:11434", "http://[::1]:11434"]) {
      const pi = mockPi();
      const store = mockAuthStore();
      const ui = mockUI({ selects: ["Local"], editors: [url] });
      const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;
      const fetched: string[] = [];
      fetchSpy.mockImplementation(async (fetchedUrl: string) => {
        fetched.push(String(fetchedUrl));
        return { ok: true, json: async () => ({ models: [] }) } as Response;
      });

      await runSetupWizard(pi, ctx, {
        localBaseUrl: "http://localhost:11434",
        cloudBaseUrl: "https://ollama.com",
        apiKey: "ollama",
        authStorage: store,
        settingsPath,
      }, async () => {});

      expect(cliInstallOrStartCalls(pi).length).toBeGreaterThan(0);
      expect(fetched.some((fetchedUrl) => fetchedUrl.startsWith(`${url}/api/tags`))).toBe(true);
    }
  });

  it("does not install or start a local server before ollama signin", async () => {
    process.env.OLLAMA_HOST = "10.4.5.6:11434";
    const pi = mockPi({ ollamaSigninCode: 0 });
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "Browser login"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async (url: string) => {
      throw new Error(`unexpected fetch ${url}`);
    });

    let newBaseUrl: string | undefined;
    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async (mode, baseUrl) => {
      expect(mode).toBe("local");
      newBaseUrl = baseUrl;
    });

    expect(pi.exec).toHaveBeenCalledWith("ollama", ["signin"], expect.anything());
    expect(cliInstallOrStartCalls(pi)).toEqual([]);
    expect(ui.editor).not.toHaveBeenCalled();
    expect(newBaseUrl).toBe("http://10.4.5.6:11434");
  });

  it("Cloud API key setup does not prompt for a local URL", async () => {
    const pi = mockPi();
    const store = mockAuthStore();
    const ui = mockUI({ selects: ["Cloud", "API key"], inputs: ["my-api-key"] });
    const ctx = { ui, hasUI: true, modelRegistry: { authStorage: store } } as unknown as ExtensionCommandContext;

    fetchSpy.mockImplementation(async () => {
      return { ok: true, status: 200, json: async () => ({ models: [] }) } as Response;
    });

    await runSetupWizard(pi, ctx, {
      localBaseUrl: "http://localhost:11434",
      cloudBaseUrl: "https://ollama.com",
      apiKey: "ollama",
      authStorage: store,
      settingsPath,
    }, async () => {});

    expect(ui.editor).not.toHaveBeenCalled();
    expect(store.set).toHaveBeenCalledWith("ollama-cloud", { type: "api_key", key: "my-api-key" });
  });
});