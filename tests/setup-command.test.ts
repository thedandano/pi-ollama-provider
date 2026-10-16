/**
 * Runs the registered /ollama-setup handler the same way index.ts wires it.
 * Current Pi exposes ctx.modelRegistry without an authStorage property.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import extension from "../extensions/pi-ollama-provider/index.js";
import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

const originalHost = process.env.OLLAMA_HOST;
const originalApiKey = process.env.OLLAMA_API_KEY;
const originalHome = process.env.HOME;

describe("/ollama-setup command wiring", () => {
  let home: string;
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    home = mkdtempSync(join(tmpdir(), "pi-ollama-home-"));
    process.env.HOME = home;
    delete process.env.OLLAMA_API_KEY;
    delete process.env.OLLAMA_API_BASE;
    process.env.OLLAMA_HOST = "http://192.168.1.9:11434";
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url === "http://192.168.1.9:11434/api/tags") {
        return { ok: true, status: 200, json: async () => ({ models: [] }) } as Response;
      }
      return { ok: false, status: 500, text: async () => "nope" } as Response;
    });
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    if (originalHost === undefined) delete process.env.OLLAMA_HOST;
    else process.env.OLLAMA_HOST = originalHost;
    if (originalApiKey === undefined) delete process.env.OLLAMA_API_KEY;
    else process.env.OLLAMA_API_KEY = originalApiKey;
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    rmSync(home, { recursive: true, force: true });
  });

  function installExtension() {
    const commands = new Map<string, { handler: (args: unknown, ctx: ExtensionCommandContext) => Promise<void> }>();
    const pi = {
      registerCommand: vi.fn((name: string, def: { handler: (args: unknown, ctx: ExtensionCommandContext) => Promise<void> }) => {
        commands.set(name, def);
      }),
      registerProvider: vi.fn(),
      unregisterProvider: vi.fn(),
      on: vi.fn(),
      exec: vi.fn(async () => {
        throw new Error("CLI should not run for a remote OLLAMA_HOST");
      }),
    };
    return { commands, pi: pi as unknown as ExtensionAPI & { exec: ReturnType<typeof vi.fn> } };
  }

  function commandUi(editorValue: string) {
    const notifications: { message: string; type?: string }[] = [];
    return {
      select: vi.fn(async () => "Local — Run models on this machine or another Ollama server"),
      editor: vi.fn(async () => editorValue),
      input: vi.fn(async () => undefined),
      confirm: vi.fn(async () => false),
      notify: vi.fn((message: string, type?: string) => notifications.push({ message, type })),
      setStatus: vi.fn(),
      notifications,
    };
  }

  it("completes local setup when modelRegistry has no authStorage", async () => {
    const authPath = join(home, ".pi", "agent", "auth.json");
    mkdirSync(join(home, ".pi", "agent"), { recursive: true });
    writeFileSync(authPath, JSON.stringify({
      anthropic: { type: "api_key", key: "keep-me" },
      ollama: { type: "api_key", key: "old-local" },
      "ollama-cloud": { type: "api_key", key: "old-cloud" },
    }));

    const { commands, pi } = installExtension();
    await extension(pi);

    const ui = commandUi("http://192.168.1.9:11434");
    const ctx = {
      ui,
      hasUI: true,
      // Current Pi ModelRegistry is a facade with no authStorage field.
      modelRegistry: {},
    } as unknown as ExtensionCommandContext;

    await expect(commands.get("ollama-setup")!.handler(undefined, ctx)).resolves.toBeUndefined();

    expect(ui.notify).toHaveBeenCalledWith("OLLAMA_HOST set to http://192.168.1.9:11434", "info");
    expect(ui.notifications.some((note) => note.message.includes("Setup complete"))).toBe(true);
    expect(pi.exec).not.toHaveBeenCalled();

    const saved = JSON.parse(readFileSync(authPath, "utf-8"));
    expect(saved.anthropic).toEqual({ type: "api_key", key: "keep-me" });
    expect(saved.ollama).toBeUndefined();
    expect(saved["ollama-cloud"]).toBeUndefined();
  });

  it("still uses modelRegistry.authStorage when Pi provides it", async () => {
    const removed: string[] = [];
    const authStorage = {
      has: vi.fn((key: string) => key === "ollama" || key === "ollama-cloud"),
      get: vi.fn(),
      set: vi.fn(),
      remove: vi.fn((key: string) => removed.push(key)),
    };

    const { commands, pi } = installExtension();
    await extension(pi);

    const ui = commandUi("http://192.168.1.9:11434");
    const ctx = {
      ui,
      hasUI: true,
      modelRegistry: { authStorage },
    } as unknown as ExtensionCommandContext;

    await commands.get("ollama-setup")!.handler(undefined, ctx);

    expect(removed).toEqual(expect.arrayContaining(["ollama", "ollama-cloud"]));
    expect(existsSync(join(home, ".pi", "agent", "auth.json"))).toBe(false);
  });
});
