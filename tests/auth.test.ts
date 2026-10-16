/**
 * Tests for auth.ts — async config resolution, OLLAMA_API_BASE, auth source tracking.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { writeFileSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import {
  resolveConfig,
  resolveConfigAsync,
  readOllamaAuthFromJson,
  describeAuthSource,
  authHeaders,
  DEFAULT_LOCAL_URL,
  DEFAULT_CLOUD_URL,
  type OllamaConfig,
} from "../extensions/pi-ollama-provider/auth.js";

let tempDir: string;
let authPath: string;

beforeEach(() => {
  tempDir = join(tmpdir(), `pi-ollama-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tempDir, { recursive: true });
  authPath = join(tempDir, "auth.json");
  delete process.env.OLLAMA_API_KEY;
  delete process.env.OLLAMA_API_BASE;
  delete process.env.OLLAMA_HOST;
});

afterEach(() => {
  delete process.env.OLLAMA_API_KEY;
  delete process.env.OLLAMA_API_BASE;
  delete process.env.OLLAMA_HOST;
  if (existsSync(tempDir)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

// ════════════════════════════════════════════════════════════════
// readOllamaAuthFromJson (unchanged, but still tested)
// ════════════════════════════════════════════════════════════════

describe("readOllamaAuthFromJson", () => {
  it("reads a valid api_key credential from 'ollama' key", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "my-key-123" } }));
    expect(readOllamaAuthFromJson(authPath)).toEqual({ type: "api_key", key: "my-key-123" });
  });

  it("reads a valid api_key credential from 'ollama-cloud' key", () => {
    writeFileSync(authPath, JSON.stringify({ "ollama-cloud": { type: "api_key", key: "cloud-key" } }));
    expect(readOllamaAuthFromJson(authPath)).toEqual({ type: "api_key", key: "cloud-key" });
  });

  it("prefers ollama-cloud over ollama", () => {
    writeFileSync(authPath, JSON.stringify({
      ollama: { type: "api_key", key: "old-key" },
      "ollama-cloud": { type: "api_key", key: "new-cloud-key" },
    }));
    expect(readOllamaAuthFromJson(authPath)).toEqual({ type: "api_key", key: "new-cloud-key" });
  });

  it("returns undefined when no ollama entry", () => {
    writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api_key", key: "sk-ant" } }));
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });

  it("returns undefined when file doesn't exist", () => {
    expect(readOllamaAuthFromJson(join(tempDir, "nope.json"))).toBeUndefined();
  });

  it("returns undefined for wrong credential type", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "oauth", access: "token" } }));
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });

  it("returns undefined for missing key field", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key" } }));
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });

  it("returns undefined for malformed JSON", () => {
    writeFileSync(authPath, "not-json{{{");
    expect(readOllamaAuthFromJson(authPath)).toBeUndefined();
  });
});

// ════════════════════════════════════════════════════════════════
// resolveConfig (sync) — priority chain + OLLAMA_API_BASE
// ════════════════════════════════════════════════════════════════

describe("resolveConfig", () => {
  it("stored credential wins over env var", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "stored-key" } }));
    const config = resolveConfig({ authPath, envKey: "env-key" });
    expect(config.apiKey).toBe("stored-key");
    expect(config.mode).toBe("cloud");
    expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
  });

  it("env var wins when no stored credential", () => {
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath, envKey: "env-key-123" });
    expect(config.apiKey).toBe("env-key-123");
    expect(config.mode).toBe("cloud");
    expect(config.baseUrl).toBe(DEFAULT_CLOUD_URL);
  });

  it("defaults to local when nothing configured", () => {
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
  });

  it("defaults to local when file missing", () => {
    const config = resolveConfig({ authPath: join(tempDir, "nope.json") });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
  });

  it("env var used when auth.json has no ollama entry", () => {
    writeFileSync(authPath, JSON.stringify({ anthropic: { type: "api_key", key: "sk-ant" } }));
    const config = resolveConfig({ authPath, envKey: "env-ollama-key" });
    expect(config.apiKey).toBe("env-ollama-key");
    expect(config.mode).toBe("cloud");
  });

  it("stored key='ollama' → local mode", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "ollama" } }));
    const config = resolveConfig({ authPath });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
  });

  it("no env var, no stored → local", () => {
    const config = resolveConfig({ authPath, envKey: undefined });
    expect(config.mode).toBe("local");
  });

  it("empty string env var is ignored", () => {
    const config = resolveConfig({ authPath, envKey: "" });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
  });

  it("uses real process.env.OLLAMA_API_KEY when envKey not specified", () => {
    process.env.OLLAMA_API_KEY = "from-real-env";
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath });
    expect(config.apiKey).toBe("from-real-env");
    expect(config.mode).toBe("cloud");
  });

  // ── NEW: OLLAMA_API_BASE ──

  it("OLLAMA_API_BASE overrides cloud baseUrl", () => {
    process.env.OLLAMA_API_BASE = "https://custom-ollama.example.com/";
    writeFileSync(authPath, JSON.stringify({ "ollama-cloud": { type: "api_key", key: "my-key" } }));
    const config = resolveConfig({ authPath });
    expect(config.baseUrl).toBe("https://custom-ollama.example.com");
    expect(config.mode).toBe("cloud");
  });

  it("OLLAMA_API_BASE strips trailing slashes", () => {
    process.env.OLLAMA_API_BASE = "https://custom.example.com/ollama///";
    writeFileSync(authPath, JSON.stringify({ "ollama-cloud": { type: "api_key", key: "key" } }));
    const config = resolveConfig({ authPath });
    expect(config.baseUrl).toBe("https://custom.example.com/ollama");
  });

  it("OLLAMA_API_BASE via options override", () => {
    const config = resolveConfig({ authPath, envBase: "https://override.example.com" });
    // Since no auth and no envKey, mode is local, so baseUrl is the local server
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe(DEFAULT_LOCAL_URL);
  });

  it("local mode uses normalized OLLAMA_HOST as the base URL", () => {
    process.env.OLLAMA_HOST = "10.2.3.4";
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath, envKey: "" });
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe("http://10.2.3.4:11434");
  });

  it("OLLAMA_API_BASE with cloud mode uses custom endpoint", () => {
    process.env.OLLAMA_API_BASE = "https://enterprise-ollama.internal";
    const config = resolveConfig({ authPath, envKey: "enterprise-key" });
    expect(config.mode).toBe("cloud");
    expect(config.baseUrl).toBe("https://enterprise-ollama.internal");
    expect(config.apiKey).toBe("enterprise-key");
  });

  // ── NEW: authSource tracking ──

  it("tracks authSource for stored ollama-cloud credential", () => {
    writeFileSync(authPath, JSON.stringify({ "ollama-cloud": { type: "api_key", key: "cloud-key" } }));
    const config = resolveConfig({ authPath });
    expect(config.authSource).toBe("auth-json-ollama-cloud");
  });

  it("tracks authSource for stored ollama credential", () => {
    writeFileSync(authPath, JSON.stringify({ ollama: { type: "api_key", key: "my-key" } }));
    const config = resolveConfig({ authPath });
    expect(config.authSource).toBe("auth-json-ollama-cloud"); // ollama-cloud takes priority
  });

  it("tracks authSource for env var", () => {
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath, envKey: "env-key" });
    expect(config.authSource).toBe("env-var");
  });

  it("tracks authSource as default for local mode", () => {
    writeFileSync(authPath, JSON.stringify({}));
    const config = resolveConfig({ authPath });
    expect(config.authSource).toBe("default");
  });
});

// ════════════════════════════════════════════════════════════════
// resolveConfigAsync — AuthStorage-based resolution
// ════════════════════════════════════════════════════════════════

describe("resolveConfigAsync", () => {
  function mockAuthStore(keys: Record<string, string | undefined>) {
    return {
      getApiKey: async (provider: string) => keys[provider],
    };
  }

  it("resolves ollama-cloud key from AuthStorage", async () => {
    const store = mockAuthStore({ "ollama-cloud": "cloud-api-key" });
    const config = await resolveConfigAsync(store);
    expect(config.apiKey).toBe("cloud-api-key");
    expect(config.mode).toBe("cloud");
    expect(config.authSource).toBe("auth-json-ollama-cloud");
  });

  it("falls back to ollama key when ollama-cloud is not available", async () => {
    const store = mockAuthStore({ ollama: "local-key" });
    const config = await resolveConfigAsync(store);
    expect(config.apiKey).toBe("local-key");
    expect(config.mode).toBe("cloud");
    expect(config.authSource).toBe("auth-json-ollama");
  });

  it("prefers ollama-cloud over ollama when both available", async () => {
    const store = mockAuthStore({ "ollama-cloud": "cloud-key", ollama: "local-key" });
    const config = await resolveConfigAsync(store);
    expect(config.apiKey).toBe("cloud-key");
    expect(config.authSource).toBe("auth-json-ollama-cloud");
  });

  it("falls back to env var when no AuthStorage keys", async () => {
    const store = mockAuthStore({});
    const config = await resolveConfigAsync(store, { envKey: "env-key" });
    expect(config.apiKey).toBe("env-key");
    expect(config.authSource).toBe("env-var");
    expect(config.mode).toBe("cloud");
  });

  it("defaults to local mode with 'ollama' key when nothing configured", async () => {
    const store = mockAuthStore({});
    const config = await resolveConfigAsync(store, { envKey: undefined });
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
    expect(config.authSource).toBe("default");
  });

  it("uses OLLAMA_API_BASE for cloud endpoint", async () => {
    const store = mockAuthStore({ "ollama-cloud": "my-key" });
    const config = await resolveConfigAsync(store, { envBase: "https://custom.example.com" });
    expect(config.baseUrl).toBe("https://custom.example.com");
  });

  it("treats 'ollama' key from AuthStorage as local mode", async () => {
    const store = mockAuthStore({ ollama: "ollama" });
    const config = await resolveConfigAsync(store);
    expect(config.apiKey).toBe("ollama");
    expect(config.mode).toBe("local");
  });

  it("local mode uses normalized OLLAMA_HOST", async () => {
    process.env.OLLAMA_HOST = "gpu.lan";
    const store = mockAuthStore({});
    const config = await resolveConfigAsync(store, { envKey: undefined });
    expect(config.mode).toBe("local");
    expect(config.baseUrl).toBe("http://gpu.lan:11434");
  });
});

// ════════════════════════════════════════════════════════════════
// describeAuthSource
// ════════════════════════════════════════════════════════════════

describe("describeAuthSource", () => {
  it("describes auth-json-ollama-cloud", () => {
    const config: OllamaConfig = { mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: "key", authSource: "auth-json-ollama-cloud" };
    expect(describeAuthSource(config)).toContain("ollama-cloud");
  });

  it("describes auth-json-ollama", () => {
    const config: OllamaConfig = { mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: "key", authSource: "auth-json-ollama" };
    expect(describeAuthSource(config)).toContain("ollama");
  });

  it("describes env-var", () => {
    const config: OllamaConfig = { mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: "key", authSource: "env-var" };
    expect(describeAuthSource(config)).toContain("OLLAMA_API_KEY");
  });

  it("describes default as local", () => {
    const config: OllamaConfig = { mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "ollama", authSource: "default" };
    expect(describeAuthSource(config)).toContain("local");
  });

  it("describes runtime override", () => {
    const config: OllamaConfig = { mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: "key", authSource: "runtime-override" };
    expect(describeAuthSource(config)).toContain("runtime override");
  });
});

// ════════════════════════════════════════════════════════════════
// authHeaders
// ════════════════════════════════════════════════════════════════

describe("authHeaders", () => {
  it("includes Bearer header for cloud apiKey", () => {
    const headers = authHeaders({ mode: "cloud", baseUrl: DEFAULT_CLOUD_URL, apiKey: "my-key" });
    expect(headers["Authorization"]).toBe("Bearer my-key");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("omits Bearer header for default 'ollama' key", () => {
    const headers = authHeaders({ mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "ollama" });
    expect(headers["Authorization"]).toBeUndefined();
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("omits Bearer header for empty string key", () => {
    const headers = authHeaders({ mode: "local", baseUrl: DEFAULT_LOCAL_URL, apiKey: "" });
    expect(headers["Authorization"]).toBeUndefined();
  });
});