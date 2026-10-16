/**
 * Auth resolution for Ollama.
 *
 * Auth priority (per pi convention, using AuthStorage):
 *   1. Runtime override (CLI --api-key flag, via AuthStorage)
 *   2. auth.json "ollama-cloud" credential (via AuthStorage)
 *   3. auth.json "ollama" credential (via AuthStorage)
 *   4. OLLAMA_API_KEY environment variable (via AuthStorage)
 *   5. OLLAMA_API_BASE env var for custom cloud endpoint
 *   6. Default "ollama" (local, no auth)
 *
 * Uses pi's built-in AuthStorage for file-lock-safe auth resolution.
 * Falls back to direct readFileSync only for synchronous startup cache.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { getOllamaHost, DEFAULT_LOCAL_OLLAMA_URL } from "./discovery.js";

const AUTH_PATH = join(homedir(), ".pi", "agent", "auth.json");

export const DEFAULT_LOCAL_URL = DEFAULT_LOCAL_OLLAMA_URL;
export const DEFAULT_CLOUD_URL = "https://ollama.com";

export interface OllamaConfig {
  mode: "local" | "cloud";
  baseUrl: string;
  apiKey: string;
  /** Which auth source was used (for logging/debugging) */
  authSource?: "runtime-override" | "auth-json-ollama-cloud" | "auth-json-ollama" | "env-var" | "default";
}

/**
 * Read the "ollama" or "ollama-cloud" credential from a JSON auth file.
 * Accepts an explicit path for testing; defaults to the real auth.json.
 *
 * NOTE: This is kept for synchronous cache-first startup fallback.
 * At runtime, AuthStorage.getApiKey() should be preferred for full
 * resolution (including runtime overrides and file locking).
 */
export function readOllamaAuthFromJson(
  authPath: string = AUTH_PATH,
): { type: "api_key"; key: string } | undefined {
  try {
    const data = readFileSync(authPath, "utf-8");
    const parsed = JSON.parse(data);
    // Check both "ollama-cloud" (preferred for cloud) and "ollama"
    const cred = parsed?.["ollama-cloud"] || parsed?.ollama;
    if (cred?.type === "api_key" && typeof cred.key === "string") {
      return cred;
    }
  } catch {}
  return undefined;
}

/**
 * Resolve which key source was used (for logging/debugging).
 * Returns a human-readable label matching the OllamaConfig.authSource field.
 */
export function describeAuthSource(config: OllamaConfig): string {
  switch (config.authSource) {
    case "runtime-override":
      return `runtime override (--api-key flag)`;
    case "auth-json-ollama-cloud":
      return `auth.json "ollama-cloud" entry`;
    case "auth-json-ollama":
      return `auth.json "ollama" entry`;
    case "env-var":
      return `OLLAMA_API_KEY env var`;
    case "default":
    default:
      return config.mode === "local" ? "local (no auth)" : "default";
  }
}

/**
 * Resolve the active Ollama config using AuthStorage (async, preferred).
 *
 * Priority for API key:
 *   1. Runtime override (CLI --api-key flag, via AuthStorage)
 *   2. auth.json "ollama-cloud" credential (via AuthStorage)
 *   3. auth.json "ollama" credential (fallback, via AuthStorage)
 *   4. OLLAMA_API_KEY environment variable (via AuthStorage)
 *   5. default "ollama" (works for local / unauthenticated)
 */
export async function resolveConfigAsync(
  authStorage?: { getApiKey: (provider: string) => Promise<string | undefined> },
  options?: { envKey?: string; envBase?: string },
): Promise<OllamaConfig> {
  const ollamaCloudKey = await authStorage?.getApiKey("ollama-cloud");
  const ollamaKey = await authStorage?.getApiKey("ollama");

  // Priority: runtime override > auth.json ollama-cloud > auth.json ollama > env var
  let apiKey: string;
  let authSource: OllamaConfig["authSource"];

  if (ollamaCloudKey) {
    // AuthStorage already handled runtime overrides and env vars
    // We need to determine the source more precisely
    apiKey = ollamaCloudKey;
    authSource = "auth-json-ollama-cloud";
  } else if (ollamaKey && ollamaKey !== "ollama") {
    apiKey = ollamaKey;
    authSource = "auth-json-ollama";
  } else {
    const envKey = options?.envKey ?? process.env.OLLAMA_API_KEY;
    if (envKey) {
      apiKey = envKey;
      authSource = "env-var";
    } else {
      apiKey = "ollama";
      authSource = "default";
    }
  }

  // OLLAMA_API_BASE env var for custom cloud endpoint
  const cloudBaseUrl = (options?.envBase ?? process.env.OLLAMA_API_BASE?.replace(/\/+$/, "")) || DEFAULT_CLOUD_URL;
  const mode: "local" | "cloud" = apiKey !== "ollama" ? "cloud" : "local";
  const baseUrl = mode === "cloud" ? cloudBaseUrl : getOllamaHost();

  return { mode, baseUrl, apiKey, authSource };
}

/**
 * Synchronous config resolution (for cache-first startup).
 * Uses direct readFileSync — not file-lock-safe, but sufficient
 * for cache-only startup before AuthStorage resolves.
 */
export function resolveConfig(options?: {
  authPath?: string;
  envKey?: string;
  envBase?: string;
}): OllamaConfig {
  const stored = readOllamaAuthFromJson(options?.authPath);
  const envKey = options?.envKey ?? process.env.OLLAMA_API_KEY;
  const cloudBaseUrl = (options?.envBase ?? process.env.OLLAMA_API_BASE?.replace(/\/+$/, "")) || DEFAULT_CLOUD_URL;

  if (stored) {
    const apiKey = stored.key;
    const mode: "local" | "cloud" = apiKey !== "ollama" ? "cloud" : "local";
    const baseUrl = mode === "cloud" ? cloudBaseUrl : getOllamaHost();
    // Determine source: ollama-cloud vs ollama
    const authSource: OllamaConfig["authSource"] =
      apiKey !== "ollama" ? "auth-json-ollama-cloud" : "auth-json-ollama";
    return { mode, baseUrl, apiKey, authSource };
  }

  if (envKey) {
    return { mode: "cloud", baseUrl: cloudBaseUrl, apiKey: envKey, authSource: "env-var" };
  }

  return { mode: "local", baseUrl: getOllamaHost(), apiKey: "ollama", authSource: "default" };
}

/**
 * Return auth headers for Ollama API requests.
 */
export function authHeaders(config?: OllamaConfig): Record<string, string> {
  const resolvedConfig = config ?? resolveConfig();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (resolvedConfig.apiKey && resolvedConfig.apiKey !== "ollama") {
    headers["Authorization"] = `Bearer ${resolvedConfig.apiKey}`;
  }
  return headers;
}