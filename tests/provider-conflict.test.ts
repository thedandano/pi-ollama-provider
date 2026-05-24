/**
 * Test: ollama provider should use unique API name to avoid conflicts
 *
 * Bug: When ollama extension registers streamSimple with "openai-completions" API,
 * it overwrites other providers (openrouter, azure) that share the same API.
 *
 * The apiProviderRegistry is a Map keyed by API name. When multiple providers
 * use the same API (e.g., "openai-completions"), the last registration wins.
 * Extensions load after built-in providers, so ollama was overwriting openrouter.
 *
 * Fix: Ollama uses "ollama-native" API name to avoid conflicts while still
 * enabling native /api/chat streaming with streamSimple.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { hasToolSupport, hasVision, assembleModelsFromCache } from "../extensions/pi-ollama-provider/discovery.js";

const __dirname = join(fileURLToPath(new URL(".", import.meta.url)), "..");

describe("Provider registration", () => {
  it("should use ollama-native API instead of openai-completions", () => {
    // Regression test: verify the source code uses OLLAMA_NATIVE_API constant
    const indexPath = join(__dirname, "extensions/pi-ollama-provider/index.ts");
    const source = readFileSync(indexPath, "utf-8");

    // Should define the custom API constant
    expect(source).toContain('OLLAMA_NATIVE_API = "ollama-native"');

    // Should have streamSimple registration with ollama-native API
    // Match only the native streaming block (ends at the closing brace + else)
    const nativeStreamingMatch = source.match(
      /pi\.registerProvider\(LOCAL_PROVIDER_NAME,\s*\{[\s\S]*?streamSimple:\s*createNativeStreamSimple[\s\S]*?^\s+\}\);/m,
    );

    // This match MUST succeed - native streaming must be registered
    expect(nativeStreamingMatch).not.toBeNull();
    expect(nativeStreamingMatch![0]).toContain("api: OLLAMA_NATIVE_API");
    // The native streaming block should NOT contain openai-completions
    // (that's only in the else branch for compat mode)
    expect(nativeStreamingMatch![0]).not.toContain('api: "openai-completions"');
  });

  it("should include tools property in model registration", () => {
    // Verify tools property is passed to piModels
    const indexPath = join(__dirname, "extensions/pi-ollama-provider/index.ts");
    const source = readFileSync(indexPath, "utf-8");

    // Check for tools: m.toolSupport - must appear at least twice (both registration paths)
    const toolSupportMatches = source.matchAll(/tools:\s*m\.toolSupport/g);
    const matchCount = Array.from(toolSupportMatches).length;

    expect(matchCount).toBeGreaterThanOrEqual(2);
  });
});

describe("GLM tool capability", () => {
  it("should detect GLM models as tool-capable", () => {
    // Test hasToolSupport function with GLM models (capabilities first, then model name for pattern matching)
    expect(hasToolSupport(["tools", "completion"], {}, undefined, "zai/glm-5.1")).toBe(true);
    expect(hasToolSupport(["tools", "completion"], {}, undefined, "glm-5.1")).toBe(true);
    // GLM family in family parameter
    expect(hasToolSupport([], {}, "glm")).toBe(true);
    expect(hasToolSupport([], {}, "glm-5.1")).toBe(true);

    // Test via assembleModelsFromCache with raw showResponses
    const rawCacheData = {
      version: 2 as const,
      timestamp: Date.now(),
      tagsModels: [{ name: "glm-5.1:cloud", model: "glm-5.1:cloud", size: 0, modified_at: "", digest: "" }],
      showResponses: {
        "glm-5.1:cloud": {
          capabilities: ["completion", "tools"],
        },
      },
      mode: "cloud" as const,
    };

    const models = assembleModelsFromCache(rawCacheData, "cloud");
    expect(models.length).toBeGreaterThan(0);
    expect(models[0].toolSupport).toBe(true);
  });
});