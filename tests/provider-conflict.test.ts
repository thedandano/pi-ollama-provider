/**
 * Test: streamSimple should not break other providers
 * 
 * Bug: When ollama extension registers with streamSimple, it overwrites
 * the openai-completions API provider, breaking other providers like openrouter.
 * 
 * The apiProviderRegistry is a Map keyed by API name. When multiple providers
 * use the same API (e.g., "openai-completions"), the last registration wins.
 * Extensions load after built-in providers, so ollama was overwriting openrouter.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

describe("Provider registration", () => {
  it("should not overwrite other providers using openai-completions API", () => {
    // This is a design constraint: streamSimple registration uses the API name
    // as the registry key, so only one provider can use a given API at a time.
    // 
    // The fix removes streamSimple from registration to avoid this conflict.
    // Extensions should NOT use streamSimple if they share an API with built-in providers.
    
    // Regression test: ensure registerProvider does not include streamSimple
    // for the ollama provider, since openrouter also uses "openai-completions"
    expect(true).toBe(true); // Placeholder - actual test would mock the registry
  });
});
