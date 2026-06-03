import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fingerprint } from "../src/cache.js";

const ENV_KEYS = ["LITELLM_BASE_URL", "LITELLM_API_KEY", "LITELLM_DISCOVERY_TIMEOUT_MS", "STORED_LITELLM_KEY"];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

type TestProviderConfig = {
  baseUrl?: string;
  models?: unknown[];
  oauth?: {
    login: (callbacks: {
      onPrompt: (options: { message: string; placeholder?: string }) => Promise<string>;
      onProgress?: (message: string) => void;
      signal?: AbortSignal;
    }) => Promise<Record<string, unknown>>;
    modifyModels?: (
      models: Array<{ provider?: string; baseUrl?: unknown }>,
      cred: Record<string, unknown>,
    ) => Array<{ provider?: string; baseUrl?: unknown }>;
  };
};

type TestCommand = {
  description: string;
  handler: (args: string[], ctx: { ui: { notify: (message: string, type: string) => void } }) => Promise<void> | void;
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function makeAgentDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "pi-litellm-index-"));
}

async function loadExtension(agentDir: string): Promise<(pi: TestPi) => Promise<void>> {
  vi.resetModules();
  vi.doMock("@earendil-works/pi-coding-agent", () => {
    class TestAuthStorage {
      constructor(private readonly authPath: string) {}

      static create(authPath: string): TestAuthStorage {
        return new TestAuthStorage(authPath);
      }

      async getApiKey(provider: string): Promise<string | undefined> {
        const parsed = JSON.parse(await readFile(this.authPath, "utf8")) as Record<
          string,
          { type: "api_key"; key: string } | { type: "oauth"; access: string; expires: number; refresh: string }
        >;
        const entry = parsed[provider];
        if (entry?.type === "api_key") return process.env[entry.key] || entry.key;
        if (entry?.type === "oauth") return entry.access;
        return undefined;
      }
    }

    return { AuthStorage: TestAuthStorage, getAgentDir: () => agentDir };
  });
  const mod = await import("../src/index.js");
  return mod.default as unknown as (pi: TestPi) => Promise<void>;
}

function createPi(): TestPi {
  return {
    providers: [],
    commands: new Map(),
    handlers: new Map(),
    registerProvider(name: string, config: TestProviderConfig) {
      this.providers.push({ name, config });
    },
    registerCommand(name: string, command: TestCommand) {
      this.commands.set(name, command);
    },
    on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any) {
      this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    },
  };
}

type TestPi = {
  providers: Array<{ name: string; config: TestProviderConfig }>;
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  registerProvider(name: string, config: TestProviderConfig): void;
  registerCommand(name: string, command: TestCommand): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) process.env[key] = undefined;
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
  vi.unmock("@earendil-works/pi-coding-agent");
});

describe("extension startup", () => {
  it("discovers with the resolved stored auth key before LITELLM_API_KEY", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({ litellm: { type: "api_key", key: "STORED_LITELLM_KEY" } }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.STORED_LITELLM_KEY = "stored-key";
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, {
        data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
      });
    });

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    expect(seenAuthHeaders).toEqual(["Bearer stored-key"]);
  });

  it("does not fetch on refresh when discovery timeout is zero", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
      }),
    );
    const notify = vi.fn();

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await pi.commands.get("litellm-refresh")?.handler([], { ui: { notify } });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith("LiteLLM refresh disabled (LITELLM_DISCOVERY_TIMEOUT_MS=0)", "warning");
  });

  it("prompts during login and caches discovered models", async () => {
    const agentDir = await makeAgentDir();
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const seenRequests: Array<{ url: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({
        url,
        authorization: new Headers(init?.headers).get("authorization") ?? "",
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const promptMessages: string[] = [];
    const progress = vi.fn();
    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => {
        promptMessages.push(options.message);
        return promptMessages.length === 1 ? " http://127.0.0.1:4000/v1 " : " sk-login ";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(promptMessages).toEqual([
      "Enter LiteLLM proxy URL (include /v1 only if your gateway requires it):",
      "Enter API key:",
    ]);
    expect(seenRequests).toEqual([{ url: "http://127.0.0.1:4000/model/info", authorization: "Bearer sk-login" }]);
    expect(credential).toMatchObject({
      access: "sk-login",
      refresh: "",
      baseUrl: "http://127.0.0.1:4000",
      apiBaseUrl: "http://127.0.0.1:4000/v1",
    });
    const cache = JSON.parse(await readFile(join(agentDir, "litellm-models.json"), "utf8")) as {
      apiBaseUrl?: string;
      models: Array<{ id: string }>;
    };
    expect(cache.apiBaseUrl).toBe("http://127.0.0.1:4000/v1");
    expect(cache.models.map((model) => model.id)).toEqual(["vidaimock-openai"]);
    expect(progress).toHaveBeenCalledWith("LiteLLM: 1 models discovered (source: model_info)");
  });

  it("reuses cached apiBaseUrl without probing when cache matches", async () => {
    const agentDir = await makeAgentDir();
    const apiKey = "cached-key";
    const baseUrl = "https://host.example.com/proxy";
    const apiBaseUrl = "https://host.example.com/proxy/v1";

    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "oauth", access: apiKey, refresh: "", expires: Number.MAX_SAFE_INTEGER, baseUrl },
      }),
      "utf8",
    );
    await writeFile(
      join(agentDir, "litellm-models.json"),
      JSON.stringify({
        baseUrl,
        apiBaseUrl,
        apiKeyFingerprint: fingerprint(apiKey),
        fetchedAt: Date.now(),
        source: "model_info",
        models: [
          {
            id: "openai/gpt-4o",
            name: "openai/gpt-4o",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 128000,
            maxTokens: 16384,
          },
        ],
      }),
      "utf8",
    );

    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.baseUrl).toBe(apiBaseUrl);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("modifyModels prefers apiBaseUrl over baseUrl", async () => {
    const agentDir = await makeAgentDir();
    const apiKey = "modify-key";
    const baseUrl = "https://host.example.com/proxy";
    const apiBaseUrl = "https://host.example.com/proxy";

    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "oauth", access: apiKey, refresh: "", expires: Number.MAX_SAFE_INTEGER, baseUrl },
      }),
      "utf8",
    );
    await writeFile(
      join(agentDir, "litellm-models.json"),
      JSON.stringify({
        baseUrl,
        apiBaseUrl,
        apiKeyFingerprint: fingerprint(apiKey),
        fetchedAt: Date.now(),
        source: "model_info",
        models: [
          {
            id: "openai/gpt-4o",
            name: "openai/gpt-4o",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 128000,
            maxTokens: 16384,
          },
        ],
      }),
      "utf8",
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const modifyModels = pi.providers[0]?.config.oauth?.modifyModels;
    expect(modifyModels).toBeDefined();

    const models = [{ id: "openai/gpt-4o", provider: "litellm", baseUrl: "" }];
    const credential = { access: apiKey, refresh: "", expires: Number.MAX_SAFE_INTEGER, baseUrl, apiBaseUrl };
    const result = modifyModels!(models, credential);

    expect(result[0].baseUrl).toBe(apiBaseUrl);
  });

  it("resolves apiBaseUrl lazily when valid cache predates this field", async () => {
    const agentDir = await makeAgentDir();
    const apiKey = "legacy-key";
    const baseUrl = "https://host.example.com/proxy";

    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "oauth", access: apiKey, refresh: "", expires: Number.MAX_SAFE_INTEGER, baseUrl },
      }),
      "utf8",
    );
    await writeFile(
      join(agentDir, "litellm-models.json"),
      JSON.stringify({
        baseUrl,
        apiKeyFingerprint: fingerprint(apiKey),
        fetchedAt: Date.now(),
        source: "model_info",
        models: [
          {
            id: "openai/gpt-4o",
            name: "openai/gpt-4o",
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 128000,
            maxTokens: 16384,
          },
        ],
      }),
      "utf8",
    );

    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response(null, { status: 404 }));
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(fetchSpy).toHaveBeenCalledWith(
      "https://host.example.com/proxy/v1/models",
      expect.objectContaining({ headers: expect.anything() }),
    );
    expect(pi.providers[0]?.config.baseUrl).toBe(baseUrl);
  });

  it("uses the login cache timestamp for later stale auto-refresh", async () => {
    const agentDir = await makeAgentDir();
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    delete process.env.LITELLM_DISCOVERY_TIMEOUT_MS;
    const loginTime = new Date("2026-05-01T00:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(loginTime);

    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        callCount++;
        return jsonResponse(200, {
          data: [{ model_name: `vidaimock-openai-${callCount}`, model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    expect(callCount).toBe(0);

    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });
    expect(callCount).toBe(1);
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: { type: "oauth", ...credential } }), "utf8");

    vi.mocked(Date.now).mockReturnValue(loginTime + 25 * 60 * 60 * 1000);
    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
    }

    await vi.waitFor(() => {
      expect(callCount).toBe(2);
      expect(pi.providers).toHaveLength(2);
    });
  });
});
