import type {
  Conversation,
  Engine,
  Message,
} from '@litert-lm/core';
import { getLocalModel, type LanguageModelId } from '../modelRegistry';
import {
  BRAIN_SYSTEM_PROMPT,
  throwIfBrainAborted,
  type BrainConversationMessage,
  type BrainGenerationOptions,
  type BrainLoadProgress,
  type LocalBrainRuntime,
} from './BrainContracts';
import { LITERT_LM_WASM_BASE_URL } from './RuntimeAssetUrls';

const MODEL_CACHE = 'litertlm-models-v1';
const GEMMA_E2B_REVISION = '9262660a1676eed6d0c477ab1a86344430854664';
const GEMMA_E4B_REVISION = 'f7ad3343bd6ebc9607f4dc3bc4f2398bd5749bc5';

const HF_PROXY = 'https://hf-proxy.doublethew.workers.dev';

const LITERT_MODEL_URLS: Partial<Record<LanguageModelId, string>> = {
  'gemma-4-e2b-it-litert-web':
    `${HF_PROXY}/litert-community/gemma-4-E2B-it-litert-lm/resolve/${GEMMA_E2B_REVISION}/gemma-4-E2B-it-web.litertlm`,

  'gemma-4-e4b-it-litert-web':
    `${HF_PROXY}/litert-community/gemma-4-E4B-it-litert-lm/resolve/${GEMMA_E4B_REVISION}/gemma-4-E4B-it-web.litertlm`,
};

interface ModelStreamResult {
  stream: ReadableStream<Uint8Array>;
  cacheWrite: Promise<void>;
}

function modelUrl(modelId: LanguageModelId): string {
  const url = LITERT_MODEL_URLS[modelId];
  if (!url) throw new Error('This model is not supported by the LiteRT-LM browser runtime.');
  return url;
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content;
  if (!Array.isArray(message.content)) return '';
  return message.content
    .filter((part): part is Extract<(typeof message.content)[number], { type: 'text' }> =>
      part.type === 'text')
    .map((part) => part.text)
    .join('');
}

async function openModelCache(): Promise<Cache | null> {
  if (typeof globalThis.caches === 'undefined') return null;
  try {
    return await globalThis.caches.open(MODEL_CACHE);
  } catch {
    return null;
  }
}

function progressStream(
  source: ReadableStream<Uint8Array>,
  expectedBytes: number,
  cached: boolean,
  onProgress: ((progress: BrainLoadProgress) => void) | undefined,
  signal: AbortSignal | undefined,
): ReadableStream<Uint8Array> {
  let loadedBytes = 0;
  const transformer = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      loadedBytes += chunk.byteLength;
      onProgress?.({
        progress: Math.min(0.9, 0.04 + (loadedBytes / Math.max(1, expectedBytes)) * 0.86),
        text: `${cached ? 'Reading cached' : 'Downloading'} Gemma weights · ${(
          loadedBytes / 1_000_000
        ).toFixed(0)} / ${(expectedBytes / 1_000_000).toFixed(0)} MB`,
      });
      controller.enqueue(chunk);
    },
  });
  return source.pipeThrough(transformer, signal ? { signal } : undefined);
}

async function prepareModelStream(
  modelId: LanguageModelId,
  onProgress: ((progress: BrainLoadProgress) => void) | undefined,
  signal: AbortSignal | undefined,
): Promise<ModelStreamResult> {
  const url = modelUrl(modelId);
  const expectedBytes = getLocalModel(modelId).artifact.estimatedDownloadBytes;
  const cache = await openModelCache();
  throwIfBrainAborted(signal);

  const cachedResponse = await cache?.match(url);
  if (cachedResponse?.body) {
    const total = Number(cachedResponse.headers.get('content-length')) || expectedBytes;
    return {
      stream: progressStream(cachedResponse.body, total, true, onProgress, signal),
      cacheWrite: Promise.resolve(),
    };
  }

  const response = await fetch(url, { signal, credentials: 'omit' });
  if (!response.ok || !response.body) {
    throw new Error(`Gemma model download failed (${response.status} ${response.statusText}).`);
  }
  const total = Number(response.headers.get('content-length')) || expectedBytes;
  if (!cache) {
    return {
      stream: progressStream(response.body, total, false, onProgress, signal),
      cacheWrite: Promise.resolve(),
    };
  }

  const [engineBody, cacheBody] = response.body.tee();
  const cacheWrite = cache
    .put(
      url,
      new Response(cacheBody, {
        headers: {
          'content-type': 'application/octet-stream',
          'content-length': String(total),
        },
      }),
    )
    .catch(() => undefined);
  return {
    stream: progressStream(engineBody, total, false, onProgress, signal),
    cacheWrite,
  };
}

export class LiteRtLlmRuntime implements LocalBrainRuntime {
  private engine: Engine | null = null;
  private modelId: LanguageModelId | null = null;
  private activeConversation: Conversation | null = null;
  private activeReader: ReadableStreamDefaultReader<Message> | null = null;
  private lastRuntimeStats = 'LiteRT-LM local brain is not loaded.';

  async isCached(modelId: LanguageModelId): Promise<boolean> {
    const cache = await openModelCache();
    return (await cache?.match(modelUrl(modelId))) !== undefined;
  }

  async load(
    modelId: LanguageModelId,
    onProgress?: (progress: BrainLoadProgress) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    modelUrl(modelId);
    await this.dispose();
    throwIfBrainAborted(signal);
    onProgress?.({ progress: 0.01, text: 'Loading local LiteRT-LM WebGPU runtime…' });

    const { Engine: LiteRtEngine, getOrLoadGlobalLiteRtLm } =
      await import('@litert-lm/core');
    await getOrLoadGlobalLiteRtLm(LITERT_LM_WASM_BASE_URL);
    throwIfBrainAborted(signal);
    onProgress?.({ progress: 0.04, text: 'Preparing Gemma model weights…' });

    const { stream, cacheWrite } = await prepareModelStream(
      modelId,
      onProgress,
      signal,
    );
    let engine: Engine | null = null;
    try {
      engine = await LiteRtEngine.create({
        model: stream,
        mainExecutorSettings: { maxNumTokens: 4096 },
        benchmarkEnabled: true,
      });
      throwIfBrainAborted(signal);
      this.engine = engine;
      this.modelId = modelId;
      engine = null;
      await cacheWrite;
      onProgress?.({ progress: 1, text: 'Gemma ready · local WebGPU' });
    } finally {
      if (engine) await engine.delete().catch(() => undefined);
    }
  }

  async generate(
    history: readonly BrainConversationMessage[],
    options: BrainGenerationOptions,
  ): Promise<string> {
    let reply = '';
    for await (const token of this.stream(history, options.signal)) {
      reply += token;
      options.onToken(token);
    }
    return reply.trim();
  }

  async *stream(
    history: readonly BrainConversationMessage[],
    signal?: AbortSignal,
  ): AsyncGenerator<string, void, void> {
    if (!this.engine || !this.modelId) {
      throw new Error('Install the local language model before talking.');
    }
    if (this.activeConversation) {
      throw new Error('The local brain is already generating a response.');
    }
    throwIfBrainAborted(signal);

    let userIndex = -1;
    for (let index = history.length - 1; index >= 0; index -= 1) {
      if (history[index].role === 'user') {
        userIndex = index;
        break;
      }
    }
    if (userIndex < 0) throw new Error('A user message is required for local generation.');
    const userMessage = history[userIndex];
    const prefaceHistory = history.slice(Math.max(0, userIndex - 7), userIndex);
    const conversation = await this.engine.createConversation({
      sessionConfig: {
        samplerParams: { temperature: 0.62, p: 0.9, k: 40 },
        maxOutputTokens: 128,
      },
      preface: {
        messages: [
          { role: 'system', content: BRAIN_SYSTEM_PROMPT },
          ...prefaceHistory,
        ],
        extra_context: { enable_thinking: false },
      },
      prefillPrefaceOnInit: true,
      filterChannelContentFromKvCache: true,
    });
    this.activeConversation = conversation;

    const responseStream = conversation.sendMessageStreaming(userMessage.content);
    const reader = responseStream.getReader();
    this.activeReader = reader;
    const abort = (): void => {
      conversation.cancel();
      void reader.cancel(signal?.reason).catch(() => undefined);
    };
    signal?.addEventListener('abort', abort, { once: true });
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done || signal?.aborted) break;
        const token = messageText(value);
        if (token) yield token;
      }
      if (!signal?.aborted) {
        const stats = await conversation.getBenchmarkInfo();
        this.lastRuntimeStats = [
          `LiteRT-LM ${this.modelId}`,
          `${stats.lastPrefillTokensPerSecond.toFixed(1)} prefill tok/s`,
          `${stats.lastDecodeTokensPerSecond.toFixed(1)} decode tok/s`,
          `${(stats.timeToFirstTokenInSecond * 1_000).toFixed(0)} ms first token`,
        ].join(' · ');
      }
    } finally {
      signal?.removeEventListener('abort', abort);
      if (this.activeReader === reader) this.activeReader = null;
      if (this.activeConversation === conversation) this.activeConversation = null;
      await reader.cancel().catch(() => undefined);
      await conversation.delete().catch(() => undefined);
    }
  }

  interrupt(): void {
    this.activeConversation?.cancel();
    void this.activeReader?.cancel('interrupted').catch(() => undefined);
  }

  async runtimeStats(): Promise<string> {
    return this.lastRuntimeStats;
  }

  async dispose(): Promise<void> {
    const reader = this.activeReader;
    const conversation = this.activeConversation;
    this.activeReader = null;
    this.activeConversation = null;
    conversation?.cancel();
    await reader?.cancel('disposed').catch(() => undefined);
    await conversation?.delete().catch(() => undefined);
    const engine = this.engine;
    this.engine = null;
    this.modelId = null;
    if (engine) await engine.delete().catch(() => undefined);
    this.lastRuntimeStats = 'LiteRT-LM local brain is not loaded.';
  }
}
