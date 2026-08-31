import { env, pipeline } from '@huggingface/transformers';

import type { SpeechToTextModelId } from '../modelRegistry';

import { ONNX_WASM_BASE_URL } from './RuntimeAssetUrls';

import {
  normalizeDownloadProgress,
  postWorkerError,
  postWorkerProgress,
  postWorkerResult,
  type RpcWorkerRequest,
} from './workerProtocol';

const HF_PROXY = 'https://hf-proxy.doublethew.workers.dev/';
  
interface LoadPayload {
  modelId: SpeechToTextModelId;
}

interface TranscribePayload {
  audio: Float32Array;
  sampleRate: number;
}

interface AsrResult {
  text: string;
  chunks?: Array<{ text: string; timestamp: [number, number] }>;
}

interface AsrPipeline {
  (audio: Float32Array, options: Record<string, unknown>): Promise<AsrResult>;
  dispose?: () => Promise<void> | void;
}

const MODEL_REPOSITORIES: Record<SpeechToTextModelId, string> = {
  'moonshine-tiny-q8': 'onnx-community/moonshine-tiny-ONNX',
  'whisper-tiny-en-q8': 'onnx-community/whisper-tiny.en',
};

const HF_PROXY = 'https://hf-proxy.doublethew.workers.dev/';

env.remoteHost = HF_PROXY;
env.remotePathTemplate = '{model}/resolve/{revision}/{file}';

if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.wasmPaths = ONNX_WASM_BASE_URL;
}
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

env.remoteHost = 'https://hf-proxy.doublethew.workers.dev';

env.remotePathTemplate = '{model}/resolve/{revision}/{file}';

let transcriber: AsrPipeline | null = null;
let loadedModel: SpeechToTextModelId | null = null;
let queue = Promise.resolve();

async function loadModel(id: number, payload: LoadPayload): Promise<void> {
  if (transcriber && loadedModel === payload.modelId) {
    postWorkerResult(id, { modelId: payload.modelId, cached: true });
    return;
  }
  await transcriber?.dispose?.();
  transcriber = null;
  const model = await pipeline(
    'automatic-speech-recognition',
    MODEL_REPOSITORIES[payload.modelId],
    {
      device: 'wasm',
      dtype: 'q8',
      progress_callback: (progress: unknown) => {
        postWorkerProgress(id, normalizeDownloadProgress(progress));
      },
    },
  );
  transcriber = model as unknown as AsrPipeline;
  loadedModel = payload.modelId;
  postWorkerResult(id, { modelId: payload.modelId, cached: false });
}

async function transcribe(id: number, payload: TranscribePayload): Promise<void> {
  if (!transcriber) throw new Error('Speech recognition is not installed.');
  if (!(payload.audio instanceof Float32Array) || payload.audio.length === 0) {
    throw new Error('Speech recognition received no microphone audio.');
  }
  const startedAt = performance.now();
  const result = await transcriber(payload.audio, {
    sampling_rate: payload.sampleRate,
    return_timestamps: loadedModel === 'whisper-tiny-en-q8',
  });
  postWorkerResult(id, {
    text: result.text.trim(),
    chunks: result.chunks ?? [],
    elapsedMs: performance.now() - startedAt,
  });
}

async function handle(message: RpcWorkerRequest): Promise<void> {
  if (message.operation === 'load') {
    await loadModel(message.id, message.payload as LoadPayload);
  } else if (message.operation === 'transcribe') {
    await transcribe(message.id, message.payload as TranscribePayload);
  } else if (message.operation === 'dispose') {
    await transcriber?.dispose?.();
    transcriber = null;
    loadedModel = null;
    postWorkerResult(message.id, undefined);
  } else {
    throw new Error(`Unknown STT worker operation: ${message.operation}`);
  }
}

globalThis.addEventListener('message', (event: MessageEvent<RpcWorkerRequest>) => {
  queue = queue.then(() => handle(event.data)).catch((error: unknown) => {
    postWorkerError(event.data.id, error);
  });
});
