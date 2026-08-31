import { env as transformersEnv } from '@huggingface/transformers';
import {
  env as kokoroEnv,
  KokoroTTS,
  TextSplitterStream,
} from 'kokoro-js';

import { normalizeSpokenText } from '../SpokenTextNormalizer';
import type { TextToSpeechModelId } from '../modelRegistry';
import { ONNX_WASM_BASE_URL } from './RuntimeAssetUrls';

import {
  normalizeDownloadProgress,
  postWorkerError,
  postWorkerProgress,
  postWorkerResult,
  type RpcWorkerRequest,
} from './workerProtocol';

const HF_PROXY = 'https://hf-proxy.doublethew.workers.dev';

kokoroEnv.wasmPaths = ONNX_WASM_BASE_URL;

interface LoadPayload {
  modelId: TextToSpeechModelId;
}

interface SynthesizePayload {
  text: string;
  voice: string;
  speed: number;
}

// Kokoro/ONNX WASM files
kokoroEnv.wasmPaths = ONNX_WASM_BASE_URL;

// Hugging Face model files
transformersEnv.remoteHost = HF_PROXY;
transformersEnv.remotePathTemplate = '{model}/resolve/{revision}';

transformersEnv.allowLocalModels = false;
transformersEnv.allowRemoteModels = true;
transformersEnv.useBrowserCache = true;

let tts: KokoroTTS | null = null;
let loadedModel: TextToSpeechModelId | null = null;
let loadedBackend: 'webgpu' | 'wasm' = 'wasm';
let queue = Promise.resolve();

async function loadModel(id: number, payload: LoadPayload): Promise<void> {
  if (tts && loadedModel === payload.modelId) {
    postWorkerResult(id, { modelId: payload.modelId, cached: true });
    return;
  }
  await tts?.model.dispose();
  const webgpu = payload.modelId === 'kokoro-82m-fp32-webgpu';
  tts = await KokoroTTS.from_pretrained(
    'onnx-community/Kokoro-82M-v1.0-ONNX',
    {
      device: webgpu ? 'webgpu' : 'wasm',
      dtype: webgpu ? 'fp32' : 'q8',
      progress_callback: (progress: unknown) => {
        postWorkerProgress(id, normalizeDownloadProgress(progress));
      },
    },
  );
  loadedModel = payload.modelId;
  loadedBackend = webgpu ? 'webgpu' : 'wasm';

  // Force the first ONNX invocation during installation, not the first reply.
  await tts.generate('Ready.', { voice: 'am_michael', speed: 1.05 });
  postWorkerResult(id, { modelId: payload.modelId, cached: false });
}

async function synthesize(id: number, payload: SynthesizePayload): Promise<void> {
  if (!tts) throw new Error('Kokoro is not installed.');
  const text = normalizeSpokenText(payload.text);
  if (!text) throw new Error('Kokoro received an empty clause.');
  const splitter = new TextSplitterStream();
  splitter.push(text);
  splitter.close();
  const startedAt = performance.now();
  const chunks: Array<{ text: string; phonemes: string; audio: Float32Array; sampleRate: number }> = [];
  const transfers: Transferable[] = [];
  for await (const chunk of tts.stream(splitter, {
    voice: payload.voice as 'am_michael',
    speed: payload.speed,
  })) {
    const audio = new Float32Array(chunk.audio.audio);
    chunks.push({
      text: chunk.text,
      phonemes: chunk.phonemes,
      audio,
      sampleRate: chunk.audio.sampling_rate,
    });
    transfers.push(audio.buffer);
  }
  const elapsedMs = performance.now() - startedAt;
  const audioDurationSeconds = chunks.reduce(
    (total, chunk) => total + chunk.audio.length / chunk.sampleRate,
    0,
  );
  postWorkerResult(id, {
    engine: 'kokoro',
    backend: loadedBackend,
    preset: 'standard',
    chunks,
    elapsedMs,
    audioDurationSeconds,
    realTimeFactor: elapsedMs / Math.max(1, audioDurationSeconds * 1000),
  }, transfers);
}

async function handle(message: RpcWorkerRequest): Promise<void> {
  if (message.operation === 'load') {
    await loadModel(message.id, message.payload as LoadPayload);
  } else if (message.operation === 'synthesize') {
    await synthesize(message.id, message.payload as SynthesizePayload);
  } else if (message.operation === 'dispose') {
    await tts?.model.dispose();
    tts = null;
    loadedModel = null;
    postWorkerResult(message.id, undefined);
  } else {
    throw new Error(`Unknown Kokoro worker operation: ${message.operation}`);
  }
}

globalThis.addEventListener('message', (event: MessageEvent<RpcWorkerRequest>) => {
  queue = queue.then(() => handle(event.data)).catch((error: unknown) => {
    postWorkerError(event.data.id, error);
  });
});
