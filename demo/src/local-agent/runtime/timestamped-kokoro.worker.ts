import {
  AutoTokenizer,
  env,
  StyleTextToSpeech2Model,
  Tensor,
} from '@huggingface/transformers';
import {
  normalizeDownloadProgress,
  postWorkerError,
  postWorkerProgress,
  postWorkerResult,
  type RpcWorkerRequest,
} from './workerProtocol';
import { phonemizeForTimestampedKokoro } from './TimestampedKokoroPhonemizer';
import {
  KOKORO_SAMPLE_RATE,
  kokoroNativeDurationsToIntervals,
} from './TimestampedKokoroTiming';
import type { TimestampedKokoroModelId } from './TimestampedKokoroRuntime';
import { ONNX_WASM_BASE_URL } from './RuntimeAssetUrls';

interface LoadPayload {
  modelId: TimestampedKokoroModelId;
  warmupVoice?: string;
}

interface SynthesizePayload {
  text: string;
  voice: string;
  speed: number;
}

interface CallableModel {
  (inputs: Record<string, Tensor>): Promise<Record<string, unknown>>;
  dispose(): Promise<unknown>;
}

interface CallableTokenizer {
  (text: string, options?: Record<string, unknown>): { input_ids: Tensor };
}

interface NumericTensorLike {
  data: ArrayLike<number | bigint>;
}

const MODEL_REPOSITORY = 'onnx-community/Kokoro-82M-v1.0-ONNX-timestamped';
const VOICE_REPOSITORY =
  'https://hf-proxy.doublethew.workers.dev/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices';
const STYLE_DIMENSION = 256;
const MAX_CONTENT_TOKENS = 509;
const DEFAULT_VOICE = 'am_michael';

if (env.backends.onnx.wasm) {
  env.backends.onnx.wasm.wasmPaths = ONNX_WASM_BASE_URL;
}

env.remoteHost = 'https://hf-proxy.doublethew.workers.dev';
env.remotePathTemplate = '{model}/resolve/{revision}/{file}';

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let model: CallableModel | null = null;
let tokenizer: CallableTokenizer | null = null;
let loadedModel: TimestampedKokoroModelId | null = null;
let loadedBackend: 'webgpu' | 'wasm' = 'wasm';
let queue = Promise.resolve();
const voiceCache = new Map<string, Float32Array>();

function modelOptions(modelId: TimestampedKokoroModelId): {
  device: 'webgpu' | 'wasm';
  dtype: 'fp32' | 'q8';
} {
  return modelId === 'kokoro-82m-timestamped-fp32-webgpu'
    ? { device: 'webgpu', dtype: 'fp32' }
    : { device: 'wasm', dtype: 'q8' };
}

function numericData(value: unknown, label: string): number[] {
  if (!value || typeof value !== 'object') {
    throw new Error(`Timestamped Kokoro did not return ${label}.`);
  }
  const data = Reflect.get(value, 'data') as NumericTensorLike['data'] | undefined;
  if (!data || typeof data.length !== 'number') {
    throw new Error(`Timestamped Kokoro returned invalid ${label}.`);
  }
  return Array.from(data, (entry) => Number(entry));
}

function pcmData(value: unknown): Float32Array {
  if (!value || typeof value !== 'object') {
    throw new Error('Timestamped Kokoro did not return a waveform.');
  }
  const data = Reflect.get(value, 'data') as NumericTensorLike['data'] | undefined;
  if (!data || typeof data.length !== 'number') {
    throw new Error('Timestamped Kokoro returned an invalid waveform.');
  }
  // Copy once into a dedicated transferable buffer without first allocating a
  // same-sized JavaScript number array.
  return data instanceof Float32Array
    ? data.slice()
    : Float32Array.from(data, (entry) => Number(entry));
}

async function loadVoice(name: string): Promise<Float32Array> {
  const cached = voiceCache.get(name);
  if (cached) return cached;
  if (!/^[a-z]{2}_[a-z]+$/u.test(name)) {
    throw new Error(`Invalid Kokoro voice name "${name}".`);
  }
  const response = await fetch(`${VOICE_REPOSITORY}/${encodeURIComponent(name)}.bin`, {
    cache: 'force-cache',
  });
  if (!response.ok) {
    throw new Error(`Could not download Kokoro voice "${name}" (${response.status}).`);
  }
  const voice = new Float32Array(await response.arrayBuffer());
  voiceCache.set(name, voice);
  return voice;
}

function voiceStyle(voice: Float32Array, contentTokenCount: number): Tensor {
  const offset = contentTokenCount * STYLE_DIMENSION;
  const end = offset + STYLE_DIMENSION;
  if (contentTokenCount < 1 || contentTokenCount > MAX_CONTENT_TOKENS || end > voice.length) {
    throw new Error(
      `Kokoro voice embedding is unavailable for ${contentTokenCount} content tokens.`,
    );
  }
  return new Tensor('float32', voice.slice(offset, end), [1, STYLE_DIMENSION]);
}

async function infer(text: string, voiceName: string, speed: number): Promise<{
  text: string;
  audio: Float32Array;
  phonemes: ReturnType<typeof kokoroNativeDurationsToIntervals>;
}> {
  if (!model || !tokenizer) throw new Error('Timestamped Kokoro is not installed.');
  const clean = text.normalize('NFKC').replace(/\s+/gu, ' ').trim();
  if (!clean) throw new Error('Timestamped Kokoro received an empty clause.');
  if (!Number.isFinite(speed) || speed < 0.5 || speed > 2) {
    throw new RangeError('Timestamped Kokoro speed must be between 0.5 and 2.');
  }

  const ipa = await phonemizeForTimestampedKokoro(clean);
  const contentSymbols = Array.from(ipa.normalize('NFC'));
  if (contentSymbols.length > MAX_CONTENT_TOKENS) {
    throw new Error(
      `Timestamped Kokoro clauses are limited to ${MAX_CONTENT_TOKENS} phoneme symbols.`,
    );
  }
  const { input_ids } = tokenizer(ipa, { truncation: true });
  const contentTokenCount = input_ids.size - 2;
  if (contentTokenCount !== contentSymbols.length) {
    throw new Error(
      `Kokoro tokenizer produced ${contentTokenCount} content tokens for ` +
      `${contentSymbols.length} phoneme symbols.`,
    );
  }
  const voice = await loadVoice(voiceName);
  const output = await model({
    input_ids,
    style: voiceStyle(voice, contentTokenCount),
    speed: new Tensor('float32', [speed], [1]),
  });
  const audio = pcmData(output.waveform);
  if (audio.length === 0) throw new Error('Timestamped Kokoro returned empty audio.');
  const audioDurationSeconds = audio.length / KOKORO_SAMPLE_RATE;
  // The timestamped ONNX export adds `pred_dur`; keep `durations` as a
  // compatibility fallback for future Transformers.js output normalization.
  const modelDurationsFrames = numericData(
    output.pred_dur ?? output.durations,
    'token durations',
  );
  const intervals = kokoroNativeDurationsToIntervals({
    text: clean,
    phonemes: ipa,
    modelDurationsFrames,
    audioDurationSeconds,
  });
  return { text: clean, audio, phonemes: intervals };
}

async function loadModel(id: number, payload: LoadPayload): Promise<void> {
  if (model && tokenizer && loadedModel === payload.modelId) {
    postWorkerResult(id, { modelId: payload.modelId, cached: true });
    return;
  }
  await model?.dispose();
  model = null;
  tokenizer = null;
  loadedModel = null;
  const options = modelOptions(payload.modelId);
  const progress = (value: unknown): void => {
    postWorkerProgress(id, normalizeDownloadProgress(value));
  };
  const [nextModel, nextTokenizer] = await Promise.all([
    StyleTextToSpeech2Model.from_pretrained(MODEL_REPOSITORY, {
      device: options.device,
      dtype: options.dtype,
      progress_callback: progress,
    }),
    AutoTokenizer.from_pretrained(MODEL_REPOSITORY, {
      progress_callback: progress,
    }),
  ]);
  model = nextModel as unknown as CallableModel;
  tokenizer = nextTokenizer as unknown as CallableTokenizer;
  loadedBackend = options.device;

  // Compile ONNX kernels and cache the selected voice during installation.
  await infer('Ready.', payload.warmupVoice ?? DEFAULT_VOICE, 1.05);
  loadedModel = payload.modelId;
  postWorkerResult(id, { modelId: payload.modelId, cached: false });
}

function splitClause(text: string): [string, string] {
  const clean = text
    .normalize('NFKC')
    .replace(/\s+/gu, ' ')
    .trim();

  if (!clean) {
    return ['', ''];
  }

  // พยายามตัดตรงกลางประโยค โดยให้ความสำคัญกับ
  // punctuation / space ก่อนการตัดกลางคำ
  const middle = Math.floor(clean.length / 2);

  const breakCandidates = [
    clean.lastIndexOf(' ', middle),
    clean.lastIndexOf('。', middle),
    clean.lastIndexOf('.', middle),
    clean.lastIndexOf('!', middle),
    clean.lastIndexOf('?', middle),
    clean.lastIndexOf('…', middle),
    clean.lastIndexOf('ฯ', middle),
    clean.lastIndexOf(',', middle),
    clean.lastIndexOf('，', middle),
  ];

  const before = Math.max(...breakCandidates);

  if (before > 0) {
    return [
      clean.slice(0, before + 1).trim(),
      clean.slice(before + 1).trim(),
    ];
  }

  // ภาษาไทยอาจไม่มี space ระหว่างคำ
  // จึง fallback มาตัดตรงกลางข้อความ
  return [
    clean.slice(0, middle).trim(),
    clean.slice(middle).trim(),
  ];
}

async function inferWithChunking(
  text: string,
  voice: string,
  speed: number,
): Promise<Array<Awaited<ReturnType<typeof infer>>>> {
  try {
    return [await infer(text, voice, speed)];
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : String(error);

    // ถ้าไม่ใช่ 509 phoneme error ให้ส่ง error เดิมออกไป
    if (!message.includes('phoneme symbols')) {
      throw error;
    }

    const [left, right] = splitClause(text);

    if (!left || !right || left === text || right === text) {
      throw error;
    }

    const [leftChunks, rightChunks] = await Promise.all([
      inferWithChunking(left, voice, speed),
      inferWithChunking(right, voice, speed),
    ]);

    return [...leftChunks, ...rightChunks];
  }
}

async function synthesize(
  id: number,
  payload: SynthesizePayload,
): Promise<void> {
  const startedAt = performance.now();

  const chunks = await inferWithChunking(
    payload.text,
    payload.voice,
    payload.speed,
  );

  const elapsedMs = performance.now() - startedAt;

  const audioDurationSeconds = chunks.reduce(
    (total, chunk) =>
      total + chunk.audio.length / KOKORO_SAMPLE_RATE,
    0,
  );

  const transfers: Transferable[] = chunks.map(
    (chunk) => chunk.audio.buffer,
  );

  postWorkerResult(
    id,
    {
      engine: 'kokoro-timestamped',
      backend: loadedBackend,
      preset: 'standard',
      chunks: chunks.map((chunk) => ({
        text: chunk.text,
        phonemes: chunk.phonemes,
        audio: chunk.audio,
        sampleRate: KOKORO_SAMPLE_RATE,
      })),
      elapsedMs,
      audioDurationSeconds,
      realTimeFactor:
        elapsedMs /
        Math.max(1, audioDurationSeconds * 1000),
    },
    transfers,
  );
}

async function handle(message: RpcWorkerRequest): Promise<void> {
  if (message.operation === 'load') {
    await loadModel(message.id, message.payload as LoadPayload);
  } else if (message.operation === 'synthesize') {
    await synthesize(message.id, message.payload as SynthesizePayload);
  } else if (message.operation === 'dispose') {
    await model?.dispose();
    model = null;
    tokenizer = null;
    loadedModel = null;
    postWorkerResult(message.id, undefined);
  } else {
    throw new Error(`Unknown timestamped Kokoro worker operation: ${message.operation}`);
  }
}

globalThis.addEventListener('message', (event: MessageEvent<RpcWorkerRequest>) => {
  queue = queue.then(() => handle(event.data)).catch((error: unknown) => {
    postWorkerError(event.data.id, error);
  });
});
