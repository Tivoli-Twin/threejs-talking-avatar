export type LocalModelKind = 'stt' | 'llm' | 'tts';

export type LocalExecutionBackend = 'wasm' | 'webgpu';

export type LocalModelRuntime =
  | 'onnx-runtime-web'
  | 'transformers-js'
  | 'litert-lm'
  | 'webllm';

export type LocalModelPrecision = 'q8' | 'q4' | 'q4f16' | 'fp32';

export type LocalModelCapability = 'wasm' | 'webgpu' | 'shader-f16';

export type LocalModelReleaseChannel =
  | 'safe-default'
  | 'optional'
  | 'experimental';

export type DownloadSizeConfidence = 'artifact-listing' | 'estimate';

export interface LocalModelLicense {
  /** Null when the license is not represented by an SPDX identifier. */
  spdxId: 'Apache-2.0' | 'MIT' | null;
  name: string;
  url: string;
  requiresAcceptance: boolean;
}

export interface LocalModelArtifact {
  /** Stable upstream repository identifier. It is metadata, not a fetch URL. */
  repository: string;
  /** Runtime-specific artifact/config identifier planned for the downloader. */
  artifactId: string;
  estimatedDownloadBytes: number;
  downloadSizeConfidence: DownloadSizeConfidence;
  /** Must be pinned before a production downloader is enabled. */
  revision: string | null;
}

export interface LocalModelBackendProfile {
  runtime: LocalModelRuntime;
  execution: LocalExecutionBackend;
  precision: LocalModelPrecision;
  requiredCapabilities: readonly LocalModelCapability[];
}

export interface LocalModelPolicy {
  releaseChannel: LocalModelReleaseChannel;
  allowAutomaticDownload: boolean;
  requiresExplicitConsent: boolean;
  warning?: string;
}

interface LocalModelDescriptorBase {
  id: LocalModelId;
  kind: LocalModelKind;
  displayName: string;
  description: string;
  artifact: LocalModelArtifact;
  backends: readonly LocalModelBackendProfile[];
  license: LocalModelLicense;
  policy: LocalModelPolicy;
}

export interface SpeechToTextModelDescriptor extends LocalModelDescriptorBase {
  kind: 'stt';
  id: SpeechToTextModelId;
  languageSupport: 'english' | 'multilingual';
  streaming: boolean;
}

export interface LanguageModelDescriptor extends LocalModelDescriptorBase {
  kind: 'llm';
  id: LanguageModelId;
  contextWindowTokens: number;
  multimodal: boolean;
}

export interface TextToSpeechModelDescriptor extends LocalModelDescriptorBase {
  kind: 'tts';
  id: TextToSpeechModelId;
  voices: 'kokoro' | 'supertonic' | 'kitten';
  producesNativeVisemeTiming: boolean;
}

export type LocalModelDescriptor =
  | SpeechToTextModelDescriptor
  | LanguageModelDescriptor
  | TextToSpeechModelDescriptor;

export const SPEECH_TO_TEXT_MODEL_IDS = [
  'moonshine-tiny-q8',
  'whisper-tiny-en-q8',
] as const;

export const LANGUAGE_MODEL_IDS = [
  'qwen2.5-0.5b-instruct-q4f16',
  'qwen3.5-0.8b-q4f16',
  'qwen3.5-2b-q4f16',
  'qwen3.5-4b-q4f16',
  'gemma-4-e2b-it-litert-web',
  'gemma-4-e4b-it-litert-web',
  'gemma-3-1b-it-q4f16',
] as const;

export const TEXT_TO_SPEECH_MODEL_IDS = [
  'kokoro-82m-q8-wasm',
  'kokoro-82m-fp32-webgpu',
  'kokoro-82m-timestamped-q8-wasm',
  'kokoro-82m-timestamped-fp32-webgpu',
  'supertonic-2-instant-webgpu',
  'supertonic-2-quality-webgpu',
  'kitten-tts-nano-0.8-fp32-webgpu',
] as const;

export type SpeechToTextModelId = (typeof SPEECH_TO_TEXT_MODEL_IDS)[number];
export type LanguageModelId = (typeof LANGUAGE_MODEL_IDS)[number];
export type TextToSpeechModelId = (typeof TEXT_TO_SPEECH_MODEL_IDS)[number];
export type LocalModelId =
  | SpeechToTextModelId
  | LanguageModelId
  | TextToSpeechModelId;

// Upstream artifact listings report decimal MB/GB; preserve that convention.
const MB = 1_000_000;

const MIT_LICENSE: LocalModelLicense = {
  spdxId: 'MIT',
  name: 'MIT License',
  url: 'https://opensource.org/license/mit',
  requiresAcceptance: false,
};

const APACHE_2_LICENSE: LocalModelLicense = {
  spdxId: 'Apache-2.0',
  name: 'Apache License 2.0',
  url: 'https://www.apache.org/licenses/LICENSE-2.0',
  requiresAcceptance: false,
};

const GEMMA_TERMS: LocalModelLicense = {
  spdxId: null,
  name: 'Gemma Terms of Use',
  url: 'https://ai.google.dev/gemma/terms',
  requiresAcceptance: true,
};

const OPENRAIL_M_LICENSE: LocalModelLicense = {
  spdxId: null,
  name: 'OpenRAIL-M License',
  url: 'https://huggingface.co/Supertone/supertonic-2/blob/75e6727618a02f323c720cba9478152d4bc16ca4/LICENSE',
  requiresAcceptance: false,
};

const WASM_Q8_ONNX = [
  {
    runtime: 'onnx-runtime-web',
    execution: 'wasm',
    precision: 'q8',
    requiredCapabilities: ['wasm'],
  },
] as const satisfies readonly LocalModelBackendProfile[];

const WEBGPU_Q4F16_WEBLLM = [
  {
    runtime: 'webllm',
    execution: 'webgpu',
    precision: 'q4f16',
    requiredCapabilities: ['webgpu', 'shader-f16'],
  },
] as const satisfies readonly LocalModelBackendProfile[];

const WEBGPU_Q4F16_TRANSFORMERS = [
  {
    runtime: 'transformers-js',
    execution: 'webgpu',
    precision: 'q4f16',
    requiredCapabilities: ['webgpu', 'shader-f16'],
  },
] as const satisfies readonly LocalModelBackendProfile[];

const WEBGPU_Q4_LITERT_LM = [
  {
    runtime: 'litert-lm',
    execution: 'webgpu',
    precision: 'q4',
    requiredCapabilities: ['webgpu'],
  },
] as const satisfies readonly LocalModelBackendProfile[];

export const SPEECH_TO_TEXT_MODELS = {
  'moonshine-tiny-q8': {
    id: 'moonshine-tiny-q8',
    kind: 'stt',
    displayName: 'Moonshine Tiny q8',
    description: 'Low-latency English speech recognition for the safe local path.',
    artifact: {
      repository: 'onnx-community/moonshine-tiny-ONNX',
      artifactId: 'moonshine-tiny-q8-onnx',
      estimatedDownloadBytes: 34 * MB,
      downloadSizeConfidence: 'estimate',
      revision: null,
    },
    backends: WASM_Q8_ONNX,
    license: MIT_LICENSE,
    policy: {
      releaseChannel: 'safe-default',
      allowAutomaticDownload: true,
      requiresExplicitConsent: false,
    },
    languageSupport: 'english',
    streaming: false,
  },
  'whisper-tiny-en-q8': {
    id: 'whisper-tiny-en-q8',
    kind: 'stt',
    displayName: 'Whisper Tiny.en q8',
    description: 'English recognition fallback with broader ecosystem support.',
    artifact: {
      repository: 'onnx-community/whisper-tiny.en',
      artifactId: 'whisper-tiny-en-q8-onnx',
      estimatedDownloadBytes: 42 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: null,
    },
    backends: WASM_Q8_ONNX,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    languageSupport: 'english',
    streaming: false,
  },
} as const satisfies Record<SpeechToTextModelId, SpeechToTextModelDescriptor>;

export const LANGUAGE_MODELS = {
  'qwen2.5-0.5b-instruct-q4f16': {
    id: 'qwen2.5-0.5b-instruct-q4f16',
    kind: 'llm',
    displayName: 'Qwen2.5 0.5B Instruct',
    description: 'Fast, compact WebLLM model selected for the safe local path.',
    artifact: {
      repository: 'mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
      artifactId: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
      estimatedDownloadBytes: 290 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: null,
    },
    backends: WEBGPU_Q4F16_WEBLLM,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'safe-default',
      allowAutomaticDownload: true,
      requiresExplicitConsent: false,
    },
    contextWindowTokens: 32_768,
    multimodal: false,
  },
  'qwen3.5-0.8b-q4f16': {
    id: 'qwen3.5-0.8b-q4f16',
    kind: 'llm',
    displayName: 'Qwen3.5 0.8B',
    description: 'Newer WebLLM quality option; never selected or downloaded implicitly.',
    artifact: {
      repository: 'mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC',
      artifactId: 'Qwen3.5-0.8B-q4f16_1-MLC',
      estimatedDownloadBytes: 447 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: null,
    },
    backends: WEBGPU_Q4F16_WEBLLM,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    contextWindowTokens: 262_144,
    multimodal: false,
  },
  'qwen3.5-2b-q4f16': {
    id: 'qwen3.5-2b-q4f16',
    kind: 'llm',
    displayName: 'Qwen3.5 2B',
    description: 'Balanced WebLLM option with a substantial quality gain over sub-billion models.',
    artifact: {
      repository: 'mlc-ai/Qwen3.5-2B-q4f16_1-MLC',
      artifactId: 'Qwen3.5-2B-q4f16_1-MLC',
      estimatedDownloadBytes: 1_080 * MB,
      downloadSizeConfidence: 'estimate',
      revision: null,
    },
    backends: WEBGPU_Q4F16_WEBLLM,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    contextWindowTokens: 262_144,
    multimodal: false,
  },
  'qwen3.5-4b-q4f16': {
    id: 'qwen3.5-4b-q4f16',
    kind: 'llm',
    displayName: 'Qwen3.5 4B',
    description: 'Higher-quality WebLLM option for devices with roughly 4 GB of free GPU memory.',
    artifact: {
      repository: 'mlc-ai/Qwen3.5-4B-q4f16_1-MLC',
      artifactId: 'Qwen3.5-4B-q4f16_1-MLC',
      estimatedDownloadBytes: 1_590 * MB,
      downloadSizeConfidence: 'estimate',
      revision: null,
    },
    backends: WEBGPU_Q4F16_WEBLLM,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
      warning:
        'Requires approximately 3.9 GB of WebGPU memory and may compete with rendering and speech synthesis.',
    },
    contextWindowTokens: 262_144,
    multimodal: false,
  },
  'gemma-4-e2b-it-litert-web': {
    id: 'gemma-4-e2b-it-litert-web',
    kind: 'llm',
    displayName: 'Gemma 4 E2B IT',
    description: 'Recommended smart local model through Google LiteRT-LM WebGPU.',
    artifact: {
      repository: 'litert-community/gemma-4-E2B-it-litert-lm',
      artifactId: 'gemma-4-E2B-it-web.litertlm',
      estimatedDownloadBytes: 2_010 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: '9262660a1676eed6d0c477ab1a86344430854664',
    },
    backends: WEBGPU_Q4_LITERT_LM,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
      warning:
        'Two-gigabyte first download. Keep other GPU-heavy models unloaded while it initializes.',
    },
    contextWindowTokens: 131_072,
    multimodal: false,
  },
  'gemma-4-e4b-it-litert-web': {
    id: 'gemma-4-e4b-it-litert-web',
    kind: 'llm',
    displayName: 'Gemma 4 E4B IT',
    description: 'Highest-quality Gemma browser option, with materially higher memory and latency.',
    artifact: {
      repository: 'litert-community/gemma-4-E4B-it-litert-lm',
      artifactId: 'gemma-4-E4B-it-web.litertlm',
      estimatedDownloadBytes: 2_970 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: 'f7ad3343bd6ebc9607f4dc3bc4f2398bd5749bc5',
    },
    backends: WEBGPU_Q4_LITERT_LM,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'experimental',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
      warning:
        'Three-gigabyte first download with high GPU-memory pressure; may disrupt rendering or lose the WebGPU device.',
    },
    contextWindowTokens: 131_072,
    multimodal: false,
  },
  'gemma-3-1b-it-q4f16': {
    id: 'gemma-3-1b-it-q4f16',
    kind: 'llm',
    displayName: 'Gemma 3 1B IT',
    description: 'Optional compact Gemma comparison model.',
    artifact: {
      repository: 'onnx-community/gemma-3-1b-it-ONNX',
      artifactId: 'gemma-3-1b-it-q4f16-onnx',
      estimatedDownloadBytes: 783 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: null,
    },
    backends: WEBGPU_Q4F16_TRANSFORMERS,
    license: GEMMA_TERMS,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: true,
    },
    contextWindowTokens: 32_768,
    multimodal: false,
  },
} as const satisfies Record<LanguageModelId, LanguageModelDescriptor>;

export const TEXT_TO_SPEECH_MODELS = {
  'kokoro-82m-q8-wasm': {
    id: 'kokoro-82m-q8-wasm',
    kind: 'tts',
    displayName: 'Kokoro 82M q8 (WASM)',
    description: 'CPU/WASM voice synthesis that leaves the GPU available to the face and LLM.',
    artifact: {
      repository: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      artifactId: 'kokoro-82m-q8-onnx',
      estimatedDownloadBytes: 92 * MB,
      downloadSizeConfidence: 'estimate',
      revision: null,
    },
    backends: WASM_Q8_ONNX,
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'safe-default',
      allowAutomaticDownload: true,
      requiresExplicitConsent: false,
    },
    voices: 'kokoro',
    producesNativeVisemeTiming: false,
  },
  'kokoro-82m-fp32-webgpu': {
    id: 'kokoro-82m-fp32-webgpu',
    kind: 'tts',
    displayName: 'Kokoro 82M fp32 (WebGPU)',
    description: 'Opt-in synthesis path for devices with spare GPU memory.',
    artifact: {
      repository: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      artifactId: 'kokoro-82m-fp32-onnx',
      estimatedDownloadBytes: 326 * MB,
      downloadSizeConfidence: 'estimate',
      revision: null,
    },
    backends: [
      {
        runtime: 'onnx-runtime-web',
        execution: 'webgpu',
        precision: 'fp32',
        requiredCapabilities: ['webgpu'],
      },
    ],
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    voices: 'kokoro',
    producesNativeVisemeTiming: false,
  },
  'kokoro-82m-timestamped-q8-wasm': {
    id: 'kokoro-82m-timestamped-q8-wasm',
    kind: 'tts',
    displayName: 'Kokoro Timestamped q8 (WASM)',
    description: 'CPU synthesis with the per-token durations used to construct its own waveform.',
    artifact: {
      repository: 'onnx-community/Kokoro-82M-v1.0-ONNX-timestamped',
      artifactId: 'kokoro-82m-timestamped-q8-onnx',
      estimatedDownloadBytes: 92.4 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: null,
    },
    backends: [
      {
        runtime: 'transformers-js',
        execution: 'wasm',
        precision: 'q8',
        requiredCapabilities: ['wasm'],
      },
    ],
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    voices: 'kokoro',
    producesNativeVisemeTiming: true,
  },
  'kokoro-82m-timestamped-fp32-webgpu': {
    id: 'kokoro-82m-timestamped-fp32-webgpu',
    kind: 'tts',
    displayName: 'Kokoro Timestamped fp32',
    description: 'Recommended lip-sync path: WebGPU Kokoro with synthesis-native phone durations.',
    artifact: {
      repository: 'onnx-community/Kokoro-82M-v1.0-ONNX-timestamped',
      artifactId: 'kokoro-82m-timestamped-fp32-onnx',
      estimatedDownloadBytes: 326 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: null,
    },
    backends: [
      {
        runtime: 'transformers-js',
        execution: 'webgpu',
        precision: 'fp32',
        requiredCapabilities: ['webgpu'],
      },
    ],
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
      warning: 'Uses WebGPU alongside the face and local brain; unload larger models if the adapter runs out of memory.',
    },
    voices: 'kokoro',
    producesNativeVisemeTiming: true,
  },
  'supertonic-2-instant-webgpu': {
    id: 'supertonic-2-instant-webgpu',
    kind: 'tts',
    displayName: 'Supertonic 2 Instant',
    description: 'Two-step low-latency synthesis with WebGPU and automatic WASM fallback.',
    artifact: {
      repository: 'Supertone/supertonic-2',
      artifactId: 'supertonic-2-onnx',
      estimatedDownloadBytes: 263 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: '75e6727618a02f323c720cba9478152d4bc16ca4',
    },
    backends: [
      {
        runtime: 'onnx-runtime-web',
        execution: 'webgpu',
        precision: 'fp32',
        requiredCapabilities: ['webgpu'],
      },
      {
        runtime: 'onnx-runtime-web',
        execution: 'wasm',
        precision: 'fp32',
        requiredCapabilities: ['wasm'],
      },
    ],
    license: OPENRAIL_M_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    voices: 'supertonic',
    producesNativeVisemeTiming: false,
  },
  'supertonic-2-quality-webgpu': {
    id: 'supertonic-2-quality-webgpu',
    kind: 'tts',
    displayName: 'Supertonic 2 Quality',
    description: 'Five-step higher-quality synthesis with WebGPU and automatic WASM fallback.',
    artifact: {
      repository: 'Supertone/supertonic-2',
      artifactId: 'supertonic-2-onnx',
      estimatedDownloadBytes: 263 * MB,
      downloadSizeConfidence: 'artifact-listing',
      revision: '75e6727618a02f323c720cba9478152d4bc16ca4',
    },
    backends: [
      {
        runtime: 'onnx-runtime-web',
        execution: 'webgpu',
        precision: 'fp32',
        requiredCapabilities: ['webgpu'],
      },
      {
        runtime: 'onnx-runtime-web',
        execution: 'wasm',
        precision: 'fp32',
        requiredCapabilities: ['wasm'],
      },
    ],
    license: OPENRAIL_M_LICENSE,
    policy: {
      releaseChannel: 'optional',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
    },
    voices: 'supertonic',
    producesNativeVisemeTiming: false,
  },
  'kitten-tts-nano-0.8-fp32-webgpu': {
    id: 'kitten-tts-nano-0.8-fp32-webgpu',
    kind: 'tts',
    displayName: 'KittenTTS Nano 0.8',
    description: 'Tiny single-pass 15M-parameter voice for the fastest low-memory comparison path.',
    artifact: {
      repository: 'KittenML/kitten-tts-nano-0.8-fp32',
      artifactId: 'kitten-tts-nano-v0.8-fp32-onnx',
      estimatedDownloadBytes: 60_045_997,
      downloadSizeConfidence: 'artifact-listing',
      revision: '7a1db645b1f3ab9420761d87428e042b9cec3f26',
    },
    backends: [
      {
        runtime: 'onnx-runtime-web',
        execution: 'webgpu',
        precision: 'fp32',
        requiredCapabilities: ['webgpu'],
      },
      {
        runtime: 'onnx-runtime-web',
        execution: 'wasm',
        precision: 'fp32',
        requiredCapabilities: ['wasm'],
      },
    ],
    license: APACHE_2_LICENSE,
    policy: {
      releaseChannel: 'experimental',
      allowAutomaticDownload: false,
      requiresExplicitConsent: false,
      warning: 'KittenTTS 0.8 is a developer preview; compare its pronunciation and voice quality before choosing it.',
    },
    voices: 'kitten',
    producesNativeVisemeTiming: false,
  },
} as const satisfies Record<TextToSpeechModelId, TextToSpeechModelDescriptor>;

export const LOCAL_MODEL_REGISTRY = {
  ...SPEECH_TO_TEXT_MODELS,
  ...LANGUAGE_MODELS,
  ...TEXT_TO_SPEECH_MODELS,
} as const satisfies Record<LocalModelId, LocalModelDescriptor>;

export const LOCAL_MODELS = Object.freeze(
  Object.values(LOCAL_MODEL_REGISTRY),
) as readonly LocalModelDescriptor[];

export function isLocalModelId(value: string): value is LocalModelId {
  return Object.hasOwn(LOCAL_MODEL_REGISTRY, value);
}

export function getLocalModel(modelId: LocalModelId): LocalModelDescriptor {
  return LOCAL_MODEL_REGISTRY[modelId];
}

export function getLocalModelsByKind<K extends LocalModelKind>(
  kind: K,
): readonly Extract<LocalModelDescriptor, { kind: K }>[] {
  return LOCAL_MODELS.filter(
    (model): model is Extract<LocalModelDescriptor, { kind: K }> =>
      model.kind === kind,
  );
}
