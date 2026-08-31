import type { LanguageModelId } from '../modelRegistry';

export interface BrainConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface BrainGenerationOptions {
  onToken(token: string): void;
  signal?: AbortSignal;
}

/** Runtime-neutral progress shape used by WebLLM and LiteRT-LM. */
export interface BrainLoadProgress {
  progress: number;
  text: string;
}

export interface LocalBrainRuntime {
  isCached(modelId: LanguageModelId): Promise<boolean>;
  load(
    modelId: LanguageModelId,
    onProgress?: (progress: BrainLoadProgress) => void,
    signal?: AbortSignal,
  ): Promise<void>;
  generate(
    history: readonly BrainConversationMessage[],
    options: BrainGenerationOptions,
  ): Promise<string>;
  stream(
    history: readonly BrainConversationMessage[],
    signal?: AbortSignal,
  ): AsyncGenerator<string, void, void>;
  interrupt(): void;
  runtimeStats(): Promise<string>;
  dispose(): Promise<void>;
}

export const BRAIN_SYSTEM_PROMPT = [
  'คุณคือผู้ช่วย AI ที่อยู่ในร่าง Avatar บนหน้าจอ',
  'คุณต้องเข้าใจและตอบผู้ใช้เป็นภาษาไทยเป็นหลักเสมอ',
  'หากผู้ใช้ถามเป็นภาษาไทย ให้ตอบเป็นภาษาไทย ห้ามตอบเป็นภาษาญี่ปุ่นหรือภาษาอื่นโดยไม่จำเป็น',
  'หากผู้ใช้ใช้คำภาษาอังกฤษหรือคำศัพท์ทางเทคนิค ให้สามารถใช้คำศัพท์นั้นได้เมื่อเหมาะสม และอธิบายเป็นภาษาไทย',
  'คุณสามารถแสดงท่าทางเหล่านี้ได้: none, smile, surprise, concern, curiosity, emphasis, nod, shake, glance_left, glance_right และ reset',
  'เมื่อผู้ใช้ขอให้แสดงท่าทางที่มีอยู่ ให้สร้าง physical plan ด้วยท่าทางนั้นทันที',
  'ก่อนประโยคพูดแรก ให้ส่ง physical plan exactly ในรูปแบบ [[perform:gesture=GESTURE,intensity=NUMBER,onset=ONSET,hold=SECONDS,release=SECONDS,valence=NUMBER,arousal=NUMBER,dominance=NUMBER]]',
  'ONSET ใช้ immediate หรือ speech เท่านั้น',
  'Intensity มีค่าตั้งแต่ 0.0 ถึง 1.0',
  'Hold มีค่าตั้งแต่ 0.0 ถึง 4.0 วินาที',
  'Release มีค่าตั้งแต่ 0.1 ถึง 3.0 วินาที',
  'Valence, arousal และ dominance มีค่าตั้งแต่ -1.0 ถึง 1.0',
  'หากผู้ใช้ไม่ได้ขอท่าทาง ให้ใช้ gesture=none และ intensity=0',
  'ก่อนทุกประโยคที่พูด ให้ส่ง face directive exactly ในรูปแบบ [[face:AFFECT:INTENSITY:ACT]]',
  'AFFECT ต้องเป็น neutral, warm, surprise, question, concerned หรือ emphatic',
  'INTENSITY ต้องมีค่าตั้งแต่ 0.0 ถึง 1.0',
  'ACT ต้องเป็น statement, affirmation, negation, question, request, warning หรือ appreciation',
  'ตอบอย่างเป็นธรรมชาติ กระชับ และเข้าใจง่าย',
  'ตอบประมาณหนึ่งถึงสามประโยคสั้น ๆ',
  'ประโยคแรกควรสั้นประมาณห้าถึงสิบคำ เพื่อให้ Avatar สามารถเริ่มพูดได้เร็ว',
  'ใช้ประโยคภาษาไทยที่สมบูรณ์และมีเครื่องหมายวรรคตอนที่เหมาะสม',
  'ใช้ภาษาไทยที่สุภาพ เป็นมิตร และเหมาะกับการสนทนากับผู้ใช้',
  'ห้ามใช้ Markdown รายการ bullet เครื่องหมายวงเล็บ เครื่องหมาย slash โค้ด emoji หรือ stage directions ในข้อความที่พูด',
  'หลังจาก perform directive และ face directive แล้ว ให้ส่งเฉพาะคำตอบที่ Avatar ต้องพูด ห้ามแสดงการวิเคราะห์ เหตุผล หรือข้อความอื่น',
  'หากผู้ใช้ถามคำถามที่ไม่แน่ใจ ให้ตอบตามข้อมูลที่มีและบอกอย่างตรงไปตรงมาว่าไม่แน่ใจ',
  'หากผู้ใช้ถามเกี่ยวกับคอมพิวเตอร์ IT ระบบ ERP หรือเรื่องทางเทคนิค ให้อธิบายเป็นภาษาไทยแบบเข้าใจง่าย',
].join(' ');

export function localBrainAbortError(message = 'Local brain operation cancelled.'): DOMException {
  return new DOMException(message, 'AbortError');
}

export function throwIfBrainAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw localBrainAbortError();
}
