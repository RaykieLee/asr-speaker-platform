export function percentile(values, p) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[index];
}

export function resampleTo16k(input, inputRate) {
  if (!inputRate || inputRate < 16000) {
    if (inputRate === 16000) return new Float32Array(input);
    throw new Error('浏览器音频采样率必须不低于 16000 Hz');
  }
  if (inputRate === 16000) return new Float32Array(input);
  const ratio = inputRate / 16000;
  const output = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < output.length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    output[i] = sum / Math.max(1, end - start);
  }
  return output;
}

export function toS16LE(samples) {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
  }
  return pcm.buffer;
}

export function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

export function hotwordId(item) {
  return item?.set_id || item?.id || null;
}

export function hotwordEntriesToText(entries = []) {
  return entries.map(entry => typeof entry === 'string' ? entry : entry?.word || entry?.phrase || '').filter(Boolean).join('\n');
}

export function mergeHotwordEntries(text, previous = []) {
  const prior = new Map(previous.map(entry => {
    const word = typeof entry === 'string' ? entry : entry?.word || entry?.phrase;
    return [String(word || '').trim().toLocaleLowerCase(), entry];
  }));
  return text.split(/\r?\n/).map(word => word.trim()).filter(Boolean).map(word => {
    const old = prior.get(word.toLocaleLowerCase());
    if (!old || typeof old === 'string') return word;
    return { ...old, word };
  });
}

export function normalizeJobForDisplay(job = {}) {
  const segments = job.effective_segments || job.segments || [];
  return {
    segments: segments.map(segment => ({
      ...segment,
      start: segment.start_seconds ?? segment.start ?? 0,
      end: segment.end_seconds ?? segment.end ?? 0,
      speaker_name: segment.speaker?.display_name || segment.speaker_name ||
        segment.original_prediction?.speaker_name || segment.metadata?.speaker_name ||
        segment.registered_speaker_id || segment.local_speaker_id || null,
    })),
    duration: job.result_metadata?.duration ?? job.result_metadata?.duration_seconds ?? job.duration ?? null,
    speakerDiarization: job.result_metadata?.speaker_diarization ?? job.speaker_diarization ?? '服务未提供状态',
    text: job.result_metadata?.text || job.text || '',
  };
}

export function groupCorrectionTargets(segments = []) {
  const groups = new Map();
  for (const segment of segments) {
    const localId = segment.local_speaker_id;
    if (!localId) continue;
    if (!groups.has(localId)) groups.set(localId, { local_speaker_id: localId, count: 0,
      display_name: segment.speaker?.display_name || segment.speaker_name ||
        segment.original_prediction?.speaker_name || segment.metadata?.speaker_name || '' });
    groups.get(localId).count++;
  }
  return [...groups.values()];
}

export function speakerCorrectionPayload({ registeredSpeakerId, displayName }) {
  if (registeredSpeakerId) return { registered_speaker_id: registeredSpeakerId };
  const cleanName = String(displayName || '').trim();
  if (cleanName) return { display_name: cleanName };
  throw new Error('请选择已注册人员或填写显示名称');
}

export function authorizationHeaders(token, base = {}) {
  const headers = new Headers(base);
  if (token?.trim()) headers.set('Authorization', `Bearer ${token.trim()}`);
  return headers;
}

export function streamStartMessage(options = {}) {
  return {
    type: 'start', protocol_version: 1, sample_rate: 16000, format: 's16le', channels: 1,
    hotword_set_id: options.hotwordSetId || null,
    ...(options.apiKey?.trim() ? { api_key: options.apiKey.trim() } : {}),
  };
}
