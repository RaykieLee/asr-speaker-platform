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

export function readinessLabel(snapshot) {
  if (snapshot?.status === 'ready') return { text: '服务就绪', ready: true };
  const checks = snapshot?.checks || {};
  const modelChecks = ['online_asr', 'offline_asr', 'speaker_embedding', 'file_diarization'];
  return {
    text: modelChecks.some(key => checks[key] === false) ? '模型未就绪' : '服务未就绪',
    ready: false,
  };
}

export function hotwordId(item) {
  return item?.set_id || item?.id || null;
}

export function hotwordEntriesToText(entries = []) {
  return entries.map(entry => typeof entry === 'string' ? entry : entry?.word || entry?.phrase || '').filter(Boolean).join('\n');
}

export function mergeHotwordEntries(text, previous = []) {
  const prior = new Map(previous.map(normalizeHotwordEntry).filter(Boolean)
    .map(entry => [entry.word.toLocaleLowerCase(), entry]));
  return text.split(/\r?\n/).map(word => word.trim()).filter(Boolean).map(word =>
    prior.get(word.toLocaleLowerCase()) || normalizeHotwordEntry(word));
}

export function normalizeHotwordEntry(raw = {}) {
  if (typeof raw === 'string') raw = { word: raw };
  const word = String(raw?.word ?? raw?.phrase ?? '').trim();
  if (!word) return null;
  const rawLanguage = String(raw.language ?? raw.lang ?? '').trim();
  const language = rawLanguage && !['auto', '自动'].includes(rawLanguage.toLocaleLowerCase()) ? rawLanguage : null;
  return {
    word,
    aliases: Array.isArray(raw.aliases) ? raw.aliases.map(x => String(x).trim()).filter(Boolean) : [],
    language,
    enabled: raw.enabled !== false,
  };
}

export function serializeHotwordEntries(entries = []) {
  return entries.map(normalizeHotwordEntry).filter(Boolean).map(entry => ({
    ...entry,
    language: entry.language || null,
  }));
}

export function fileTranscriptionSettings(language, channelMode) {
  return Object.freeze({
    language: String(language || 'auto'),
    channelMode: String(channelMode || 'mixdown'),
  });
}

export function trackTranscriptionJob(state, createdJob) {
  state.job = createdJob;
  return createdJob;
}

function parseCsvRows(text) {
  const rows = [];
  let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); if (row.some(value => value.trim())) rows.push(row);
      row = []; cell = '';
    } else cell += char;
  }
  if (quoted) throw new Error('CSV 含有未闭合的双引号');
  row.push(cell); if (row.some(value => value.trim())) rows.push(row);
  return rows;
}

export function parseHotwordImport(text, format = 'paste') {
  if (format === 'paste') {
    return text.split(/\r?\n/).map(line => line.trim()).filter(Boolean).map(word => normalizeHotwordEntry(word));
  }
  if (format !== 'csv') throw new Error('不支持的导入格式');
  const rows = parseCsvRows(text);
  if (!rows.length) return [];
  const canonical = value => value.trim().toLowerCase().replace(/[ _-]/g, '');
  const first = rows[0].map(canonical);
  const hasHeader = first.some(name => ['word', 'phrase', 'hotword'].includes(name));
  const headers = hasHeader ? first : ['word', 'aliases', 'language', 'enabled'];
  const dataRows = hasHeader ? rows.slice(1) : rows;
  return dataRows.map((cells, index) => {
    const record = Object.fromEntries(headers.map((header, i) => [header, cells[i] || '']));
    const word = String(record.word || record.phrase || record.hotword || '').trim();
    if (!word) throw new Error(`CSV 第 ${index + (hasHeader ? 2 : 1)} 行缺少 word`);
    const rawEnabled = String(record.enabled ?? '').trim().toLowerCase();
    if (rawEnabled && !['true', 'false', '1', '0', 'yes', 'no', '启用', '停用'].includes(rawEnabled)) {
      throw new Error(`CSV 第 ${index + (hasHeader ? 2 : 1)} 行 enabled 只能填写 true/false`);
    }
    const enabled = !['false', '0', 'no', '停用'].includes(rawEnabled);
    return normalizeHotwordEntry({
      word,
      aliases: String(record.aliases || record.alias || '').split('|').map(value => value.trim()).filter(Boolean),
      language: record.language || record.lang || '',
      enabled,
    });
  });
}

export function dedupeHotwordEntries(entries = [], existing = []) {
  const result = existing.map(normalizeHotwordEntry).filter(Boolean);
  const seen = new Set();
  for (const entry of result) for (const phrase of [entry.word, ...entry.aliases]) seen.add(phrase.toLocaleLowerCase());
  const duplicates = [];
  const added = [];
  for (const raw of entries) {
    const entry = normalizeHotwordEntry(raw);
    if (!entry) continue;
    const key = entry.word.toLocaleLowerCase();
    if (seen.has(key)) { duplicates.push(entry.word); continue; }
    seen.add(key);
    entry.aliases = entry.aliases.filter(alias => {
      const aliasKey = alias.toLocaleLowerCase();
      if (seen.has(aliasKey)) { duplicates.push(alias); return false; }
      seen.add(aliasKey); return true;
    });
    result.push(entry); added.push(entry);
  }
  return { entries: result, added, duplicates };
}

export function compareHotwordTranscripts(withoutHotwords, withHotwords) {
  const rawText = job => {
    const normalized = normalizeJobForDisplay(job || {});
    return normalized.segments.map(segment => segment.text || '').join('\n') || normalized.text || '';
  };
  const baseline = rawText(withoutHotwords);
  const boosted = rawText(withHotwords);
  return {
    sameTranscript: baseline === boosted,
    without: { jobId: withoutHotwords?.job_id || withoutHotwords?.id || null, status: withoutHotwords?.status || 'missing', text: baseline },
    with: { jobId: withHotwords?.job_id || withHotwords?.id || null, status: withHotwords?.status || 'missing', text: boosted },
    comparisonLabel: '原始转写差异；未使用人工真值计算命中率或收益',
  };
}

export function hotwordComparisonPlan(hotwordSetId) {
  if (!hotwordSetId) throw new Error('先选择热词方案，再运行开关对比');
  return [
    { mode: 'off', hotwordSetId: null },
    { mode: 'on', hotwordSetId },
  ];
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
