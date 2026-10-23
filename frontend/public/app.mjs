import { authorizationHeaders, compareHotwordTranscripts, dedupeHotwordEntries, formatTime, hotwordComparisonPlan, hotwordId,
  fileTranscriptionSettings, groupCorrectionTargets, normalizeHotwordEntry, normalizeJobForDisplay, parseHotwordImport, readinessLabel,
  resampleTo16k, serializeHotwordEntries, speakerCorrectionPayload, streamStartMessage, toS16LE,
  trackTranscriptionJob } from './logic.mjs';

const $ = (id) => document.getElementById(id);
const state = { hotwords: [], speakers: [], selectedSpeaker: null, selectedHotword: null,
  live: null, liveSegments: new Map(), job: null, jobData: null, poll: null,
  hotwordDraftEntries: [], importPreview: null };
const views = {
  live: ['实时识别', 'VOICE CAPTURE / 01', '让每段对话，都有迹可循。', '实时查看识别结果与说话人状态，录音会直接送往你配置的本地服务。'],
  files: ['文件转写', 'FILE TRANSCRIPTION / 02', '把音频整理成可检索的文字。', '上传文件后，任务会在本地识别服务中排队处理。'],
  speakers: ['声纹人员', 'VOICE IDENTITY / 03', '建立可复用的说话人档案。', '声纹样本由本地模型检查和处理。'],
  hotwords: ['热词词库', 'CONTEXT BOOST / 04', '让专有名词更容易被听见。', '每次保存都会创建新版本，已开始的识别任务保留其版本快照。']
};

function apiBase() { return $('api-origin').value.trim().replace(/\/$/, '') || location.origin; }
function apiUrl(path) { return `${apiBase()}${path}`; }
async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), { ...options, headers: authorizationHeaders($('api-token')?.value || '', options.headers) });
  const type = response.headers.get('content-type') || '';
  const body = type.includes('json') ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = body?.detail;
    const error = new Error(typeof detail === 'string' ? detail : detail?.message || body?.message || `${response.status} ${response.statusText}`);
    error.quality = detail?.quality;
    error.code = detail?.code || body?.code || body?.type || null;
    error.statusCode = response.status;
    error.payload = body;
    throw error;
  }
  return body;
}
function toast(message, kind = 'info') {
  const node = document.createElement('div'); node.className = `toast ${kind}`; node.textContent = message;
  $('toast-region').append(node); setTimeout(() => node.remove(), 4200);
}
function showError(element, error) { element.textContent = error.message || String(error); element.classList.remove('hidden'); }
function clearError(element) { element.textContent = ''; element.classList.add('hidden'); }
function escapeHtml(value = '') { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function activateView(name) {
  document.querySelectorAll('.view').forEach(node => node.classList.toggle('active', node.id === `view-${name}`));
  document.querySelectorAll('.nav-item').forEach(node => node.classList.toggle('active', node.dataset.view === name));
  const [title, eyebrow, heading, description] = views[name];
  $('current-section').textContent = title; $('page-eyebrow').textContent = eyebrow;
  $('page-title').textContent = heading; $('page-description').textContent = description;
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (name === 'speakers') loadSpeakers(); if (name === 'hotwords') loadHotwords();
}
function setConnection(alive, text, ready = alive) {
  const pill = $('connection-pill'); pill.classList.toggle('online', alive && ready);
  pill.classList.toggle('degraded', alive && !ready); pill.classList.toggle('offline', !alive);
  pill.lastElementChild.textContent = text; $('sidebar-status-dot').classList.toggle('online', alive && ready);
  $('notice-title').textContent = !alive ? '无法连接识别服务' : ready ? '识别服务已就绪' : '服务在线但未就绪';
  $('notice-copy').textContent = !alive ? '请检查服务地址和后端进程。' : ready
    ? '在线与离线识别、声纹和分段模型均可用。'
    : '进程仍在运行，但模型或服务组件尚未就绪；请检查后端模型配置。';
}
async function checkHealth() {
  $('footer-api').textContent = apiBase();
  try { await request('/health'); }
  catch (error) { setConnection(false, '服务离线'); return; }
  try {
    const result = await request('/ready');
    const label = readinessLabel(result);
    setConnection(true, label.text, label.ready);
  } catch (error) {
    if (error.statusCode === 503) {
      const label = readinessLabel(error.payload);
      setConnection(true, label.text, false);
    } else setConnection(true, '状态暂不可用', false);
  }
}
function selectOptions(select, selected = '') {
  const keep = select.value || selected;
  select.innerHTML = '<option value="">不使用热词</option>' + state.hotwords.filter(x => x.status === 'active').map(x => `<option value="${escapeHtml(hotwordId(x))}">${escapeHtml(x.name)} · v${x.current_version || x.version || 1}</option>`).join('');
  select.value = keep;
}
async function loadHotwords() {
  try {
    const result = await request('/api/v1/hotword-sets?status=all');
    state.hotwords = Array.isArray(result) ? result : result.items || [];
    $('hotword-count').textContent = state.hotwords.length;
    $('hotword-list').innerHTML = state.hotwords.length ? state.hotwords.map(item => { const id=hotwordId(item); return `<button class="library-row ${hotwordId(state.selectedHotword) === id ? 'chosen' : ''}" data-hotword="${escapeHtml(id)}"><span class="library-glyph">✳</span><span><strong>${escapeHtml(item.name)}</strong><small>${(item.entries || []).length} 个词条 · v${item.current_version || item.version || 1} · ${item.status === 'inactive' ? '已停用' : '启用中'}</small></span><b>→</b></button>`; }).join('') : '<div class="empty-state compact"><strong>还没有热词方案</strong><p>创建后可在实时和文件识别中选用。</p></div>';
    state.hotwords.forEach(item => $('hotword-list').querySelector(`[data-hotword="${CSS.escape(hotwordId(item))}"]`)?.addEventListener('click', () => editHotword(item)));
    selectOptions($('live-hotword')); selectOptions($('file-hotword'));
  } catch (error) { toast(`读取热词失败：${error.message}`, 'error'); }
}
function editHotword(item) {
  state.selectedHotword = item; $('hotword-editor-empty').classList.add('hidden'); $('hotword-editor-form').classList.remove('hidden');
  state.importPreview = null; $('hotword-import-preview').classList.add('hidden'); $('apply-hotword-import').disabled = true;
  state.hotwordDraftEntries = (item.entries || []).map(normalizeHotwordEntry).filter(Boolean);
  renderHotwordEntries(); $('hotword-name').value = item.name;
  $('hotword-weight').value = item.weight ?? 1; $('weight-value').textContent = $('hotword-weight').value;
  $('hotword-version').textContent = `版本 ${item.current_version || item.version || 1}`; $('hotword-editor-title').textContent = item.name;
  $('save-hotword').textContent = '保存新版本'; $('deactivate-hotword').classList.toggle('hidden', item.status === 'inactive');
  $('hotword-stale').classList.add('hidden'); clearError($('hotword-error')); clearError($('hotword-import-error'));
  setHotwordReadOnly(item.status === 'inactive');
  document.querySelectorAll('[data-hotword]').forEach(n => n.classList.toggle('chosen', n.dataset.hotword === hotwordId(item)));
}
function newHotword() {
  state.selectedHotword = null; $('hotword-editor-empty').classList.add('hidden'); $('hotword-editor-form').classList.remove('hidden');
  state.importPreview = null; $('hotword-import-preview').classList.add('hidden'); $('apply-hotword-import').disabled = true;
  state.hotwordDraftEntries = []; renderHotwordEntries(); $('hotword-name').value = ''; $('hotword-weight').value = 1;
  $('weight-value').textContent = '1'; $('hotword-version').textContent = '新方案'; $('hotword-editor-title').textContent = '新建方案';
  $('save-hotword').textContent = '创建方案'; $('deactivate-hotword').classList.add('hidden');
  $('hotword-stale').classList.add('hidden'); clearError($('hotword-error')); setHotwordReadOnly(false);
}
function setHotwordReadOnly(readOnly) {
  $('hotword-name').disabled = readOnly; $('hotword-weight').disabled = readOnly;
  $('add-hotword-entry').disabled = readOnly; $('preview-hotword-import').disabled = readOnly;
  $('hotword-import-text').disabled = readOnly; $('hotword-import-format').disabled = readOnly; $('hotword-csv-file').disabled = readOnly;
  $('apply-hotword-import').disabled = readOnly || !state.importPreview?.added.length;
  $('save-hotword').disabled = readOnly;
}
function renderHotwordEntries() {
  $('hotword-entry-count').textContent = state.hotwordDraftEntries.length;
  $('hotword-entry-list').innerHTML = state.hotwordDraftEntries.map((entry, index) => `<div class="hotword-entry" data-entry-index="${index}"><label>热词<input data-entry-field="word" value="${escapeHtml(entry.word)}" maxlength="256" placeholder="必填词或短语"></label><label>可选别名<input data-entry-field="aliases" value="${escapeHtml(entry.aliases.join(' | '))}" placeholder="别名 A | 别名 B"></label><label>语言<input data-entry-field="language" value="${escapeHtml(entry.language || '')}" placeholder="自动"></label><label class="entry-enabled"><input data-entry-field="enabled" type="checkbox" ${entry.enabled ? 'checked' : ''}>启用</label><button type="button" class="row-delete" data-entry-remove aria-label="删除热词">×</button></div>`).join('') || '<div class="empty-import">尚无词条。可手动添加，或从下方批量导入。</div>';
  $('hotword-entry-list').querySelectorAll('[data-entry-field]').forEach(input => input.disabled = state.selectedHotword?.status === 'inactive');
  $('hotword-entry-list').querySelectorAll('[data-entry-remove]').forEach(button => button.disabled = state.selectedHotword?.status === 'inactive');
}
function previewHotwordImport() {
  clearError($('hotword-import-error'));
  try {
    const parsed = parseHotwordImport($('hotword-import-text').value, $('hotword-import-format').value);
    state.importPreview = dedupeHotwordEntries(parsed, state.hotwordDraftEntries);
    const preview = $('hotword-import-preview'); preview.classList.remove('hidden');
    const words = state.importPreview.added.slice(0, 12).map(item => escapeHtml(item.word)).join('、');
    preview.innerHTML = `<strong>新增 ${state.importPreview.added.length} 条，去重 ${state.importPreview.duplicates.length} 条</strong><span>${words || '没有可添加的新词'}${state.importPreview.added.length > 12 ? '…' : ''}</span>${state.importPreview.duplicates.length ? `<small>重复项：${state.importPreview.duplicates.slice(0,8).map(escapeHtml).join('、')}${state.importPreview.duplicates.length > 8 ? '…' : ''}</small>` : ''}`;
    $('apply-hotword-import').disabled = !state.importPreview.added.length || state.selectedHotword?.status === 'inactive';
  } catch (error) { state.importPreview = null; $('apply-hotword-import').disabled = true; showError($('hotword-import-error'), error); }
}
function applyHotwordImport() {
  if (!state.importPreview) return;
  state.hotwordDraftEntries = state.importPreview.entries; renderHotwordEntries();
  $('hotword-import-text').value = ''; $('hotword-import-preview').classList.add('hidden'); $('apply-hotword-import').disabled = true;
  state.importPreview = null;
}
function reloadSelectedHotword() {
  const id = hotwordId(state.selectedHotword);
  const latest = state.hotwords.find(item => hotwordId(item) === id);
  if (latest) editHotword(latest);
}
async function saveHotword() {
  clearError($('hotword-error'));
  const entries = state.hotwordDraftEntries.map(entry => ({...entry, aliases:[...entry.aliases]}));
  if (entries.some(entry => !entry.word.trim())) { showError($('hotword-error'), new Error('请填写每条热词的词或短语。')); return; }
  const payload = { name: $('hotword-name').value.trim(), entries: serializeHotwordEntries(entries), weight: Number($('hotword-weight').value) };
  try {
    const editing = state.selectedHotword;
    const saved = editing
      ? await request(`/api/v1/hotword-sets/${encodeURIComponent(hotwordId(editing))}`, { method: 'PUT', headers: {'content-type':'application/json'}, body: JSON.stringify({...payload, expected_version: editing.current_version || editing.version}) })
      : await request('/api/v1/hotword-sets', { method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify(payload) });
    toast('热词方案已保存'); await loadHotwords();
    const id = hotwordId(saved) || hotwordId(editing);
    const updated = state.hotwords.find(x=>hotwordId(x)===id) || state.hotwords.find(x=>x.name===payload.name);
    if(updated) editHotword(updated);
  } catch (error) {
    if (error.code === 'HOTWORD_VERSION_CONFLICT') $('hotword-stale').classList.remove('hidden');
    showError($('hotword-error'), error.code === 'HOTWORD_VERSION_CONFLICT'
      ? new Error('服务器上的方案版本已更新。你的草稿仍保留；确认后可重新载入最新版本。') : error);
  }
}
async function deactivateHotword() {
  if (!state.selectedHotword) return;
  try { await request(`/api/v1/hotword-sets/${encodeURIComponent(hotwordId(state.selectedHotword))}`, {method:'DELETE'}); state.selectedHotword = null; toast('方案已停用'); await loadHotwords(); $('hotword-editor-form').classList.add('hidden'); $('hotword-editor-empty').classList.remove('hidden'); }
  catch (error) { showError($('hotword-error'), error); }
}

async function loadSpeakers() {
  try {
    state.speakers = await request('/api/v1/speakers'); $('speaker-count').textContent = state.speakers.length;
    $('speaker-list').innerHTML = state.speakers.length ? state.speakers.map(person => { const id=person.speaker_id || person.id; return `<button class="library-row ${state.selectedSpeaker?.speaker_id === id ? 'chosen' : ''}" data-speaker="${escapeHtml(id)}"><span class="avatar small">${escapeHtml((person.name || '?').slice(0,1))}</span><span><strong>${escapeHtml(person.name)}</strong><small>${person.registration_status === 'registered' || person.profile ? '已有声纹 profile' : '尚未注册声纹'} · ${person.sample_count ?? (person.samples || []).length} 个样本</small></span><b>→</b></button>`; }).join('') : '<div class="empty-state compact"><strong>还没有人员记录</strong><p>创建人员后即可上传声纹样本。</p></div>';
    state.speakers.forEach(person => { const id=person.speaker_id || person.id; $('speaker-list').querySelector(`[data-speaker="${CSS.escape(id)}"]`)?.addEventListener('click', () => selectSpeaker({...person,speaker_id:id})); });
  } catch (error) { toast(`读取人员失败：${error.message}`, 'error'); }
}
function selectSpeaker(person) {
  state.selectedSpeaker = person; $('enrollment-empty').classList.add('hidden'); $('enrollment-form').classList.remove('hidden');
  $('selected-name').textContent = person.name; $('selected-avatar').textContent = (person.name || '?').slice(0,1);
  $('selected-profile').textContent = person.profile ? '已注册 profile' : '尚无已注册 profile'; $('enrollment-title').textContent = `为 ${person.name} 录入样本`;
  document.querySelectorAll('[data-speaker]').forEach(n => n.classList.toggle('chosen', n.dataset.speaker === person.speaker_id));
}
async function createSpeaker() {
  const name = prompt('输入人员姓名'); if (!name?.trim()) return;
  try { const person = await request('/api/v1/speakers', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({name:name.trim()})}); await loadSpeakers(); selectSpeaker({...person,speaker_id:person.speaker_id || person.id}); toast('人员已创建'); }
  catch(error) { toast(`创建失败：${error.message}`, 'error'); }
}
async function uploadSpeakerSample() {
  if (!state.selectedSpeaker || !$('speaker-audio').files[0]) return;
  clearError($('speaker-error')); const data = new FormData(); data.append('file', $('speaker-audio').files[0]);
  try {
    const speakerId = state.selectedSpeaker.speaker_id;
    const result = await request(`/api/v1/speakers/${encodeURIComponent(speakerId)}/samples`, {method:'POST', body:data});
    renderQuality(result.quality || {}, '样本检查完成');
    toast('样本已处理');
    const refreshed = await request(`/api/v1/speakers/${encodeURIComponent(speakerId)}`);
    await loadSpeakers(); selectSpeaker({...refreshed,speaker_id:speakerId});
  } catch(error) { if (error.quality) renderQuality(error.quality, '样本未通过检查'); showError($('speaker-error'), error); }
}

function renderQuality(quality, heading) {
  const rows = [
    ['时长', quality.duration_seconds, ' 秒'], ['有效语音', quality.speech_seconds, ' 秒'],
    ['语音占比', quality.speech_ratio, '%', true], ['整体音量 RMS', quality.rms, ''],
    ['削波占比', quality.clipped_ratio, '%', true],
  ].filter(([,value]) => Number.isFinite(value));
  $('quality-card').innerHTML = `<strong>${escapeHtml(heading)}</strong><span>${quality.is_silent ? '检测到静音' : quality.clipped_ratio > 0.01 ? '存在明显削波' : '已完成检查'}</span><div class="quality-metrics">${rows.map(([label,value,unit,ratio])=>`<small>${label}：${ratio ? (value*100).toFixed(1) : Number(value).toFixed(2)}${unit}</small>`).join('')}</div>`;
  $('quality-card').classList.remove('hidden');
}

function renderLive() {
  const root = $('live-transcript');
  if (!state.liveSegments.size) { root.innerHTML = '<div class="empty-state"><span class="empty-wave">∿</span><strong>等你开口</strong><p>partial 会持续更新，final 片段会固定显示。</p></div>'; return; }
  root.innerHTML = [...state.liveSegments.values()].sort((a,b) => a.start-b.start || a.segment_id-b.segment_id).map(s => `<div class="transcript-item ${s.status === 'partial' ? 'is-partial' : ''}"><div class="transcript-meta"><span>${formatTime(s.start)} – ${formatTime(s.end)}</span><span class="speaker-chip">${escapeHtml(s.speaker_name || s.registered_speaker_id || s.local_speaker_id || '说话人未标注')}</span><span class="segment-state">${s.status === 'partial' ? '识别中' : '已定稿'}</span></div><p>${escapeHtml(s.text || '')}</p></div>`).join('');
  $('live-segment-count').textContent = state.liveSegments.size;
  const version = $('live-hotword').selectedOptions[0]?.textContent.match(/v(\d+)/)?.[1]; $('live-hotword-version').textContent = version ? `v${version}` : '—';
  root.scrollTop = root.scrollHeight;
}
function handleLiveMessage(event) {
  let data; try { data = JSON.parse(event.data); } catch { return; }
  if (data.type === 'started') { $('live-state-title').textContent = '正在识别'; $('live-state-copy').textContent = data.hotword_set_version ? `热词快照 v${data.hotword_set_version}` : '音频已连接'; if (data.hotword_set_version) $('live-hotword-version').textContent = `v${data.hotword_set_version}`; }
  else if (data.type === 'segment') { state.liveSegments.set(data.segment_id, data); renderLive(); }
  else if (data.type === 'completed') { stopCapture(false); $('live-state-title').textContent = '会话已完成'; $('live-state-copy').textContent = `时长 ${formatTime(data.duration)}`; toast('实时识别已完成'); }
  else if (data.type === 'error') { stopCapture(false); $('live-state-title').textContent = '识别遇到问题'; $('live-state-copy').textContent = data.message; toast(data.message || '实时识别失败', 'error'); }
}
async function startLive() {
  if (!navigator.mediaDevices?.getUserMedia) { toast('当前页面无法访问麦克风，请使用 localhost 或 HTTPS。', 'error'); return; }
  $('start-live').disabled = true; $('live-state-title').textContent = '等待麦克风权限';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true}});
    const context = new AudioContext(); await context.resume();
    const wsUrl = apiUrl('/api/v1/stream').replace(/^http/, 'ws'); const socket = new WebSocket(wsUrl); socket.binaryType = 'arraybuffer';
    socket.addEventListener('open', () => socket.send(JSON.stringify(streamStartMessage({hotwordSetId:$('live-hotword').value, apiKey:$('api-token').value}))));
    socket.addEventListener('message', handleLiveMessage); socket.addEventListener('error', () => toast('实时连接失败，请确认服务已启动。', 'error'));
    const source = context.createMediaStreamSource(stream); const processor = context.createScriptProcessor(2048, 1, 1);
    processor.onaudioprocess = e => { if(socket.readyState !== WebSocket.OPEN) return; const pcm = resampleTo16k(e.inputBuffer.getChannelData(0), context.sampleRate); if(pcm.length) socket.send(toS16LE(pcm)); };
    source.connect(processor); processor.connect(context.destination);
    state.live = {stream,context,socket,source,processor,start:Date.now()}; state.liveSegments.clear(); renderLive();
    $('stop-live').disabled = false; $('mic-disc').classList.add('recording'); $('live-state-title').textContent = '正在连接'; $('live-state-copy').textContent = '等待识别会话就绪';
    state.live.timer = setInterval(() => $('live-duration').textContent = formatTime((Date.now()-state.live.start)/1000), 500);
  } catch(error) { stopCapture(false); $('live-state-title').textContent = '无法开始'; $('live-state-copy').textContent = error.message; toast(error.message, 'error'); }
}
function stopCapture(sendStop = true) {
  const live = state.live; if (!live) { if (!sendStop) $('start-live').disabled = false; return; }
  clearInterval(live.timer); live.processor.disconnect(); live.source.disconnect(); live.stream.getTracks().forEach(t=>t.stop()); live.context.close();
  if (live.socket.readyState === WebSocket.OPEN) { if(sendStop) live.socket.send(JSON.stringify({type:'stop'})); else live.socket.close(); }
  state.live = null; $('start-live').disabled = sendStop; $('stop-live').disabled = true; $('mic-disc').classList.remove('recording');
  if (sendStop) $('live-state-title').textContent = '正在收尾';
}

function renderJob(job) {
  state.jobData = job; $('job-state').textContent = job.status; $('job-percent').textContent = `${Math.round((job.progress || 0)*100)}%`;
  $('job-progress-bar').style.width = `${Math.max(4, (job.progress || 0)*100)}%`;
  $('result-summary').classList.remove('hidden'); $('result-footer').classList.remove('hidden');
  const display = normalizeJobForDisplay(job);
  $('result-status').textContent = job.status; $('result-duration').textContent = display.duration != null ? formatTime(display.duration) : '—';
  $('result-model').textContent = job.model_version || job.result_metadata?.model_version || '模型版本未知';
  const segments = display.segments;
  if (segments.length) $('file-results').innerHTML = segments.map(s => `<div class="result-segment"><div class="result-time">${formatTime(s.start)} – ${formatTime(s.end)} ${Number.isInteger(s.metadata?.channel_index) ? `<span>声道 ${s.metadata.channel_index + 1}</span>` : ''}<span>${escapeHtml(s.speaker_name || '说话人未标注')}</span></div><p>${escapeHtml(s.text || '')}</p></div>`).join('');
  else if (display.text) $('file-results').innerHTML = `<div class="result-segment"><p>${escapeHtml(display.text)}</p></div>`;
  if (job.status === 'failed') { $('upload-error').textContent = job.error_message || '任务失败'; $('upload-error').classList.remove('hidden'); }
  else clearError($('upload-error'));
  $('result-speaker-status').textContent = `说话人处理：${display.speakerDiarization}`;
  if (job.status === 'succeeded' && !state.comparisonRunning) renderSpeakerCorrections(job, segments);
  else { $('speaker-corrections').classList.add('hidden'); $('speaker-corrections').innerHTML = ''; }
}

async function renderSpeakerCorrections(job, segments) {
  const container = $('speaker-corrections');
  const jobId = job.job_id || job.id || state.job?.job_id;
  const groups = groupCorrectionTargets(segments);
  if (!groups.length) { container.classList.add('hidden'); container.innerHTML = ''; return; }
  container.classList.remove('hidden');
  container.innerHTML = '<div class="correction-heading"><strong>说话人标签纠错</strong><span>修正会同步更新结果和后续导出</span></div><p class="correction-loading">正在读取已注册人员…</p>';
  let people = [];
  let loadError = null;
  try { people = await request('/api/v1/speakers'); state.speakers = people; }
  catch (error) { loadError = error; }
  if (state.job?.job_id && state.job.job_id !== jobId) return;
  container.innerHTML = `<div class="correction-heading"><strong>说话人标签纠错</strong><span>按本地说话人统一修改，保存后会刷新有效片段与导出数据</span></div>${groups.map(group => {
    const options = people.map(person => { const id=person.speaker_id || person.id; return `<option value="${escapeHtml(id)}">${escapeHtml(person.name)}</option>`; }).join('');
    return `<div class="correction-card" data-local="${escapeHtml(group.local_speaker_id)}"><div class="correction-person"><strong>${escapeHtml(group.local_speaker_id)}</strong><span>${group.count} 个片段 · 当前：${escapeHtml(group.display_name || '未标注')}</span></div><div class="correction-fields"><label>已注册人员<select class="correction-select"><option value="">选择人员…</option>${options}</select></label><label>或填写显示名称<input class="correction-name" maxlength="100" placeholder="例如：访谈嘉宾 A"></label><button class="button button-secondary correction-save">保存纠错</button></div><div class="form-error correction-error hidden"></div></div>`;
  }).join('')}${loadError ? `<div class="form-error">读取人员列表失败：${escapeHtml(loadError.message)}；仍可使用显示名称纠错。</div>` : ''}`;
  container.querySelectorAll('.correction-card').forEach((card, index) => {
    const group = groups[index];
    const select = card.querySelector('.correction-select'); const name = card.querySelector('.correction-name');
    select.addEventListener('change', () => { name.disabled = Boolean(select.value); if (select.value) name.value = ''; });
    name.addEventListener('input', () => { select.disabled = Boolean(name.value.trim()); if (name.value.trim()) select.value = ''; });
    card.querySelector('.correction-save').addEventListener('click', () => saveSpeakerCorrection({...job, job_id:jobId}, group.local_speaker_id, card));
  });
}

async function saveSpeakerCorrection(job, localId, card) {
  const button = card.querySelector('.correction-save'); const select=card.querySelector('.correction-select'); const name=card.querySelector('.correction-name'); const errorNode=card.querySelector('.correction-error');
  clearError(errorNode); button.disabled = true; button.textContent = '正在保存…';
  try {
    const body = speakerCorrectionPayload({registeredSpeakerId:select.value, displayName:name.value});
    await request(`/api/v1/transcriptions/${encodeURIComponent(job.job_id || state.job.job_id)}/speakers/${encodeURIComponent(localId)}`, {method:'PATCH',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
    toast('说话人纠错已保存');
    const refreshed = await request(job.status_url || state.job.status_url);
    renderJob(refreshed);
  } catch(error) { showError(errorNode,error); }
  finally { if (button.isConnected) { button.disabled=false; button.textContent='保存纠错'; } }
}
async function runFileJob(file, hotwordSetId, label, settings) {
  const data = new FormData(); data.append('file', file); data.append('language', settings.language);
  data.append('channel_mode', settings.channelMode);
  if (hotwordSetId) data.append('hotword_set_id', hotwordSetId);
  const created = await request('/api/v1/transcriptions', {method:'POST', body:data});
  trackTranscriptionJob(state, created);
  $('job-id-copy').textContent = `${label === 'off' ? '关闭热词' : label === 'with' ? '启用热词' : '任务'} · ${created.job_id}`;
  while (true) {
    const job = await request(created.status_url);
    if (label !== 'off' && label !== 'with') renderJob(job);
    else {
      $('job-state').textContent = `${label === 'off' ? '关闭热词' : '启用热词'}：${job.status}`;
      $('job-percent').textContent = `${Math.round((job.progress || 0) * 100)}%`;
      $('job-progress-bar').style.width = `${Math.max(4, (job.progress || 0) * 100)}%`;
      $('result-summary').classList.remove('hidden'); $('result-status').textContent = job.status;
      $('result-model').textContent = job.model_version || job.result_metadata?.model_version || '模型版本未知';
      $('result-duration').textContent = normalizeJobForDisplay(job).duration != null ? formatTime(normalizeJobForDisplay(job).duration) : '—';
    }
    if (!['queued', 'running', 'pending'].includes(job.status)) return {...job, ...created, ...job, status_url: created.status_url};
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
}
function showComparison(withoutHotwords, withHotwords) {
  const result = compareHotwordTranscripts(withoutHotwords, withHotwords);
  const hotwordVersion = withHotwords?.hotword_set_version ?? withHotwords?.hotword_version;
  $('hotword-comparison').classList.remove('hidden');
  $('comparison-label').textContent = result.comparisonLabel;
  $('comparison-off-id').textContent = `状态：${result.without.status} · 任务：${result.without.jobId || '未创建'}`;
  $('comparison-on-id').textContent = `状态：${result.with.status} · 任务：${result.with.jobId || '未创建'} · 热词版本：${hotwordVersion ?? '服务未返回'}`;
  $('comparison-off-text').textContent = result.without.text || (result.without.status === 'succeeded' ? '服务未返回文字' : '该任务没有可展示的转写文字');
  $('comparison-on-text').textContent = result.with.text || (result.with.status === 'succeeded' ? '服务未返回文字' : '该任务没有可展示的转写文字');
  $('comparison-verdict').textContent = result.without.status !== 'succeeded' || result.with.status !== 'succeeded'
    ? '至少一个任务未成功，以上仅展示实际收到的状态和文字。'
    : result.sameTranscript ? '两次任务返回的原始文字相同。' : '两次任务返回的原始文字不同；仅为文本差异，不代表准确率或热词收益。';
  return result;
}
async function submitFile() {
  const file = $('audio-file').files[0]; if (!file) return;
  const compare = $('compare-hotwords').checked;
  const selectedId = $('file-hotword').value;
  if (compare && !selectedId) { showError($('upload-error'), new Error('请先选择热词方案，再运行同录音开 / 关对比。')); return; }
  const settings = fileTranscriptionSettings($('file-language').value, $('file-channel-mode').value);
  clearError($('upload-error')); $('upload-progress').classList.remove('hidden'); $('job-state').textContent = '正在上传'; $('job-id-copy').textContent = '';
  $('upload-file').disabled = true;
  if (compare) {
    state.comparisonRunning = true;
    $('hotword-comparison').classList.remove('hidden');
    $('comparison-label').textContent = '正在创建真实转写任务…';
    $('comparison-off-text').textContent = '排队中'; $('comparison-on-text').textContent = '等待第一项任务完成';
    $('file-results').innerHTML = '<div class="empty-state compact"><strong>同录音对比正在运行</strong><p>两项任务将顺序提交同一个文件。</p></div>';
    $('speaker-corrections').classList.add('hidden'); $('result-footer').classList.add('hidden');
  }
  try {
    if (!compare) {
      const job = await runFileJob(file, selectedId, 'single', settings);
      state.jobData = job;
      if (job.status === 'failed') showError($('upload-error'), new Error(job.error_message || '转写任务失败'));
      return;
    }
    let without = null, withWords = null;
    const [offPlan, onPlan] = hotwordComparisonPlan(selectedId);
    try { without = await runFileJob(file, offPlan.hotwordSetId, 'off', settings); }
    catch (error) { without = {status:'request_failed', error_message:error.message}; }
    $('comparison-off-id').textContent = `状态：${without.status} · 任务：${without.job_id || '未创建'}`;
    $('comparison-off-text').textContent = normalizeJobForDisplay(without).segments.map(segment=>segment.text || '').join('\n') || normalizeJobForDisplay(without).text || (without.error_message || '无转写文字');
    $('comparison-on-text').textContent = '正在运行启用热词的真实任务…';
    try { withWords = await runFileJob(file, onPlan.hotwordSetId, 'with', settings); }
    catch (error) { withWords = {status:'request_failed', error_message:error.message}; }
    state.jobData = withWords;
    showComparison(without, withWords);
    if (withWords.status === 'succeeded') renderJob(withWords);
    else showError($('upload-error'), new Error(withWords.error_message || `启用热词的任务状态：${withWords.status}`));
  } catch(error) { showError($('upload-error'), error); $('job-state').textContent = '任务未能启动'; }
  finally { state.comparisonRunning = false; $('upload-file').disabled = false; }
}
function bind() {
  document.querySelectorAll('.nav-item').forEach(b => b.addEventListener('click',()=>activateView(b.dataset.view)));
  document.querySelectorAll('[data-navigate]').forEach(b=>b.addEventListener('click',()=>activateView(b.dataset.navigate)));
  $('refresh-button').addEventListener('click', async()=>{await checkHealth(); await loadHotwords();});
  $('api-origin').value = localStorage.getItem('asr-api-origin') || '';
  $('api-origin').addEventListener('change',()=>{localStorage.setItem('asr-api-origin',$('api-origin').value.trim()); checkHealth();});
  $('api-token').value = sessionStorage.getItem('asr-api-token') || '';
  $('api-token').addEventListener('change',()=>sessionStorage.setItem('asr-api-token',$('api-token').value.trim()));
  $('dismiss-notice').addEventListener('click',()=>$('service-notice').classList.add('hidden'));
  $('start-live').addEventListener('click',startLive); $('stop-live').addEventListener('click',()=>stopCapture(true));
  $('clear-live').addEventListener('click',()=>{state.liveSegments.clear();renderLive();});
  $('audio-file').addEventListener('change',e=>{const f=e.target.files[0]; $('file-prompt').textContent=f?.name || '拖入音频文件，或 浏览文件'; $('file-details').textContent=f ? `${(f.size/1024/1024).toFixed(1)} MB · ${f.type || '音频文件'}` : '支持常见音频格式，最大 512 MB'; $('upload-file').disabled=!f;});
  $('dropzone').addEventListener('dragover',e=>{e.preventDefault();$('dropzone').classList.add('dragging');}); $('dropzone').addEventListener('dragleave',()=> $('dropzone').classList.remove('dragging'));
  $('dropzone').addEventListener('drop',e=>{e.preventDefault();$('dropzone').classList.remove('dragging');if(e.dataTransfer.files[0]){$('audio-file').files=e.dataTransfer.files;$('audio-file').dispatchEvent(new Event('change'));}});
  $('upload-file').addEventListener('click',submitFile); $('copy-result').addEventListener('click',()=>navigator.clipboard?.writeText(normalizeJobForDisplay(state.jobData || {}).segments.map(s=>s.text).join('\n') || normalizeJobForDisplay(state.jobData || {}).text).then(()=>toast('已复制转写文本')));
  $('hotword-entry-list').addEventListener('input', event => {
    const field = event.target.dataset.entryField; if (!field) return;
    const index = Number(event.target.closest('[data-entry-index]')?.dataset.entryIndex);
    const entry = state.hotwordDraftEntries[index]; if (!entry) return;
    if (field === 'word') entry.word = event.target.value;
    if (field === 'aliases') entry.aliases = event.target.value.split(/[;|]/).map(value => value.trim()).filter(Boolean);
    if (field === 'language') entry.language = event.target.value;
    if (field === 'enabled') entry.enabled = event.target.checked;
  });
  $('hotword-entry-list').addEventListener('change', event => {
    if (event.target.dataset.entryField !== 'enabled') return;
    const index = Number(event.target.closest('[data-entry-index]')?.dataset.entryIndex);
    if (state.hotwordDraftEntries[index]) state.hotwordDraftEntries[index].enabled = event.target.checked;
  });
  $('hotword-entry-list').addEventListener('click', event => {
    if (!event.target.closest('[data-entry-remove]')) return;
    const index = Number(event.target.closest('[data-entry-index]')?.dataset.entryIndex);
    state.hotwordDraftEntries.splice(index, 1); renderHotwordEntries();
  });
  $('add-hotword-entry').addEventListener('click', () => { state.hotwordDraftEntries.push({word:'',aliases:[],language:'',enabled:true}); renderHotwordEntries(); $('hotword-entry-list').lastElementChild?.querySelector('[data-entry-field="word"]')?.focus(); });
  $('preview-hotword-import').addEventListener('click',previewHotwordImport);
  $('apply-hotword-import').addEventListener('click',applyHotwordImport);
  $('hotword-import-format').addEventListener('change',()=>{state.importPreview=null;$('hotword-import-preview').classList.add('hidden');$('apply-hotword-import').disabled=true;});
  $('hotword-csv-file').addEventListener('change',async event=>{const file=event.target.files?.[0];if(!file)return;try{$('hotword-import-format').value='csv';$('hotword-import-text').value=await file.text();previewHotwordImport();}catch(error){showError($('hotword-import-error'),error);}finally{event.target.value='';}});
  $('hotword-import-text').addEventListener('input',()=>{state.importPreview=null;$('hotword-import-preview').classList.add('hidden');$('apply-hotword-import').disabled=true;});
  $('reload-hotword-version').addEventListener('click',async()=>{const id=hotwordId(state.selectedHotword);await loadHotwords();const latest=state.hotwords.find(item=>hotwordId(item)===id);if(latest)editHotword(latest);});
  $('compare-hotwords').addEventListener('change',event=>{if(!event.target.checked)$('hotword-comparison').classList.add('hidden');});
  document.querySelectorAll('[data-export-format]').forEach(button=>button.addEventListener('click',()=>downloadExport(button.dataset.exportFormat)));
  $('new-speaker-button').addEventListener('click',createSpeaker); $('refresh-speakers').addEventListener('click',loadSpeakers);
  $('speaker-audio').addEventListener('change',e=>{$('speaker-file-name').textContent=e.target.files[0]?.name||'选择语音样本';$('upload-speaker-sample').disabled=!e.target.files[0];});
  $('upload-speaker-sample').addEventListener('click',uploadSpeakerSample);
  $('new-hotword-button').addEventListener('click',newHotword); $('refresh-hotwords').addEventListener('click',loadHotwords); $('save-hotword').addEventListener('click',saveHotword); $('deactivate-hotword').addEventListener('click',deactivateHotword);
  $('hotword-weight').addEventListener('input',()=> $('weight-value').textContent=$('hotword-weight').value);
}

async function downloadExport(format) {
  if (!state.job?.job_id) return;
  try {
    const response = await fetch(apiUrl(`/api/v1/transcriptions/${encodeURIComponent(state.job.job_id)}/export?format=${encodeURIComponent(format)}`), {headers: authorizationHeaders($('api-token').value)});
    if (!response.ok) { const data=await response.json().catch(()=>({})); throw new Error(data.message || data.detail?.message || `导出失败 (${response.status})`); }
    const blob = await response.blob(); const anchor=document.createElement('a');
    anchor.href=URL.createObjectURL(blob); anchor.download=`transcription-${state.job.job_id}.${format}`;
    anchor.click(); setTimeout(()=>URL.revokeObjectURL(anchor.href),1000);
  } catch(error) { toast(error.message,'error'); }
}
bind(); checkHealth(); loadHotwords();
