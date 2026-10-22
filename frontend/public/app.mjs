import { authorizationHeaders, formatTime, hotwordEntriesToText, hotwordId,
  mergeHotwordEntries, normalizeJobForDisplay, resampleTo16k, streamStartMessage, toS16LE } from './logic.mjs';

const $ = (id) => document.getElementById(id);
const state = { hotwords: [], speakers: [], selectedSpeaker: null, selectedHotword: null,
  live: null, liveSegments: new Map(), job: null, jobData: null, poll: null };
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
function setConnection(ok, text) {
  const pill = $('connection-pill'); pill.classList.toggle('online', ok); pill.classList.toggle('offline', !ok);
  pill.lastElementChild.textContent = text; $('sidebar-status-dot').classList.toggle('online', ok);
  $('notice-title').textContent = ok ? '识别服务已连接' : '无法连接识别服务';
  $('notice-copy').textContent = ok ? '可以开始调用已配置的识别能力。' : '请检查服务地址和后端进程；模型未配置时，相应操作会返回明确错误。';
}
async function checkHealth() {
  $('footer-api').textContent = apiBase();
  try { await request('/health'); setConnection(true, '服务在线'); }
  catch (error) { setConnection(false, '服务离线'); }
}
function selectOptions(select, selected = '') {
  const keep = select.value || selected;
  select.innerHTML = '<option value="">不使用热词</option>' + state.hotwords.filter(x => x.status === 'active').map(x => `<option value="${escapeHtml(hotwordId(x))}">${escapeHtml(x.name)} · v${x.current_version || x.version || 1}</option>`).join('');
  select.value = keep;
}
async function loadHotwords() {
  try {
    const result = await request('/api/v1/hotword-sets?status=active');
    state.hotwords = Array.isArray(result) ? result : result.items || [];
    $('hotword-count').textContent = state.hotwords.length;
    $('hotword-list').innerHTML = state.hotwords.length ? state.hotwords.map(item => { const id=hotwordId(item); return `<button class="library-row ${hotwordId(state.selectedHotword) === id ? 'chosen' : ''}" data-hotword="${escapeHtml(id)}"><span class="library-glyph">✳</span><span><strong>${escapeHtml(item.name)}</strong><small>${(item.entries || []).length} 个词条 · v${item.current_version || item.version || 1}</small></span><b>→</b></button>`; }).join('') : '<div class="empty-state compact"><strong>还没有热词方案</strong><p>创建后可在实时和文件识别中选用。</p></div>';
    state.hotwords.forEach(item => $('hotword-list').querySelector(`[data-hotword="${CSS.escape(hotwordId(item))}"]`)?.addEventListener('click', () => editHotword(item)));
    selectOptions($('live-hotword')); selectOptions($('file-hotword'));
  } catch (error) { toast(`读取热词失败：${error.message}`, 'error'); }
}
function editHotword(item) {
  state.selectedHotword = item; $('hotword-editor-empty').classList.add('hidden'); $('hotword-editor-form').classList.remove('hidden');
  $('hotword-name').value = item.name; $('hotword-entries').value = hotwordEntriesToText(item.entries || []);
  $('hotword-weight').value = item.weight ?? 1; $('weight-value').textContent = $('hotword-weight').value;
  $('hotword-version').textContent = `版本 ${item.current_version || item.version || 1}`; $('hotword-editor-title').textContent = item.name;
  $('save-hotword').textContent = '保存新版本'; $('deactivate-hotword').classList.remove('hidden');
  document.querySelectorAll('[data-hotword]').forEach(n => n.classList.toggle('chosen', n.dataset.hotword === hotwordId(item)));
}
function newHotword() {
  state.selectedHotword = null; $('hotword-editor-empty').classList.add('hidden'); $('hotword-editor-form').classList.remove('hidden');
  $('hotword-name').value = ''; $('hotword-entries').value = ''; $('hotword-weight').value = 1;
  $('weight-value').textContent = '1'; $('hotword-version').textContent = '新方案'; $('hotword-editor-title').textContent = '新建方案';
  $('save-hotword').textContent = '创建方案'; $('deactivate-hotword').classList.add('hidden');
}
async function saveHotword() {
  clearError($('hotword-error'));
  const entries = mergeHotwordEntries($('hotword-entries').value, state.selectedHotword?.entries || []);
  const payload = { name: $('hotword-name').value.trim(), entries, weight: Number($('hotword-weight').value) };
  try {
    const editing = state.selectedHotword;
    const saved = editing
      ? await request(`/api/v1/hotword-sets/${encodeURIComponent(hotwordId(editing))}`, { method: 'PUT', headers: {'content-type':'application/json'}, body: JSON.stringify({...payload, expected_version: editing.current_version || editing.version}) })
      : await request('/api/v1/hotword-sets', { method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify(payload) });
    toast('热词方案已保存'); await loadHotwords();
    const id = hotwordId(saved) || hotwordId(editing);
    const updated = state.hotwords.find(x=>hotwordId(x)===id) || state.hotwords.find(x=>x.name===payload.name);
    if(updated) editHotword(updated);
  } catch (error) { showError($('hotword-error'), error); }
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
  if (segments.length) $('file-results').innerHTML = segments.map(s => `<div class="result-segment"><div class="result-time">${formatTime(s.start)} – ${formatTime(s.end)} <span>${escapeHtml(s.speaker_name || '说话人未标注')}</span></div><p>${escapeHtml(s.text || '')}</p></div>`).join('');
  else if (display.text) $('file-results').innerHTML = `<div class="result-segment"><p>${escapeHtml(display.text)}</p></div>`;
  if (job.status === 'failed') { $('upload-error').textContent = job.error_message || '任务失败'; $('upload-error').classList.remove('hidden'); }
  else clearError($('upload-error'));
  $('result-speaker-status').textContent = `说话人处理：${display.speakerDiarization}`;
}
async function submitFile() {
  const file = $('audio-file').files[0]; if (!file) return;
  clearError($('upload-error')); $('upload-progress').classList.remove('hidden'); $('job-state').textContent = '正在上传'; $('job-id-copy').textContent = '';
  const data = new FormData(); data.append('file', file); data.append('language', $('file-language').value); data.append('channel_mode','mixdown'); if($('file-hotword').value) data.append('hotword_set_id',$('file-hotword').value);
  try {
    const created = await request('/api/v1/transcriptions', {method:'POST', body:data}); state.job = created; $('job-id-copy').textContent = `任务 ${created.job_id}`;
    const poll = async () => { const job = await request(created.status_url); renderJob(job); if(['queued','running','pending'].includes(job.status)) state.poll = setTimeout(poll,1200); };
    await poll();
  } catch(error) { showError($('upload-error'), error); $('job-state').textContent = '任务未能启动'; }
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
