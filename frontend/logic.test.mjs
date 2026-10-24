import test from 'node:test';
import assert from 'node:assert/strict';
import { authorizationHeaders, compareHotwordTranscripts, dedupeHotwordEntries, fileTranscriptionSettings, formatTime, hotwordEntriesToText, hotwordId,
  groupCorrectionTargets, hotwordComparisonPlan, mergeHotwordEntries, normalizeJobForDisplay, parseHotwordImport, percentile, readinessLabel,
  resampleTo16k, serializeHotwordEntries, shouldApplyLiveSegment, speakerCorrectionPayload, streamStartMessage, toS16LE, trackTranscriptionJob } from './public/logic.mjs';

test('live offline revisions replace the same segment without accepting stale events', () => {
  assert.equal(shouldApplyLiveSegment(undefined, {revision: 1}), true);
  assert.equal(shouldApplyLiveSegment({revision: 5}, {revision: 6}), true);
  assert.equal(shouldApplyLiveSegment({revision: 6}, {revision: 5}), false);
});

test('percentile uses nearest-rank and handles empty input', () => {
  assert.equal(percentile([10, 1, 3, 2], 0.5), 2);
  assert.equal(percentile([], 0.95), null);
});

test('audio helpers resample and encode signed little-endian PCM', () => {
  assert.deepEqual([...resampleTo16k(Float32Array.from([0, 1, 0, -1]), 32000)], [0.5, -0.5]);
  assert.deepEqual([...new Int16Array(toS16LE(Float32Array.from([-1, 0, 1])))], [-32768, 0, 32767]);
});

test('time display is stable for missing and long durations', () => {
  assert.equal(formatTime(125), '02:05');
  assert.equal(formatTime(null), '—');
});

test('readiness distinguishes an offline process, missing models and a ready service', () => {
  assert.deepEqual(readinessLabel({status:'ready',checks:{}}), {text:'服务就绪',ready:true});
  assert.deepEqual(readinessLabel({status:'not_ready',checks:{online_asr:false}}), {text:'模型未就绪',ready:false});
  assert.deepEqual(readinessLabel({status:'not_ready',checks:{database:false}}), {text:'服务未就绪',ready:false});
});

test('hotword list supports database id and object entries without losing metadata', () => {
  const entries = [{word:'Ada', aliases:['A.'], language:'en', enabled:true}, '李子熙'];
  assert.equal(hotwordId({id:'hw_1'}), 'hw_1');
  assert.equal(hotwordEntriesToText(entries), 'Ada\n李子熙');
  assert.deepEqual(mergeHotwordEntries('Ada\n新词', entries), [
    {word:'Ada', aliases:['A.'], language:'en', enabled:true}, {word:'新词',aliases:[],language:null,enabled:true},
  ]);
});

test('hotword import parses aliases, language and enabled CSV values and deduplicates preview', () => {
  const imported = parseHotwordImport('word,aliases,language,enabled\n"云端工作台","云端|云台",zh,true\nAda,,en,false', 'csv');
  assert.deepEqual(imported, [
    {word:'云端工作台',aliases:['云端','云台'],language:'zh',enabled:true},
    {word:'Ada',aliases:[],language:'en',enabled:false},
  ]);
  assert.deepEqual(parseHotwordImport('词\n英文词', 'paste').map(entry=>entry.language), [null, null]);
  assert.equal(parseHotwordImport('word,aliases\n词,', 'csv')[0].language, null);
  assert.deepEqual(serializeHotwordEntries([
    {word:'旧词',aliases:[],language:'',enabled:true},
    {word:'自动词',aliases:[],language:'auto',enabled:true},
    {word:'标注词',aliases:['别名'],language:'zh',enabled:false},
  ]), [
    {word:'旧词',aliases:[],language:null,enabled:true},
    {word:'自动词',aliases:[],language:null,enabled:true},
    {word:'标注词',aliases:['别名'],language:'zh',enabled:false},
  ]);
  const preview = dedupeHotwordEntries(imported, [{word:'云端工作台', aliases:['云端'], language:'zh', enabled:true}]);
  assert.deepEqual(preview.added, [{word:'Ada',aliases:[],language:'en',enabled:false}]);
  assert.deepEqual(preview.duplicates, ['云端工作台']);
});

test('plain bulk paste is one enabled term per line and rejects malformed CSV quoting', () => {
  assert.deepEqual(parseHotwordImport('甲\n 乙 \n\n丙', 'paste').map(item=>item.word), ['甲','乙','丙']);
  assert.throws(() => parseHotwordImport('word,aliases\n"unfinished', 'csv'), /未闭合/);
});

test('hotword comparison exposes raw task transcripts without claiming quality gain', () => {
  const result = compareHotwordTranscripts(
    {job_id:'job-off',status:'succeeded',segments:[{text:'云端工作台'}]},
    {job_id:'job-on',status:'succeeded',segments:[{text:'云端工作台'}]},
  );
  assert.equal(result.sameTranscript, true);
  assert.equal(result.without.jobId, 'job-off');
  assert.equal(result.with.text, '云端工作台');
  assert.match(result.comparisonLabel, /未使用人工真值/);
  assert.deepEqual(hotwordComparisonPlan('set-1'), [
    {mode:'off',hotwordSetId:null}, {mode:'on',hotwordSetId:'set-1'},
  ]);
  assert.throws(() => hotwordComparisonPlan(''), /先选择热词方案/);
});

test('file job settings are captured once and created jobs remain available to export and correction actions', () => {
  const settings = fileTranscriptionSettings('zh','separate');
  assert.deepEqual(settings,{language:'zh',channelMode:'separate'});
  assert.equal(Object.isFrozen(settings),true);
  const state = {job:null};
  const created = {job_id:'job-123',status_url:'/api/v1/transcriptions/job-123'};
  assert.equal(trackTranscriptionJob(state,created),created);
  assert.equal(state.job.job_id,'job-123');
  assert.equal(state.job.status_url,'/api/v1/transcriptions/job-123');
});

test('job display uses effective segments and authoritative metadata', () => {
  const display = normalizeJobForDisplay({
    result_metadata:{duration:73.5, speaker_diarization:'completed', text:'Hello'},
    effective_segments:[{start_seconds:1.25,end_seconds:2.5,text:'Hi',local_speaker_id:'speaker_0',speaker:{display_name:'Mina'}}],
  });
  assert.equal(display.duration, 73.5);
  assert.equal(display.speakerDiarization, 'completed');
  assert.deepEqual(display.segments[0], {
    start_seconds:1.25,end_seconds:2.5,text:'Hi',local_speaker_id:'speaker_0',speaker:{display_name:'Mina'},
    start:1.25,end:2.5,speaker_name:'Mina',
  });
});

test('optional API token is attached to HTTP and WebSocket authentication', () => {
  assert.equal(authorizationHeaders(' secret ').get('Authorization'), 'Bearer secret');
  assert.equal(authorizationHeaders('').has('Authorization'), false);
  assert.deepEqual(streamStartMessage({hotwordSetId:'hw_1',apiKey:' secret '}), {
    type:'start',protocol_version:1,sample_rate:16000,format:'s16le',channels:1,
    hotword_set_id:'hw_1',api_key:'secret',
  });
});

test('speaker correction groups segments by local identity and creates exclusive payloads', () => {
  assert.deepEqual(groupCorrectionTargets([
    {local_speaker_id:'speaker_0',speaker:{display_name:'预测甲'}},
    {local_speaker_id:'speaker_0'},
    {local_speaker_id:'speaker_1'},
    {text:'no diarization'},
  ]), [
    {local_speaker_id:'speaker_0',count:2,display_name:'预测甲'},
    {local_speaker_id:'speaker_1',count:1,display_name:''},
  ]);
  assert.deepEqual(speakerCorrectionPayload({registeredSpeakerId:'spk_1',displayName:'ignored'}), {registered_speaker_id:'spk_1'});
  assert.deepEqual(speakerCorrectionPayload({displayName:'  嘉宾 A  '}), {display_name:'嘉宾 A'});
  assert.throws(() => speakerCorrectionPayload({}), /请选择/);
});
