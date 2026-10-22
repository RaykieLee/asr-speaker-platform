import test from 'node:test';
import assert from 'node:assert/strict';
import { authorizationHeaders, formatTime, hotwordEntriesToText, hotwordId,
  groupCorrectionTargets, mergeHotwordEntries, normalizeJobForDisplay, percentile,
  resampleTo16k, speakerCorrectionPayload, streamStartMessage, toS16LE } from './public/logic.mjs';

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

test('hotword list supports database id and object entries without losing metadata', () => {
  const entries = [{word:'Ada', aliases:['A.'], language:'en', enabled:true}, '李子熙'];
  assert.equal(hotwordId({id:'hw_1'}), 'hw_1');
  assert.equal(hotwordEntriesToText(entries), 'Ada\n李子熙');
  assert.deepEqual(mergeHotwordEntries('Ada\n新词', entries), [
    {word:'Ada', aliases:['A.'], language:'en', enabled:true}, '新词',
  ]);
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
