import test from 'node:test';
import assert from 'node:assert/strict';
import { percentile, resampleTo16k, toS16LE, formatTime } from './public/logic.mjs';

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
