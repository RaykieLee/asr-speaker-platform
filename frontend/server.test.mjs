import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import http from 'node:http';
import net from 'node:net';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

function close(server) {
  return new Promise(resolve => server.close(() => resolve()));
}

async function freePort() {
  const server = net.createServer();
  const port = await listen(server);
  await close(server);
  return port;
}

async function waitForFrontend(url, child) {
  for (let attempt = 0; attempt < 50; attempt++) {
    if (child.exitCode !== null) throw new Error(`frontend server exited: ${child.exitCode}`);
    try { return await fetch(url); } catch {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('frontend server did not start');
}

test('frontend proxy forwards liveness and readiness responses to backend', async () => {
  const paths = [];
  const backend = http.createServer((request, response) => {
    paths.push(request.url);
    if (request.url === '/ready') {
      response.writeHead(503, { 'content-type': 'application/json' });
      response.end(JSON.stringify({
        status: 'not_ready',
        checks: { online_asr: false, file_storage: true },
      }));
      return;
    }
    if (request.url === '/health') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ status: 'ok' }));
      return;
    }
    response.writeHead(404).end();
  });
  const backendPort = await listen(backend);
  const frontendPort = await freePort();
  const child = spawn(process.execPath, ['frontend/server.mjs'], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      ASR_BACKEND_URL: `http://127.0.0.1:${backendPort}`,
      FRONTEND_PORT: String(frontendPort),
    },
    stdio: 'ignore',
  });
  try {
    const base = `http://127.0.0.1:${frontendPort}`;
    const liveness = await waitForFrontend(`${base}/health`, child);
    assert.equal(liveness.status, 200);
    assert.deepEqual(await liveness.json(), { status: 'ok' });

    const readiness = await fetch(`${base}/ready`);
    assert.equal(readiness.status, 503);
    assert.deepEqual(await readiness.json(), {
      status: 'not_ready',
      checks: { online_asr: false, file_storage: true },
    });
    assert.deepEqual(paths, ['/health', '/ready']);
  } finally {
    child.kill('SIGTERM');
    await new Promise(resolve => {
      if (child.exitCode !== null) resolve();
      else child.once('exit', resolve);
    });
    await close(backend);
  }
});

test('only the curated demo WAV is served from the repository assets', async () => {
  const frontendPort = await freePort();
  const child = spawn(process.execPath, ['frontend/server.mjs'], {
    cwd: process.cwd(),
    env: {...process.env, ASR_BACKEND_URL: 'http://127.0.0.1:1', FRONTEND_PORT: String(frontendPort)},
    stdio: 'ignore',
  });
  try {
    const base = `http://127.0.0.1:${frontendPort}`;
    const page = await (await waitForFrontend(base, child)).text();
    assert.match(page, /id="load-demo-audio"/);
    assert.match(page, /AISHELL-3 测试集/);
    const url = `${base}/demo-assets/aishell3/two-speaker-abab.wav`;
    const response = await waitForFrontend(url, child);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type'), /^audio\/wav/);
    const bytes = Buffer.from(await response.arrayBuffer());
    const expected = await readFile(resolve('demo_assets/aishell3/two-speaker-abab.wav'));
    assert.equal(createHash('sha256').update(bytes).digest('hex'),
      createHash('sha256').update(expected).digest('hex'));

    const head = await fetch(url, {method: 'HEAD'});
    assert.equal(head.status, 200);
    assert.equal(Number(head.headers.get('content-length')), expected.length);
    const sourceFile = await fetch(`${base}/demo-assets/aishell3/source/SSB0394/SSB03940088.wav`);
    assert.equal(sourceFile.status, 404);
    const traversal = await fetch(`${base}/..%2f..%2fdemo_assets/aishell3/source_selection.json`);
    assert.equal(traversal.status, 403);
    const post = await fetch(url, {method: 'POST'});
    assert.equal(post.status, 405);
  } finally {
    child.kill('SIGTERM');
    await new Promise(resolve => {
      if (child.exitCode !== null) resolve();
      else child.once('exit', resolve);
    });
  }
});
