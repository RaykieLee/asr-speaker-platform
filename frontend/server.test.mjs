import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import http from 'node:http';
import net from 'node:net';

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
