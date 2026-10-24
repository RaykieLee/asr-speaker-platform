import http from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('./public/', import.meta.url)));
const demoAudio = resolve(root, '../../demo_assets/aishell3/two-speaker-abab.wav');
const backend = new URL(process.env.ASR_BACKEND_URL || 'http://127.0.0.1:8000');
const port = Number(process.env.FRONTEND_PORT || 4173);
const mime = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.png':'image/png'};

function proxy(req, res) {
  const target = new URL(req.url, backend);
  const upstream = http.request(target, {method:req.method, headers:{...req.headers, host:backend.host}}, response => {
    res.writeHead(response.statusCode || 502, response.headers); response.pipe(res);
  });
  upstream.setTimeout(5000, () => upstream.destroy(new Error('识别服务请求超时')));
  upstream.on('error', error => { res.writeHead(502, {'content-type':'application/json; charset=utf-8'}); res.end(JSON.stringify({detail:`识别服务不可达：${error.message}`})); });
  req.pipe(upstream);
}

const server = http.createServer((req,res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/') || pathname === '/health' || pathname === '/ready' || pathname === '/docs' || pathname.startsWith('/docs/') || pathname === '/openapi.json' || pathname.startsWith('/redoc')) return proxy(req,res);
  if (pathname === '/demo-assets/aishell3/two-speaker-abab.wav') {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, { 'allow': 'GET, HEAD' }).end(); return;
    }
    try {
      const size = statSync(demoAudio).size;
      res.writeHead(200, {
        'content-type': 'audio/wav', 'content-length': size,
        'cache-control': 'public, max-age=3600', 'x-content-type-options': 'nosniff',
      });
      if (req.method === 'HEAD') res.end();
      else createReadStream(demoAudio).pipe(res);
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' }).end('Not found');
    }
    return;
  }
  const decoded = decodeURIComponent(pathname === '/' ? '/index.html' : pathname);
  const path = resolve(root, `.${decoded}`);
  if (path !== root && !path.startsWith(root + sep)) { res.writeHead(403).end('Forbidden'); return; }
  try { if (!statSync(path).isFile()) throw new Error('missing'); }
  catch { res.writeHead(404, {'content-type':'text/plain; charset=utf-8'}).end('Not found'); return; }
  res.writeHead(200, {'content-type':mime[extname(path)] || 'application/octet-stream', 'cache-control':'no-cache'});
  createReadStream(path).pipe(res);
});

server.on('upgrade', (req, client, head) => {
  const target = new URL(req.url, backend);
  const upstream = http.request(target, {method:'GET', headers:{...req.headers, host:backend.host}});
  upstream.on('upgrade', (response, socket, upstreamHead) => {
    const lines = [`HTTP/1.1 ${response.statusCode} ${response.statusMessage}`, ...Object.entries(response.headers).map(([k,v]) => `${k}: ${v}`), '', ''];
    client.write(lines.join('\r\n')); if (upstreamHead.length) client.write(upstreamHead); if (head.length) socket.write(head);
    socket.pipe(client); client.pipe(socket);
  });
  upstream.on('response', response => { client.write(`HTTP/1.1 ${response.statusCode} ${response.statusMessage}\r\n\r\n`); response.pipe(client); });
  upstream.on('error', () => { client.end('HTTP/1.1 502 Bad Gateway\r\n\r\n'); });
  upstream.end();
});

server.listen(port, '127.0.0.1', () => console.log(`声笺前端：http://127.0.0.1:${port} → ${backend.origin}`));
