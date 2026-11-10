#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in Claude Code smoke test: real TLS hub/worker, fake engine, sandbox Read.

Requires CHAIR_TEST_BINARY, claude, Python 3, and openssl. No real model/provider
is contacted. Do not enable debug/content logging. All credentials/state are
temporary, and the only enabled client tool is Read in the temporary workspace.
"""
import http.server
import json
import os
import pathlib
import secrets
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.request


def main():
    binary = str(pathlib.Path(os.environ['CHAIR_TEST_BINARY']).resolve(strict=True))
    os.umask(0o077)
    with tempfile.TemporaryDirectory(prefix='chair-claude-') as temporary:
        directory = pathlib.Path(temporary)
        client, worker = secrets.token_hex(32), secrets.token_hex(32)
        (directory / 'client.token').write_text(client)
        (directory / 'worker.token').write_text(worker)
        (directory / 'workers.json').write_text(json.dumps({'workers': [{'id': 'test-gpu', 'token': worker}]}))
        (directory / 'sample.txt').write_text('CHAIR controlled tool-read fixture\n')
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-keyout', str(directory / 'hub.key'), '-out', str(directory / 'hub.crt'), '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1'], check=True, capture_output=True)
        results = []
        failures = []

        class Engine(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({'object': 'list', 'data': [{'id': 'chair-test', 'object': 'model'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    has_result = any(message['role'] == 'tool' for message in data['messages'])
                    if has_result and not any('CHAIR controlled tool-read fixture' in (message.get('content') or '') for message in data['messages'] if message['role'] == 'tool'):
                        raise ValueError('client did not return the sandbox Read result')
                    results.append(has_result)
                    if has_result:
                        message = {'role': 'assistant', 'content': 'CHAIR_TOOL_LOOP_OK'}
                        finish = 'stop'
                    else:
                        message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_fixture', 'type': 'function', 'function': {'name': 'Read', 'arguments': json.dumps({'file_path': str(directory / 'sample.txt')})}}]}
                        finish = 'tool_calls'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    if message.get('tool_calls'):
                        message['tool_calls'][0]['index'] = 0
                    for event in [
                        {'choices': [{'index': 0, 'delta': message, 'finish_reason': None}]},
                        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
                        {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}},
                    ]:
                        self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                        self.wfile.flush()
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    # Never expose a failed fixture's request or exception body.
                    failures.append('fake engine failed')
                self.close_connection = True

        engine = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Engine)
        threading.Thread(target=engine.serve_forever, daemon=True).start()
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        url = f'https://127.0.0.1:{port}'
        context = ssl.create_default_context(cafile=str(directory / 'hub.crt'))
        children = []
        try:
            children.append(subprocess.Popen([binary, 'hub', '--listen', f'127.0.0.1:{port}', '--tls-cert', 'hub.crt', '--tls-key', 'hub.key', '--client-token-file', 'client.token', '--workers-file', 'workers.json'], cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            children.append(subprocess.Popen([binary, 'worker', '--hub', url, '--ca-file', 'hub.crt', '--worker-id', 'test-gpu', '--worker-token-file', 'worker.token', '--engine-type', 'openai', '--engine', f'http://127.0.0.1:{engine.server_port}'], cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            for _ in range(100):
                try:
                    request = urllib.request.Request(url + '/v1/models', headers={'Authorization': 'Bearer ' + client})
                    with urllib.request.urlopen(request, context=context, timeout=2) as response:
                        if json.load(response)['data']:
                            break
                except Exception:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError('test worker did not become ready')
            env = {key: value for key, value in os.environ.items() if not key.startswith(('ANTHROPIC_', 'CLAUDE_'))}
            env.update({
                'CLAUDE_CONFIG_DIR': str(directory / 'claude-config'),
                'ANTHROPIC_BASE_URL': url,
                'ANTHROPIC_API_KEY': client,
                'ANTHROPIC_MODEL': 'chair-test',
                'ANTHROPIC_DEFAULT_HAIKU_MODEL': 'chair-test',
                'ANTHROPIC_DEFAULT_SONNET_MODEL': 'chair-test',
                'ANTHROPIC_DEFAULT_OPUS_MODEL': 'chair-test',
                'NODE_EXTRA_CA_CERTS': str(directory / 'hub.crt'),
                'MAX_THINKING_TOKENS': '0',
                'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
                'DISABLE_PROMPT_CACHING': '1',
            })
            version = subprocess.run(['claude', '--version'], check=True, capture_output=True, text=True).stdout.strip()
            result = subprocess.run(['claude', '--bare', '-p', 'Read sample.txt using Read, then report success.', '--model', 'chair-test', '--tools', 'Read', '--allowedTools', 'Read', '--no-session-persistence', '--setting-sources', '', '--settings', '{"alwaysThinkingEnabled":false}', '--output-format', 'json'], cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            try:
                output = json.loads(result.stdout)
            except json.JSONDecodeError:
                raise RuntimeError('Claude Code did not return a JSON result') from None
            if result.returncode or output.get('is_error') or output.get('result') != 'CHAIR_TOOL_LOOP_OK' or not any(results) or failures:
                raise RuntimeError('Claude Code did not complete the controlled Read tool loop')
            print(f'PASS: {version}: streamed tool use, sandbox Read, tool result, and final response through CHAIR.')
        finally:
            for child in children:
                child.terminate()
            for child in children:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            engine.shutdown()
            engine.server_close()


if __name__ == '__main__':
    main()
