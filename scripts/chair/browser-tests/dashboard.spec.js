// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { test, expect, request } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:http';
import { once } from 'node:events';

const password = 'dashboard-browser-test-credential-not-for-real-use';
const clientToken = 'client-browser-test-credential-not-for-real-use';
const workerToken = 'worker-browser-test-credential-not-for-real-use';
const enrollmentToken = 'enrollment-browser-test-credential-not-for-real-use';
test.use({ ignoreHTTPSErrors: true, httpCredentials: { username: 'chair', password } });
let directory, engine, hubProcess, workerProcess, url, api;

async function listen(server) {
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    return server.address().port;
}

test.beforeAll(async () => {
    test.setTimeout(60000);
    directory = await mkdtemp(join(process.env.TMPDIR || tmpdir(), 'chair-browser-'));
    const binary = process.env.CHAIR_TEST_BINARY;
    if (!binary) throw new Error('Set CHAIR_TEST_BINARY to an absolute path to the built chair executable.');
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1', '-keyout', join(directory, 'hub.key'), '-out', join(directory, 'hub.crt'), '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1'], { stdio: 'ignore' });
    for (const [name, content] of Object.entries({ 'client.token': clientToken, 'worker.token': workerToken, 'dashboard.token': password, 'enrollment.token': enrollmentToken, 'workers.json': JSON.stringify({ workers: [{ id: 'gpu-test', token: workerToken }] }) })) {
        await writeFile(join(directory, name), content, { mode: 0o600 });
    }
    await mkdir(join(directory, 'downloads'), { mode: 0o700 });
    await writeFile(join(directory, 'downloads', 'chair-linux-amd64.tar.gz'), 'browser-test archive', { mode: 0o600 });
    await writeFile(join(directory, 'downloads', 'SHA256SUMS'), 'browser-test checksums', { mode: 0o600 });
    engine = createServer((req, res) => {
        if (req.url === '/v1/chat/completions') {
            let body = '';
            req.on('data', chunk => { body += chunk; });
            req.on('end', () => {
                const input = JSON.parse(body);
                res.setHeader('Content-Type', 'text/event-stream');
                res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: 'First chunk <script>not executable</script>' } }] })}\n\n`);
                const timer = setTimeout(() => res.end('data: {"choices":[{"delta":{"content":" final chunk"}}]}\n\ndata: [DONE]\n\n'), input.messages[0].content === 'browser-cancel-test' ? 30000 : 1000);
                res.on('close', () => clearTimeout(timer));
            });
            return;
        }
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(req.url === '/api/tags' ? { models: [{ name: 'test-model:latest' }] } : {}));
    });
    const enginePort = await listen(engine);
    const reservation = createServer();
    const port = await listen(reservation);
    await new Promise(done => reservation.close(done));
    url = `https://127.0.0.1:${port}`;
    const launch = args => spawn(resolve(binary), args, { cwd: directory, stdio: 'ignore' });
    hubProcess = launch(['hub', '--listen', `127.0.0.1:${port}`, '--tls-cert', 'hub.crt', '--tls-key', 'hub.key', '--client-token-file', 'client.token', '--workers-file', 'workers.json', '--dashboard-token-file', 'dashboard.token', '--dashboard-public', '--allow-client-token-reveal', '--allow-inference-test', '--downloads-dir', 'downloads', '--enrollment-token-file', 'enrollment.token']);
    api = await request.newContext({ ignoreHTTPSErrors: true, httpCredentials: { username: 'chair', password } });
    await expect.poll(async () => {
        try { return (await api.get(url)).status(); } catch { return 0; }
    }, { timeout: 10000 }).toBe(200);
    workerProcess = launch(['worker', '--hub', url, '--ca-file', 'hub.crt', '--worker-id', 'gpu-test', '--worker-token-file', 'worker.token', '--engine', `http://127.0.0.1:${enginePort}`]);
    await expect.poll(async () => (await (await api.get(`${url}/dashboard/status`)).json()).ready_workers, { timeout: 15000 }).toBe(1);
});

test.afterAll(async () => {
    for (const child of [workerProcess, hubProcess]) {
        if (child && child.exitCode === null && child.signalCode === null) {
            const stopped = once(child, 'exit');
            child.kill('SIGKILL');
            await stopped;
        }
    }
    if (api) await api.dispose();
    if (engine) await new Promise(done => engine.close(done));
    if (directory) await rm(directory, { recursive: true, force: true });
});

test('left-margin creature preserves layout and shares its preference across pages', async ({ page }) => {
    await page.setViewportSize({ width: 3440, height: 1440 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.goto(`${url}/?refresh=off`);
    const art = page.locator('#ambient-creature');
    await expect(art).toBeVisible();
    const original = await page.locator('main').boundingBox();
    const mascot = await page.locator('#ambient-mascot').boundingBox();
    const logo = await page.locator('.brand').boundingBox();
    expect(mascot.x).toBeLessThan(original.x);
    expect(mascot.y).toBeGreaterThan(logo.y + logo.height);
    await expect(page.locator('[data-ambient-worker="gpu-test"]')).toHaveCount(1);
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('off');
    await expect(art).toBeHidden();
    expect(await page.locator('main').boundingBox()).toEqual(original);
    await page.getByRole('link', { name: 'Test inference', exact: true }).click();
    await expect(page.getByLabel('Ambient creature', { exact: true })).toHaveValue('off');
    await expect(art).toBeHidden();
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('still');
    await expect(art).toBeVisible();
    await expect(art).toHaveClass(/is-still/);
    await page.setViewportSize({ width: 390, height: 844 });
    await expect(art).toBeHidden();
    await expect(page.locator('#ambient-controls')).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    let narrowPolls = 0;
    await page.route('**/dashboard/status', async route => { narrowPolls++; await route.continue(); });
    await page.clock.install();
    await page.clock.fastForward(15000);
    expect(narrowPolls).toBe(0);
    expect(errors).toEqual([]);
});

test('creature preferences work even when browser storage is blocked', async ({ page }) => {
    await page.addInitScript(() => Object.defineProperty(window, 'localStorage', { get() { throw new Error('storage blocked'); } }));
    await page.setViewportSize({ width: 2560, height: 1440 });
    await page.goto(`${url}/?refresh=off`);
    await expect(page.locator('#ambient-creature')).toBeVisible();
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('off');
    await expect(page.locator('#ambient-creature')).toBeHidden();
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('still');
    await expect(page.locator('#ambient-creature')).toBeVisible();
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-still/);
});

test('tentacles keep worker identities, wake for jobs, and retract on disconnect', async ({ page }) => {
    await page.setViewportSize({ width: 3440, height: 1440 });
    const view = await (await api.get(`${url}/dashboard/status`)).json();
    const first = view.workers.find(worker => worker.id === 'gpu-test');
    view.workers = [first, ...['gpu-fern', 'gpu-moss', 'gpu-north', 'gpu-night', 'gpu-reef'].map(id => ({ ...first, id, inflight: 0 }))];
    let requests = 0;
    await page.route('**/dashboard/status', route => { requests++; return route.fulfill({ json: view }); });
    await page.goto(url);
    await page.getByRole('link', { name: 'Refresh overview', exact: true }).click();
    await expect(page.locator('[data-ambient-worker]')).toHaveCount(6);
    expect(await page.locator('[data-ambient-worker="gpu-fern"] .ambient-branch').evaluate(node => node.getAnimations().some(animation => animation.effect.getTiming().duration === 2600))).toBe(true);
    await page.locator('[data-ambient-worker="gpu-test"]').evaluate(node => { node.dataset.identity = 'original'; });
    const path = await page.locator('[data-ambient-worker="gpu-test"] .ambient-branch').getAttribute('d');
    first.inflight = 1;
    view.workers[2].inflight = 1;
    await page.getByRole('link', { name: 'Refresh overview', exact: true }).click();
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-awake/);
    await expect(page.locator('[data-ambient-worker="gpu-test"]')).toHaveAttribute('data-identity', 'original');
    expect(await page.locator('[data-ambient-worker="gpu-test"] .ambient-branch').getAttribute('d')).toBe(path);
    await expect(page.locator('.has-packet')).toHaveCount(2);
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('still');
    await page.screenshot({ path: test.info().outputPath('creature-ultrawide-light.png'), fullPage: true });
    await page.screenshot({ path: test.info().outputPath('creature-closeup.png'), clip: { x: 0, y: 65, width: 240, height: 280 } });
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.screenshot({ path: test.info().outputPath('creature-ultrawide-dark.png'), fullPage: true });
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('on');
    await page.clock.install();
    first.inflight = 0;
    view.workers[2].inflight = 0;
    view.workers[1].status = 'Disconnected';
    await page.getByRole('link', { name: 'Refresh overview', exact: true }).click();
    await expect(page.locator('[data-ambient-worker]')).toHaveCount(5);
    await page.clock.fastForward(2500);
    await page.getByRole('link', { name: 'Refresh overview', exact: true }).click();
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-awake/);
    await page.clock.fastForward(1800);
    await expect(page.locator('#ambient-creature')).not.toHaveClass(/is-awake/);
    await page.getByRole('link', { name: 'Pause live refresh', exact: true }).click();
    const pausedRequests = requests;
    await page.clock.fastForward(10000);
    expect(requests).toBe(pausedRequests);
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-paused/);
});

test('ambient connections are bounded, labels are safe, and reduced motion is static', async ({ page }) => {
    await page.setViewportSize({ width: 3440, height: 1440 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    const view = await (await api.get(`${url}/dashboard/status`)).json();
    const worker = view.workers[0];
    view.workers = Array.from({ length: 80 }, (_, i) => ({ ...worker, id: `gpu-${i}`, status: 'Ready', inflight: 1 }));
    view.workers[0].id = '<script>unsafe</script>';
    await page.route('**/dashboard/status', route => route.fulfill({ json: view }));
    await page.goto(url);
    await page.getByRole('link', { name: 'Refresh overview', exact: true }).click();
    await expect(page.locator('[data-ambient-worker]')).toHaveCount(12);
    await expect(page.locator('.ambient-worker-label').last()).toHaveText('+ 69 workers');
    await expect(page.locator('.ambient-worker-label').first()).toContainText('<script>');
    await expect(page.locator('#ambient-connections script')).toHaveCount(0);
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-still/);
    expect(await page.locator('#ambient-creature').evaluate(node => node.animationsPaused())).toBe(true);
    expect(await page.locator('#ambient-creature').evaluate(node => node.getAnimations({ subtree: true }).filter(animation => animation.playState === 'running').length)).toBe(0);
    expect(await page.locator('.has-packet').count()).toBeLessThanOrEqual(4);
});

test('test page wakes the creature immediately and stops ambient polling when off', async ({ page }) => {
    await page.setViewportSize({ width: 2560, height: 1440 });
    await page.goto(`${url}/dashboard/test`);
    await expect(page.locator('[data-ambient-worker="gpu-test"]')).toHaveCount(1);
    await expect(page.getByLabel('Model', { exact: true })).toBeEnabled();
    await page.getByLabel('Prompt', { exact: true }).fill('browser-cancel-test');
    await page.getByRole('button', { name: 'Send prompt', exact: true }).click();
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-awake/);
    await expect(page.locator('#test-output')).toContainText('First chunk');
    await expect(page.locator('.ambient-connection.is-busy')).toHaveCount(1);
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('still');
    await page.screenshot({ path: test.info().outputPath('creature-test-page.png'), fullPage: true });
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Canceled.');
    await page.getByLabel('Ambient creature', { exact: true }).selectOption('off');
    let polls = 0;
    await page.route('**/dashboard/status', async route => { polls++; await route.continue(); });
    await page.clock.install();
    await page.clock.fastForward(15000);
    expect(polls).toBe(0);
    await expect(page.getByRole('button', { name: 'Send prompt', exact: true })).toBeEnabled();
});

test('stale status suppresses lights and test-page visibility suspends its polling', async ({ page }) => {
    await page.setViewportSize({ width: 2560, height: 1440 });
    const view = await (await api.get(`${url}/dashboard/status`)).json();
    view.workers[0].inflight = 1;
    let failed = false;
    let polls = 0;
    await page.route('**/dashboard/status', route => {
        polls++;
        return failed ? route.fulfill({ status: 503 }) : route.fulfill({ json: view });
    });
    await page.goto(`${url}/dashboard/test`);
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-awake/);
    failed = true;
    await page.clock.install();
    await page.clock.fastForward(3100);
    await expect(page.locator('#ambient-state')).toHaveText('Status unavailable');
    await expect(page.locator('#ambient-creature')).toHaveClass(/is-stale/);
    expect(await page.locator('#ambient-creature').evaluate(node => node.animationsPaused())).toBe(true);
    await page.evaluate(() => {
        Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
        document.dispatchEvent(new Event('visibilitychange'));
    });
    const paused = polls;
    await page.clock.fastForward(15000);
    expect(polls).toBe(paused);
    failed = false;
    await page.evaluate(() => {
        delete document.hidden;
        document.dispatchEvent(new Event('visibilitychange'));
    });
    await page.clock.fastForward(3100);
    await expect(page.locator('#ambient-creature')).not.toHaveClass(/is-stale/);
});

test('inference test streams safe text and clears without exposing credentials', async ({ page }) => {
    await page.goto(url);
    await page.getByRole('link', { name: 'Test inference', exact: true }).click();
    await expect(page).toHaveURL(`${url}/dashboard/test`);
    await expect(page.getByLabel('Model', { exact: true })).toBeEnabled();
    await page.getByLabel('Prompt', { exact: true }).fill('browser-completion-test');
    const requestPromise = page.waitForRequest(req => req.url().endsWith('/dashboard/test/run'));
    await page.getByRole('button', { name: 'Send prompt', exact: true }).click();
    const sent = await requestPromise;
    expect(sent.postDataJSON()).toEqual({ model: 'test-model:latest', prompt: 'browser-completion-test' });
    await expect(page.locator('#test-output')).toContainText('First chunk');
    await expect(page.getByRole('status')).toHaveText('Generating…');
    await expect(page.getByRole('button', { name: 'Send prompt' })).toBeDisabled();
    await expect(page.getByRole('status')).toHaveText('Completed.');
    await expect(page.locator('#test-output')).toContainText('final chunk');
    await expect(page.locator('#test-output script')).toHaveCount(0);
    for (const token of [clientToken, workerToken, password, enrollmentToken]) expect(await page.content()).not.toContain(token);
    await page.screenshot({ path: test.info().outputPath('test-inference-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: test.info().outputPath('test-inference-mobile.png'), fullPage: true });
    await page.getByRole('button', { name: 'Clear', exact: true }).click();
    await expect(page.locator('#test-output')).toBeEmpty();
    await expect(page.getByLabel('Prompt', { exact: true })).toHaveValue('');
});

test('canceling a browser inference test releases its active job', async ({ page }) => {
    await page.goto(`${url}/dashboard/test`);
    await expect(page.getByLabel('Model', { exact: true })).toBeEnabled();
    await page.getByLabel('Prompt', { exact: true }).fill('browser-cancel-test');
    await page.getByRole('button', { name: 'Send prompt', exact: true }).click();
    await expect(page.locator('#test-output')).toContainText('First chunk');
    expect((await (await api.get(`${url}/dashboard/status`)).json()).jobs).toHaveLength(1);
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Canceled.');
    await expect.poll(async () => (await (await api.get(`${url}/dashboard/status`)).json()).jobs.length).toBe(0);
    await expect(page.getByRole('button', { name: 'Send prompt' })).toBeEnabled();
});

test('inference test handles empty models and HTTP failures without retrying', async ({ page }) => {
    await page.route('**/dashboard/test/models', route => route.fulfill({ json: { data: [] } }));
    await page.goto(`${url}/dashboard/test`);
    await expect(page.getByRole('status')).toContainText('No models are available.');
    await expect(page.getByRole('button', { name: 'Send prompt' })).toBeDisabled();
    await page.route('**/dashboard/test/models', route => route.fulfill({ json: { data: [{ id: 'test-model:latest' }] } }));
    await page.getByRole('button', { name: 'Refresh models' }).click();
    await expect(page.getByLabel('Model', { exact: true })).toBeEnabled();
    let submissions = 0;
    await page.route('**/dashboard/test/run', route => {
        submissions++;
        return route.fulfill({ status: 503, body: 'private engine diagnostic' });
    });
    await page.getByLabel('Prompt', { exact: true }).fill('test failing request');
    await page.getByRole('button', { name: 'Send prompt', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Test failed (HTTP 503)');
    await expect(page.getByRole('button', { name: 'Send prompt' })).toBeEnabled();
    expect(await page.content()).not.toContain('private engine diagnostic');
    expect(submissions).toBe(1);
});

test('public dashboard does not grant anonymous inference-test access', async ({ browser }) => {
    const anonymous = await browser.newContext({ ignoreHTTPSErrors: true, httpCredentials: undefined });
    try {
        const page = await anonymous.newPage();
        await page.goto(url);
        await expect(page.getByRole('link', { name: 'Test inference', exact: true })).toBeVisible();
        expect((await anonymous.request.get(`${url}/dashboard/test`)).status()).toBe(401);
        expect((await anonymous.request.post(`${url}/dashboard/test/run`, { headers: { Origin: url, 'X-Chair-Test': '1' }, data: { model: 'test-model:latest', prompt: 'must not run' } })).status()).toBe(401);
    } finally { await anonymous.close(); }
});

test('live updates preserve expanded models, host details, and keyboard focus', async ({ page }) => {
    const snapshot = await (await api.get(`${url}/dashboard/status`)).json();
    snapshot.jobs = [{ id: 42, worker_id: 'gpu-test', model: 'test-model', path: '/api/chat', started_at: new Date().toISOString(), elapsed: '2s' }];
    snapshot.workers[0].models.push('model<script>alert(1)</script>');
    await page.route('**/dashboard/status', route => route.fulfill({ json: snapshot }));
    await page.goto(url);
    const row = page.locator('[data-worker="gpu-test"]');
    const models = row.locator('[data-field="models"]');
    const hardware = row.locator('.hardware');
    await models.locator('summary').click();
    await hardware.locator('summary').click();
    await models.locator('summary').focus();
    await row.evaluate(element => { element.dataset.identity = 'original'; });

    await expect(page.locator('#job-count')).toHaveText('1', { timeout: 10000 });

    await expect(models).toHaveAttribute('open', '');
    await expect(hardware).toHaveAttribute('open', '');
    await expect(row).toHaveAttribute('data-identity', 'original');
    await expect(models.locator('summary')).toBeFocused();
    await expect(models).toContainText('model<script>alert(1)</script>');
    await expect(models.locator('script')).toHaveCount(0);
    await expect(hardware).toContainText('logical CPUs');
});

test('pause stops polling and manual refresh still preserves expansion', async ({ page }) => {
    await page.goto(`${url}/?refresh=off`);
    const models = page.locator('[data-field="models"]');
    await models.locator('summary').click();
    let polls = 0;
    await page.route('**/dashboard/status', async route => { polls++; await route.continue(); });
    await page.clock.install();
    await page.clock.fastForward(10000);
    expect(polls).toBe(0);

    await page.locator('#refresh-now').click();
    await expect.poll(() => polls).toBe(1);
    await expect(models).toHaveAttribute('open', '');
    await expect(page.locator('#refresh-status')).toContainText('Live refresh paused');
    await page.locator('#toggle-live').click();
    await page.clock.fastForward(3100);
    await expect.poll(() => polls).toBe(2);
    await expect(models).toHaveAttribute('open', '');
});

test('failed updates keep the last snapshot and report stale data', async ({ page }) => {
    await page.goto(url);
    const models = page.locator('[data-field="models"]');
    await models.locator('summary').click();
    await page.route('**/dashboard/status', route => route.fulfill({ status: 503, body: 'unavailable' }));

    await page.locator('#refresh-now').click();

    await expect(page.locator('#refresh-status')).toContainText('Update failed');
    await expect(models).toHaveAttribute('open', '');
    await expect(models).toContainText('test-model:latest');
});

test('worker tables scroll within a narrow viewport', async ({ page }) => {
    await page.goto(`${url}/?refresh=off`);
    await page.locator('.hardware summary').click();
    await page.locator('[data-field="models"] summary').click();
    await page.screenshot({ path: test.info().outputPath('dashboard-desktop.png'), fullPage: true });

    await page.setViewportSize({ width: 390, height: 844 });

    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await expect(page.locator('#toggle-live')).toBeVisible();
    await page.screenshot({ path: test.info().outputPath('dashboard-mobile.png'), fullPage: true });
});

test('anonymous visitors can view setup and downloads but cannot reveal tokens', async ({ browser }) => {
    const anonymous = await browser.newContext({ ignoreHTTPSErrors: true, httpCredentials: undefined });
    try {
        const page = await anonymous.newPage();
        await page.goto(url);

        await expect(page.getByRole('link', { name: 'Sign in for token access' })).toBeVisible();
        await expect(page.locator('#reveal-token')).toHaveCount(0);
        expect(await page.content()).not.toContain(clientToken);
        const denied = await anonymous.request.post(`${url}/dashboard/client-token`, { headers: { Origin: url, 'X-Chair-Token-Reveal': '1' } });
        expect(denied.status()).toBe(401);

        const pending = page.waitForEvent('download');
        await page.getByRole('link', { name: /Linux · x64/ }).click();
        const download = await pending;
        expect(download.suggestedFilename()).toBe('chair-linux-amd64.tar.gz');
        const saved = test.info().outputPath('downloaded-archive.tar.gz');
        await download.saveAs(saved);
        expect(await readFile(saved, 'utf8')).toBe('browser-test archive');

        await page.getByRole('link', { name: /Setup guide/ }).click();
        await expect(page).toHaveURL(`${url}/dashboard/readme.html`);
        await expect(page.locator('body')).toContainText('Generate a TLS certificate for testing');
    } finally {
        await anonymous.close();
    }
});

test('setup guide formats bundled Markdown with current-hub examples and local navigation', async ({ page }) => {
    const externalRequests = [];
    page.on('request', req => { if (!req.url().startsWith(url)) externalRequests.push(req.url()); });
    await page.goto(`${url}/dashboard/readme.html`);

    await expect(page.getByRole('heading', { name: 'Connect your engines and clients.' })).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Guide contents' })).toBeVisible();
    await expect(page.locator('.page-header')).toContainText(url);
    await expect(page.locator('.guide-content pre').filter({ hasText: `--hub ${url}` }).first()).toBeAttached();
    await expect(page.locator('.guide-content pre').filter({ hasText: `${url}/v1/chat/completions` })).toHaveCount(1);
    await expect(page.locator('.guide-content pre').filter({ hasText: 'http://127.0.0.1:11434' }).first()).toBeAttached();
    expect(await page.locator('.guide-content').innerText()).not.toContain('https://hub.example.internal');
    for (const token of [clientToken, workerToken, password, enrollmentToken]) expect(await page.content()).not.toContain(token);
    const links = page.locator('.guide-toc a');
    for (const link of await links.all()) {
        const target = await link.getAttribute('href');
        expect(await page.locator(target).count()).toBe(1);
    }
    await page.screenshot({ path: test.info().outputPath('guide-desktop.png') });
    await page.getByRole('navigation', { name: 'Guide contents' }).getByRole('link', { name: 'Run', exact: true }).click();
    await expect(page).toHaveURL(`${url}/dashboard/readme.html#run`);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.goto(`${url}/dashboard/readme.html`);
    await page.screenshot({ path: test.info().outputPath('guide-mobile.png') });
    expect(externalRequests).toEqual([]);
});

test('authenticated users explicitly reveal, copy, and hide the client token', async ({ page }) => {
    let reveals = 0;
    page.on('request', req => { if (req.url().endsWith('/dashboard/client-token')) reveals++; });
    await page.goto(`${url}/dashboard/access?refresh=off`);
    await expect(page.locator('#reveal-token')).toBeVisible();
    expect(await page.content()).not.toContain(clientToken);
    expect(reveals).toBe(0);
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);

    await page.locator('#reveal-token').click();

    await expect(page.locator('#client-token-value')).toHaveText(clientToken);
    expect(reveals).toBe(1);
    await page.locator('#copy-token').click();
    await expect(page.locator('#token-status')).toContainText('Token copied');
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(clientToken);
    await page.locator('#refresh-now').click();
    await expect(page.locator('#refresh-status')).toContainText('Live refresh paused');
    expect(reveals).toBe(1);
    await page.locator('#hide-token').click();
    await expect(page.locator('#client-token-value')).toBeEmpty();
    await expect(page.locator('#client-token-panel')).toBeHidden();

    await page.clock.install();
    await page.locator('#reveal-token').click();
    await expect(page.locator('#client-token-value')).toHaveText(clientToken);
    await page.clock.fastForward(61000);
    await expect(page.locator('#client-token-value')).toBeEmpty();
    await expect(page.locator('#client-token-panel')).toBeHidden();
    const snapshot = await (await api.get(`${url}/dashboard/status`)).json();
    expect(JSON.stringify(snapshot)).not.toContain(clientToken);
});

test('enrollment returns a generated token once and adds a live dashboard row', async ({ browser }) => {
    const anonymous = await browser.newContext({ ignoreHTTPSErrors: true, httpCredentials: undefined });
    try {
        const overview = await anonymous.newPage();
        await overview.goto(`${url}/?refresh=off`);
        const models = overview.locator('[data-worker="gpu-test"] [data-field="models"]');
        await models.locator('summary').click();
        const page = await anonymous.newPage();
        let submissions = 0;
        page.on('request', req => { if (req.method() === 'POST' && req.url() === `${url}/join`) submissions++; });
        await page.goto(`${url}/join`);
        expect(await page.content()).not.toContain(enrollmentToken);
        await page.screenshot({ path: test.info().outputPath('join-desktop.png'), fullPage: true });
        await page.setViewportSize({ width: 390, height: 844 });
        await expect(page.locator('.brand')).toBeVisible();
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
        await page.screenshot({ path: test.info().outputPath('join-mobile.png'), fullPage: true });
        await page.getByLabel('Worker ID', { exact: true }).fill('browser-enrolled-gpu');
        await page.getByLabel('Engine API', { exact: true }).selectOption('openai');
        await page.getByLabel('Local engine URL', { exact: true }).fill('http://127.0.0.1:8000');
        await page.getByLabel('Shared enrollment token', { exact: true }).fill(enrollmentToken);

        await page.getByRole('button', { name: 'Register worker' }).click();

        await expect(page.locator('#enrollment-result')).toBeVisible();
        const token = await page.locator('#enrolled-worker-token').textContent();
        expect(token).toMatch(/^[0-9a-f]{64}$/);
        expect(submissions).toBe(1);
        await expect(page.locator('#enrollment-token')).toHaveValue('');
        await expect(page.locator('#enroll-worker')).toBeDisabled();
        await expect(page.locator('#worker-start-command')).toContainText('browser-enrolled-gpu.token');
        await expect(page.locator('#worker-start-command')).toContainText("--engine-type 'openai'");
        await expect(page.locator('#worker-start-command')).toContainText('http://127.0.0.1:8000');
        expect(await page.locator('#worker-start-command').textContent()).not.toContain(token);
        const saved = JSON.parse(await readFile(join(directory, 'workers.json'), 'utf8'));
        expect(saved.workers.find(worker => worker.id === 'browser-enrolled-gpu').token).toBe(token);
        await anonymous.grantPermissions(['clipboard-read', 'clipboard-write']);
        await page.getByRole('button', { name: 'Copy worker token' }).click();
        await expect(page.locator('#enrollment-status')).toContainText('Worker token copied');
        expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(token);
        await page.getByRole('button', { name: 'Hide worker token' }).click();
        await expect(page.locator('#enrolled-worker-token')).toBeEmpty();

        await overview.bringToFront();
        await overview.locator('#refresh-now').click();
        await expect(overview.locator('[data-worker="browser-enrolled-gpu"]')).toContainText('Disconnected');
        await expect(models).toHaveAttribute('open', '');
        expect(await overview.content()).not.toContain(token);

        await page.bringToFront();
        await page.reload();
        await expect(page.locator('#enrollment-result')).toBeHidden();
        expect(await page.content()).not.toContain(token);
        expect(submissions).toBe(1);
        await page.getByLabel('Worker ID', { exact: true }).fill('browser-enrolled-gpu');
        await page.getByLabel('Shared enrollment token', { exact: true }).fill(enrollmentToken);
        await page.getByRole('button', { name: 'Register worker' }).click();
        await expect(page.locator('#enrollment-status')).toContainText('already registered');
        await expect(page.locator('#enrollment-result')).toBeHidden();
        expect(await page.content()).not.toContain(token);
    } finally {
        await anonymous.close();
    }
});

test('invalid enrollment credentials do not register a worker or retry automatically', async ({ page }) => {
    const before = await readFile(join(directory, 'workers.json'), 'utf8');
    let submissions = 0;
    page.on('request', req => { if (req.method() === 'POST' && req.url() === `${url}/join`) submissions++; });
    await page.goto(`${url}/join`);
    await page.clock.install();
    await page.getByLabel('Worker ID', { exact: true }).fill('unauthorized-gpu');
    await page.getByLabel('Shared enrollment token', { exact: true }).fill(password);

    await page.getByRole('button', { name: 'Register worker' }).click();

    await expect(page.locator('#enrollment-status')).toContainText('Invalid enrollment token');
    await expect(page.locator('#enrollment-result')).toBeHidden();
    await page.clock.fastForward(20000);
    expect(submissions).toBe(1);
    expect(await readFile(join(directory, 'workers.json'), 'utf8')).toBe(before);
});
