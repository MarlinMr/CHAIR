// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { test, expect, request } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'node:http';
import { once } from 'node:events';

const password = 'dashboard-browser-test-credential-not-for-real-use';
const clientToken = 'client-browser-test-credential-not-for-real-use';
const workerToken = 'worker-browser-test-credential-not-for-real-use';
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
    for (const [name, content] of Object.entries({ 'client.token': clientToken, 'worker.token': workerToken, 'dashboard.token': password, 'workers.json': JSON.stringify({ workers: [{ id: 'gpu-test', token: workerToken }] }) })) {
        await writeFile(join(directory, name), content, { mode: 0o600 });
    }
    engine = createServer((req, res) => {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(req.url === '/api/tags' ? { models: [{ name: 'test-model:latest' }] } : {}));
    });
    const enginePort = await listen(engine);
    const reservation = createServer();
    const port = await listen(reservation);
    await new Promise(done => reservation.close(done));
    url = `https://127.0.0.1:${port}`;
    const launch = args => spawn(resolve(binary), args, { cwd: directory, stdio: 'ignore' });
    hubProcess = launch(['hub', '--listen', `127.0.0.1:${port}`, '--tls-cert', 'hub.crt', '--tls-key', 'hub.key', '--client-token-file', 'client.token', '--workers-file', 'workers.json', '--dashboard-token-file', 'dashboard.token']);
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
