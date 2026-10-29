// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
    let live = document.body.dataset.live === 'true';
    let busy = false;
    let failed = false;
    let timer;
    let lastUpdated = document.body.dataset.updated;
    const status = document.getElementById('refresh-status');
    const toggle = document.getElementById('toggle-live');
    const workers = new Map(Array.from(document.querySelectorAll('[data-worker]'), row => [row.dataset.worker, row]));

    function text(id, value) {
        document.getElementById(id).textContent = String(value);
    }

    function message() {
        status.textContent = failed
            ? `Update failed — showing the last snapshot from ${lastUpdated}. ${live ? 'Retrying automatically.' : 'Live refresh paused.'}`
            : `${live ? 'Updates every 3 seconds without reloading' : 'Live refresh paused'} · Updated ${lastUpdated}`;
    }

    function updateWorker(worker) {
        // Worker IDs are fixed by hub configuration. Keep each existing row,
        // including its details nodes, so expansion and keyboard focus survive.
        const row = workers.get(worker.id);
        if (!row) return;
        const field = name => row.querySelector(`[data-field="${name}"]`);
        const badge = field('status').firstElementChild;
        badge.textContent = worker.status;
        badge.className = `badge ${worker.status === 'Ready' ? 'ready' : worker.status === 'Disconnected' ? 'muted' : 'warning'}`;
        field('hardware').textContent = worker.hardware;
        field('inflight').textContent = String(worker.inflight);
        field('age').textContent = worker.inventory_age;
        const details = field('models');
        details.querySelector('summary').textContent = `${worker.models.length} ${worker.models.length === 1 ? 'model' : 'models'}`;
        const list = details.querySelector('ul');
        const names = Array.from(list.querySelectorAll('code'), code => code.textContent);
        if (JSON.stringify(names) !== JSON.stringify(worker.models)) {
            list.replaceChildren(...worker.models.map(model => {
                const item = document.createElement('li');
                const code = document.createElement('code');
                code.textContent = model;
                item.append(code);
                return item;
            }));
        }
        field('no-models').hidden = worker.models.length > 0;
    }

    function update(view) {
        text('connected-count', view.connected_workers);
        text('authorized-count', view.authorized_workers);
        text('ready-count', view.ready_workers);
        text('model-count', view.available_models);
        text('job-count', view.jobs.length);
        text('job-limit', view.max_concurrent_requests);
        view.workers.forEach(updateWorker);
        const rows = view.jobs.map(job => {
            const row = document.createElement('tr');
            const values = [`#${job.id}`, job.worker_id, job.model, job.path, new Date(job.started_at).toISOString().slice(11, 19), job.elapsed];
            values.forEach((value, index) => {
                const cell = document.createElement(index === 0 ? 'th' : 'td');
                if (index === 0) cell.scope = 'row';
                cell.textContent = value;
                row.append(cell);
            });
            return row;
        });
        if (!rows.length) {
            const row = document.createElement('tr');
            const cell = document.createElement('td');
            cell.colSpan = 6;
            cell.className = 'empty';
            cell.textContent = "No jobs in flight. Send a request to the hub's API to see its worker assignment here.";
            row.append(cell);
            rows.push(row);
        }
        document.getElementById('jobs').replaceChildren(...rows);
        lastUpdated = `${new Date(view.generated_at).toISOString().slice(11, 19)} UTC`;
        failed = false;
        message();
    }

    function schedule() {
        clearTimeout(timer);
        if (live) timer = setTimeout(refresh, 3000);
    }

    async function refresh() {
        if (busy) return;
        busy = true;
        clearTimeout(timer);
        try {
            const response = await fetch('/dashboard/status', {
                credentials: 'same-origin', cache: 'no-store',
                signal: AbortSignal.timeout(10000),
            });
            if (!response.ok) throw new Error('status unavailable');
            update(await response.json());
        } catch {
            failed = true;
            message();
        } finally {
            busy = false;
            schedule();
        }
    }

    document.getElementById('refresh-now').addEventListener('click', event => {
        event.preventDefault();
        refresh();
    });
    toggle.addEventListener('click', event => {
        event.preventDefault();
        live = !live;
        toggle.textContent = live ? 'Pause live refresh' : 'Resume live refresh';
        toggle.href = `${location.pathname}${live ? '?refresh=off' : ''}`;
        history.replaceState(null, '', `${location.pathname}${live ? '' : '?refresh=off'}`);
        message();
        schedule();
    });
    const revealButton = document.getElementById('reveal-token');
    if (revealButton) {
        const panel = document.getElementById('client-token-panel');
        const value = document.getElementById('client-token-value');
        const tokenStatus = document.getElementById('token-status');
        let hideTimer;
        let revealRequest;

        function hideToken() {
            clearTimeout(hideTimer);
            if (revealRequest) revealRequest.abort();
            value.textContent = '';
            panel.hidden = true;
            revealButton.setAttribute('aria-expanded', 'false');
            tokenStatus.textContent = 'Token hidden.';
        }

        revealButton.addEventListener('click', async () => {
            hideToken();
            const controller = new AbortController();
            revealRequest = controller;
            revealButton.disabled = true;
            tokenStatus.textContent = 'Requesting token…';
            const deadline = setTimeout(() => controller.abort(), 10000);
            try {
                const response = await fetch('/dashboard/client-token', {
                    method: 'POST', credentials: 'same-origin', cache: 'no-store',
                    headers: { 'X-Chair-Token-Reveal': '1' }, signal: controller.signal,
                });
                if (!response.ok) throw new Error('token unavailable');
                const data = await response.json();
                if (controller.signal.aborted || document.hidden) return;
                if (typeof data.client_token !== 'string') throw new Error('invalid token response');
                value.textContent = data.client_token;
                panel.hidden = false;
                revealButton.setAttribute('aria-expanded', 'true');
                tokenStatus.textContent = 'Visible for one minute. Anyone with this token can run inference.';
                hideTimer = setTimeout(hideToken, 60000);
            } catch {
                value.textContent = '';
                panel.hidden = true;
                tokenStatus.textContent = 'Token unavailable. Sign in again or check the hub configuration.';
            } finally {
                clearTimeout(deadline);
                revealButton.disabled = false;
            }
        });
        document.getElementById('hide-token').addEventListener('click', hideToken);
        document.getElementById('copy-token').addEventListener('click', async () => {
            if (panel.hidden) return;
            try {
                await navigator.clipboard.writeText(value.textContent);
                tokenStatus.textContent = 'Token copied. Keep your clipboard private.';
            } catch {
                tokenStatus.textContent = 'Copy failed. Select the visible token manually.';
            }
        });
        document.addEventListener('visibilitychange', () => { if (document.hidden) hideToken(); });
        window.addEventListener('pagehide', hideToken);
    }
    schedule();
})();
