// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
    const filter = document.getElementById('score-model');
    const status = document.getElementById('score-status');
    const pause = document.getElementById('score-pause');
    let live = true, busy = false, stopped = false, timer, controller;
    const number = value => new Intl.NumberFormat().format(value);
    const models = () => Array.from(filter.options, option => option.value);
    function render(id, rows, kind) {
        const table = document.getElementById(id);
        const result = rows.map((row, index) => {
            const tr = document.createElement('tr');
            const rank = document.createElement('td'); rank.textContent = String(index + 1);
            const worker = document.createElement('th'); worker.scope = 'row';
            const name = document.createElement('code'); name.textContent = row.worker; worker.append(name);
            if (kind === 'speed') { const model = document.createElement('small'); model.className = 'engine-type'; model.textContent = row.model; worker.append(model); }
            const total = document.createElement('td');
            const detail = document.createElement('td');
            if (kind === 'tokens') { total.textContent = number(row.tokens); detail.textContent = `${number(row.measured)} / ${number(row.completed)}`; }
            if (kind === 'speed') { total.textContent = row.rate.toFixed(1); detail.textContent = number(row.samples); }
            if (kind === 'requests') { total.textContent = number(row.completed); detail.textContent = `${number(row.failed)} / ${number(row.canceled)}`; }
            tr.append(rank, worker, total, detail);
            return tr;
        });
        if (!result.length) { const tr = document.createElement('tr'); const td = document.createElement('td'); td.colSpan = 4; td.textContent = kind === 'speed' ? 'No qualifying speed samples yet.' : kind === 'tokens' ? 'No reported output usage yet.' : 'No attributed API requests yet.'; tr.append(td); result.push(tr); }
        table.replaceChildren(...result);
    }
    function schedule() { clearTimeout(timer); if (live && !stopped && !document.hidden) timer = setTimeout(refresh, 3000); }
    async function refresh() {
        if (busy || stopped) return;
        busy = true;
        clearTimeout(timer);
        controller = new AbortController();
        const selected = filter.value;
        const timeout = setTimeout(() => controller.abort(), 10000);
        try {
            const query = selected ? `?${new URLSearchParams({ model: selected })}` : '';
            const response = await fetch(`/dashboard/scoreboard/status${query}`, { credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
            if (!response.ok) throw new Error('scores unavailable');
            const view = await response.json();
            if (!Array.isArray(view.tokens) || !Array.isArray(view.requests) || !Array.isArray(view.speed) || !Array.isArray(view.models)) throw new Error('invalid scoreboard');
            if (selected !== filter.value) return;
            if (JSON.stringify(models().slice(1)) !== JSON.stringify(view.models)) {
                filter.replaceChildren(new Option('All models', ''), ...view.models.map(model => new Option(model, model)));
                if (view.models.includes(selected)) filter.value = selected;
            }
            render('score-tokens', view.tokens, 'tokens'); render('score-speed', view.speed, 'speed'); render('score-requests', view.requests, 'requests');
            document.getElementById('score-warning').hidden = !view.missed;
            status.textContent = `${live ? 'Updates every 3 seconds' : 'Live scores paused'} · Updated ${new Date(view.generated).toLocaleTimeString()} · Statistics since ${new Date(view.since).toLocaleString()}`;
        } catch { if (!document.hidden) status.textContent = 'Scores unavailable — keeping the last snapshot. Check the hub and statistics storage.'; }
        finally { clearTimeout(timeout); busy = false; schedule(); if (!stopped && selected !== filter.value) refresh(); }
    }
    filter.addEventListener('change', () => { controller?.abort(); refresh(); });
    document.getElementById('score-refresh').addEventListener('click', refresh);
    pause.addEventListener('click', () => { live = !live; pause.textContent = live ? 'Pause live scores' : 'Resume live scores'; pause.setAttribute('aria-pressed', String(!live)); status.textContent = live ? 'Live score updates resumed.' : 'Live scores paused.'; schedule(); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) controller?.abort(); schedule(); });
    window.addEventListener('pagehide', () => { stopped = true; clearTimeout(timer); controller?.abort(); });
    window.addEventListener('pageshow', event => { if (event.persisted) { stopped = false; schedule(); } });
    schedule();
})();
