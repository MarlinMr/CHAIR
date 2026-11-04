// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
    const svg = document.getElementById('ambient-creature');
    if (!svg) return;
    const controls = document.getElementById('ambient-controls');
    const select = document.getElementById('ambient-mode');
    const state = document.getElementById('ambient-state');
    const branches = document.getElementById('ambient-connections');
    const main = document.querySelector('main');
    const overview = Boolean(document.getElementById('workers'));
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const key = 'chair.ambient-creature';
    const ns = 'http://www.w3.org/2000/svg';
    const connections = new Map();
    const retiring = new Map();
    let mode = 'on';
    try { const stored = localStorage.getItem(key); if (['on', 'still', 'off'].includes(stored)) mode = stored; } catch { /* Storage can be disabled. */ }
    select.value = mode;
    let live = !overview || document.body.dataset.live === 'true';
    let stale = false;
    let localActive = false;
    let reportedActive = false;
    let awake = false;
    let visible = false;
    let geometry;
    let sleepTimer;
    let pollTimer;
    let pollRequest;
    let frame;
    let stopped = false;

    function element(name, attributes = {}) {
        const node = document.createElementNS(ns, name);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
        return node;
    }
    const moving = () => mode === 'on' && !reduced.matches && live && !document.hidden && !stale && visible;

    function activity() {
        if (stopped) return;
        const working = localActive || (!stale && reportedActive);
        if (working) {
            clearTimeout(sleepTimer);
            sleepTimer = undefined;
            awake = true;
        } else if (awake && sleepTimer === undefined) {
            sleepTimer = setTimeout(() => { sleepTimer = undefined; awake = false; paint(); }, 4000);
        }
        paint();
    }

    function paint() {
        svg.toggleAttribute('hidden', !visible || mode === 'off');
        svg.classList.toggle('is-awake', awake && (!stale || localActive));
        svg.classList.toggle('is-still', mode !== 'on' || reduced.matches);
        svg.classList.toggle('is-paused', !live || document.hidden);
        svg.classList.toggle('is-stale', stale);
        state.textContent = stale ? 'Status unavailable' : !live ? 'Live refresh paused' : awake ? 'Gently gathering light' : 'Sleeping';
        if (moving() && awake) svg.unpauseAnimations(); else svg.pauseAnimations();
        for (const animation of svg.getAnimations({ subtree: true })) {
            if (!(animation instanceof CSSAnimation) && (mode !== 'on' || reduced.matches || !visible)) animation.cancel();
            else if (moving()) animation.play();
            else animation.pause();
        }
    }

    function route(connection) {
        if (!geometry) return;
        const { width, height, top, left, right } = geometry;
        const slot = connection.slot;
        const onRight = slot % 2 === 1;
        const row = Math.floor(slot / 2);
        const y = top + 350 + row * Math.min(90, (height - top - 415) / 5);
        const x = onRight ? width - 85 : 82;
        const edge = onRight ? Math.min(width - 180, right + 30 + row * 10) : Math.max(188, left - 35 - row * 10);
        const lane = top + 9 + row * .7;
        // Every route uses only the top gutter and outer margins. A clip is an
        // additional safety net while scrolling, resizing, or zooming.
        const d = onRight
            ? `M82 ${top + 82} C145 ${top + 120} 150 ${lane} 205 ${lane} H${edge - 40} Q${edge} ${lane} ${edge} ${lane + 45} C${edge + 15} ${y - 90} ${x - 80} ${y + 25} ${x} ${y}`
            : `M70 ${top + 88} C35 ${top + 155} ${edge} ${top + 115} ${edge} ${top + 205} C${edge + 20} ${y - 55} 22 ${y + 45} ${x} ${y}`;
        connection.path.setAttribute('d', d);
        connection.detail.setAttribute('d', d);
        connection.motion.setAttribute('path', d);
        connection.computer.setAttribute('transform', `translate(${x} ${y})`);
    }

    function create(id, slot) {
        const group = element('g', { class: 'ambient-connection', 'data-ambient-worker': id });
        const path = element('path', { class: 'ambient-branch', pathLength: 1 });
        const detail = element('path', { class: 'ambient-branch-detail' });
        const packet = element('circle', { class: 'ambient-packet', r: 4 });
        const motion = element('animateMotion', { dur: `${12 + slot * .7}s`, repeatCount: 'indefinite', keyPoints: '1;0', keyTimes: '0;1', calcMode: 'linear' });
        packet.append(motion);
        const computer = element('g');
        computer.append(element('rect', { class: 'ambient-computer', x: -20, y: -16, width: 40, height: 28, rx: 4 }), element('rect', { class: 'ambient-screen', x: -15, y: -11, width: 30, height: 17, rx: 2 }), element('path', { class: 'ambient-computer', d: 'M0 12 V19 M-11 20 H11' }));
        const label = element('text', { class: 'ambient-worker-label', 'text-anchor': 'middle', y: 36 });
        computer.append(label);
        group.append(path, detail, packet, computer);
        branches.append(group);
        const connection = { group, path, detail, motion, computer, label, slot };
        connections.set(id, connection);
        route(connection);
        if (moving()) {
            path.animate([{ strokeDasharray: '1', strokeDashoffset: '1' }, { strokeDasharray: '1', strokeDashoffset: '0' }], { duration: 2600, delay: slot * 110, easing: 'ease-in-out', fill: 'backwards' });
            detail.animate([{ opacity: 0 }, { opacity: .25 }], { duration: 600, delay: 2000 + slot * 110, fill: 'backwards' });
            computer.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 1300, delay: 900 + slot * 110, fill: 'backwards' });
        }
        return connection;
    }

    function remove(id, connection) {
        connections.delete(id);
        connection.group.removeAttribute('data-ambient-worker');
        if (!moving()) { connection.group.remove(); return; }
        connection.group.classList.remove('has-packet');
        connection.group.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 900 });
        connection.path.animate([{ strokeDasharray: '1', strokeDashoffset: '0' }, { strokeDasharray: '1', strokeDashoffset: '1' }], { duration: 900 });
        const timer = setTimeout(() => { connection.group.remove(); retiring.delete(timer); }, 900);
        retiring.set(timer, connection.group);
    }

    function update(view) {
        if (!Array.isArray(view.workers)) return;
        stale = false;
        const connected = view.workers.filter(worker => typeof worker.id === 'string' && worker.status !== 'Disconnected');
        // Preserve existing endpoints before filling free slots. Beyond eleven
        // individual workers, one grouped endpoint caps the drawing at twelve.
        const existing = connected.filter(worker => connections.has(worker.id));
        const selected = [...existing, ...connected.filter(worker => !connections.has(worker.id))].slice(0, 11);
        const selectedIDs = new Set(selected.map(worker => worker.id));
        const rest = connected.filter(worker => !selectedIDs.has(worker.id));
        for (const [id, connection] of connections) {
            if (id === '\0group' ? !rest.length : !selectedIDs.has(id)) remove(id, connection);
        }
        let packets = 0;
        for (const worker of [...selected, ...(rest.length ? [{ id: '\0group', inflight: rest.reduce((sum, worker) => sum + (Number(worker.inflight) || 0), 0) }] : [])]) {
            let connection = connections.get(worker.id);
            if (!connection) {
                const used = new Set(Array.from(connections.values(), item => item.slot));
                const slot = worker.id === '\0group' ? 11 : Array.from({ length: 11 }, (_, i) => i).find(i => !used.has(i));
                connection = create(worker.id, slot);
            }
            const busy = worker.inflight > 0;
            connection.group.classList.toggle('is-busy', busy);
            connection.group.classList.toggle('has-packet', busy && packets++ < 4);
            connection.label.textContent = worker.id === '\0group' ? `+ ${rest.length} workers` : worker.id.length > 18 ? `${worker.id.slice(0, 16)}…` : worker.id;
        }
        reportedActive = connected.some(worker => worker.inflight > 0);
        activity();
    }

    function layout() {
        frame = undefined;
        const rect = main.getBoundingClientRect();
        const top = document.querySelector('.appbar').offsetHeight;
        visible = rect.left >= 224 && innerWidth - rect.right >= 224 && innerHeight >= top + 690;
        controls.hidden = !visible;
        geometry = { width: innerWidth, height: innerHeight, top, left: rect.left, right: rect.right };
        svg.setAttribute('viewBox', `0 0 ${innerWidth} ${innerHeight}`);
        const protectedTop = Math.max(top, rect.top + 16);
        document.getElementById('ambient-margin-path').setAttribute('d', `M0 ${top} H${innerWidth} V${innerHeight} H0 Z M${rect.left - 12} ${protectedTop} H${rect.right + 12} V${innerHeight} H${rect.left - 12} Z`);
        document.getElementById('ambient-mascot').setAttribute('transform', `translate(82 ${top + 80})`);
        for (const connection of connections.values()) route(connection);
        paint();
        schedulePoll();
    }
    function requestLayout() { if (frame === undefined) frame = requestAnimationFrame(layout); }

    // Overview uses its existing refresh stream, never a second status poll.
    // Only the test page needs a small read-only poll, and only while visible.
    function schedulePoll() {
        clearTimeout(pollTimer);
        if (!overview && !stopped && visible && mode !== 'off' && !document.hidden && !pollRequest) pollTimer = setTimeout(poll, 3000);
    }
    async function poll() {
        if (pollRequest || stopped || !visible || mode === 'off' || document.hidden) return;
        const controller = new AbortController();
        pollRequest = controller;
        let timedOut = false;
        const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 10000);
        try {
            const response = await fetch('/dashboard/status', { credentials: 'same-origin', cache: 'no-store', signal: controller.signal });
            if (!response.ok) throw new Error('status unavailable');
            const view = await response.json();
            if (!Array.isArray(view.workers)) throw new Error('invalid status');
            if (!controller.signal.aborted) update(view);
        } catch { if (!controller.signal.aborted || timedOut) { stale = true; paint(); } }
        finally { clearTimeout(timeout); pollRequest = undefined; schedulePoll(); }
    }
    function preference(value) {
        if (!['on', 'still', 'off'].includes(value)) return;
        mode = value;
        select.value = mode;
        if (mode === 'off') pollRequest?.abort();
        paint();
        schedulePoll();
    }
    select.addEventListener('change', () => {
        preference(select.value);
        try { localStorage.setItem(key, mode); } catch { /* Preference remains active for this page. */ }
    });
    window.addEventListener('storage', event => { if (event.key === key) preference(event.newValue || 'on'); });
    reduced.addEventListener('change', paint);
    window.addEventListener('resize', requestLayout);
    window.addEventListener('scroll', requestLayout, { passive: true });
    const observer = new ResizeObserver(requestLayout);
    observer.observe(main);
    document.addEventListener('chair:snapshot', event => update(event.detail));
    document.addEventListener('chair:refresh', event => { live = event.detail.live; paint(); });
    document.addEventListener('chair:stale', () => { stale = true; paint(); });
    document.addEventListener('chair:test-activity', event => {
        localActive = event.detail.active;
        activity();
        // Give the POST time to reserve its worker, then pick up real job
        // metadata promptly. Never guess which worker handled a local prompt.
        if (localActive && !overview && !stopped && visible && mode !== 'off' && !document.hidden && !pollRequest) {
            clearTimeout(pollTimer);
            pollTimer = setTimeout(poll, 250);
        }
    });
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) pollRequest?.abort();
        paint();
        schedulePoll();
    });
    window.addEventListener('pagehide', () => {
        stopped = true;
        pollRequest?.abort();
        clearTimeout(pollTimer);
        clearTimeout(sleepTimer);
        sleepTimer = undefined;
        localActive = false;
        awake = false;
        for (const [timer, group] of retiring) { clearTimeout(timer); group.remove(); }
        retiring.clear();
        observer.disconnect();
        if (frame !== undefined) cancelAnimationFrame(frame);
        svg.pauseAnimations();
    });
    window.addEventListener('pageshow', event => { if (event.persisted) { stopped = false; observer.observe(main); layout(); } });
    layout();
    if (overview) {
        update({ workers: Array.from(document.querySelectorAll('[data-worker]'), row => ({ id: row.dataset.worker, status: row.querySelector('[data-field="status"]').textContent.trim(), inflight: Number(row.querySelector('[data-field="inflight"]').textContent) })) });
    } else if (visible && mode !== 'off') poll();
})();
