// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
    const svg = document.getElementById('ambient-creature');
    if (!svg) return;
    const branches = document.getElementById('ambient-connections');
    const main = document.querySelector('main');
    const overview = Boolean(document.getElementById('workers'));
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const ns = 'http://www.w3.org/2000/svg';
    const connections = new Map();
    const retiring = new Map();
    let serial = 0;
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
    const moving = () => !reduced.matches && live && !document.hidden && !stale && visible;

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
        svg.toggleAttribute('hidden', !visible);
        if (visible) {
            for (const connection of connections.values()) {
                connection.label.removeAttribute('textLength');
                if (connection.label.getComputedTextLength() > 128) {
                    connection.label.setAttribute('textLength', '128');
                    connection.label.setAttribute('lengthAdjust', 'spacingAndGlyphs');
                }
            }
        }
        svg.classList.toggle('is-awake', awake && (!stale || localActive));
        svg.classList.toggle('is-still', reduced.matches);
        svg.classList.toggle('is-paused', !live || document.hidden);
        svg.classList.toggle('is-stale', stale);
        svg.dataset.activity = stale ? 'unavailable' : !live ? 'paused' : awake ? 'awake' : 'sleeping';
        if (moving() && awake) svg.unpauseAnimations(); else svg.pauseAnimations();
        for (const animation of svg.getAnimations({ subtree: true })) {
            if (!(animation instanceof CSSAnimation) && (reduced.matches || !visible)) animation.cancel();
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
        const side = onRight ? 1 : -1;
        const curl = `C${x + side * 11} ${y - 5} ${x + side * 28} ${y + 19} ${x + side * 51} ${y + 8} C${x + side * 73} ${y - 3} ${x + side * 58} ${y - 24} ${x + side * 43} ${y - 16} C${x + side * 32} ${y - 9} ${x + side * 39} ${y} ${x + side * 47} ${y - 5}`;
        const d = onRight
            ? `M82 ${top + 82} C145 ${top + 120} 150 ${lane} 205 ${lane} H${edge - 40} Q${edge} ${lane} ${edge} ${lane + 45} C${edge + 15} ${y - 100} ${x + 75} ${y - 48} ${x + 48} ${y - 22} ${curl}`
            : `M70 ${top + 88} C35 ${top + 155} ${edge} ${top + 115} ${edge} ${top + 205} C${edge + 20} ${y - 70} ${x - 75} ${y - 48} ${x - 48} ${y - 22} ${curl}`;
        connection.mask.setAttribute('width', width);
        connection.mask.setAttribute('height', height);
        if (connection.spine.getAttribute('d') === d) return;
        connection.spine.setAttribute('d', d);
        connection.reveal.setAttribute('d', d);
        connection.rim.setAttribute('d', d);
        for (const motion of connection.motions) motion.setAttribute('path', d);
        connection.computer.setAttribute('transform', `translate(${x} ${y})`);
        connection.labelPosition = { x: x - 72, y: y + 22 };
        const length = connection.spine.getTotalLength();
        const count = Math.min(200, Math.max(48, Math.ceil(length / 18)));
        const upper = [], lower = [], cups = [];
        function sample(distance) {
            const point = connection.spine.getPointAtLength(distance);
            const before = connection.spine.getPointAtLength(Math.max(0, distance - 1));
            const after = connection.spine.getPointAtLength(Math.min(length, distance + 1));
            const angle = Math.atan2(after.y - before.y, after.x - before.x);
            return { point, angle, nx: -Math.sin(angle), ny: Math.cos(angle), radius: .8 + 5.8 * Math.pow(1 - distance / length, .6) };
        }
        const distances = new Set(Array.from({ length: count + 1 }, (_, i) => length * i / count));
        // Keep tight terminal curls round without oversampling the long,
        // nearly straight portions of an ultrawide tentacle.
        for (let distance = Math.max(0, length - 160); distance < length; distance += 2) distances.add(distance);
        for (const distance of Array.from(distances).sort((a, b) => a - b)) {
            const { point, nx, ny, radius } = sample(distance);
            upper.push(`${point.x + nx * radius},${point.y + ny * radius}`);
            lower.unshift(`${point.x - nx * radius},${point.y - ny * radius}`);
        }
        connection.path.setAttribute('d', `M${upper.join(' L')} L${lower.join(' L')} Z`);
        const spacing = Math.max(18, length / 48);
        for (let distance = 20; distance < length - 8; distance += spacing) {
            const { point, angle, nx, ny, radius } = sample(distance);
            cups.push(element('ellipse', { class: 'ambient-sucker', cx: 0, cy: 0, rx: Math.min(2.5, radius * .65), ry: Math.min(1.6, radius * .45), transform: `translate(${point.x + nx * radius * .48} ${point.y + ny * radius * .48}) rotate(${angle * 180 / Math.PI})` }));
        }
        connection.cups.replaceChildren(...cups);
    }

    function protectNames() {
        const positions = Array.from(connections.values(), item => item.labelPosition).filter(Boolean);
        for (const connection of connections.values()) connection.nameHoles.replaceChildren(...positions.map(position => element('rect', { class: 'ambient-name-hole', x: position.x, y: position.y, width: 144, height: 22, rx: 4 })));
    }

    function create(id, slot) {
        const group = element('g', { class: 'ambient-connection', 'data-ambient-worker': id });
        const maskID = `ambient-growth-${serial++}`;
        const mask = element('mask', { id: maskID, maskUnits: 'userSpaceOnUse', x: 0, y: 0 });
        const reveal = element('path', { class: 'ambient-reveal', pathLength: 1 });
        const nameHoles = element('g');
        mask.append(reveal, nameHoles);
        const spine = element('path', { class: 'ambient-spine' });
        const skin = element('g', { mask: `url(#${maskID})` });
        const path = element('path', { class: 'ambient-branch' });
        const rim = element('path', { class: 'ambient-branch-rim' });
        const cups = element('g');
        skin.append(path, rim, cups);
        const lights = element('g', { mask: `url(#${maskID})` });
        const motions = [];
        const duration = 8 + slot * .25;
        for (let i = 0; i < 3; i++) {
            const packet = element('circle', { class: 'ambient-packet', r: i === 1 ? 3.5 : 3 });
            const motion = element('animateMotion', { dur: `${duration}s`, begin: `${-i * duration / 3}s`, repeatCount: 'indefinite', keyPoints: '1;0', keyTimes: '0;1', calcMode: 'linear' });
            packet.append(motion);
            lights.append(packet);
            motions.push(motion);
        }
        const computer = element('g');
        computer.append(element('rect', { class: 'ambient-computer', x: -20, y: -16, width: 40, height: 28, rx: 4 }), element('rect', { class: 'ambient-screen', x: -15, y: -11, width: 30, height: 17, rx: 2 }), element('path', { class: 'ambient-computer', d: 'M0 12 V19 M-11 20 H11' }));
        const label = element('text', { class: 'ambient-worker-label', 'text-anchor': 'middle', y: 36 });
        computer.append(label);
        group.append(mask, spine, skin, lights, computer);
        branches.append(group);
        const connection = { group, path, spine, reveal, rim, cups, mask, nameHoles, motions, computer, label, slot };
        connections.set(id, connection);
        route(connection);
        if (moving()) {
            reveal.animate([{ strokeDasharray: '1', strokeDashoffset: '1' }, { strokeDasharray: '1', strokeDashoffset: '0' }], { duration: 2600, delay: slot * 110, easing: 'ease-in-out', fill: 'backwards' });
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
        connection.reveal.animate([{ strokeDasharray: '1', strokeDashoffset: '0' }, { strokeDasharray: '1', strokeDashoffset: '1' }], { duration: 900 });
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
            connection.group.classList.toggle('has-packet', busy && packets++ < 6);
            connection.label.textContent = worker.id === '\0group' ? `+ ${rest.length} workers` : worker.id.length > 18 ? `${worker.id.slice(0, 16)}…` : worker.id;
        }
        reportedActive = connected.some(worker => worker.inflight > 0);
        protectNames();
        activity();
    }

    function layout() {
        frame = undefined;
        const rect = main.getBoundingClientRect();
        const top = document.querySelector('.appbar').offsetHeight;
        visible = rect.left >= 224 && innerWidth - rect.right >= 224 && innerHeight >= top + 690;
        geometry = { width: innerWidth, height: innerHeight, top, left: rect.left, right: rect.right };
        svg.setAttribute('viewBox', `0 0 ${innerWidth} ${innerHeight}`);
        const protectedTop = Math.max(top, rect.top + 16);
        document.getElementById('ambient-margin-path').setAttribute('d', `M0 ${top} H${innerWidth} V${innerHeight} H0 Z M${rect.left - 12} ${protectedTop} H${rect.right + 12} V${innerHeight} H${rect.left - 12} Z`);
        document.getElementById('ambient-mascot').setAttribute('transform', `translate(82 ${top + 80})`);
        for (const connection of connections.values()) route(connection);
        protectNames();
        paint();
        schedulePoll();
    }
    function requestLayout() { if (frame === undefined) frame = requestAnimationFrame(layout); }

    // Overview uses its existing refresh stream, never a second status poll.
    // Only the test page needs a small read-only poll, and only while visible.
    function schedulePoll() {
        clearTimeout(pollTimer);
        if (!overview && !stopped && visible && !document.hidden && !pollRequest) pollTimer = setTimeout(poll, 3000);
    }
    async function poll() {
        if (pollRequest || stopped || !visible || document.hidden) return;
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
        if (localActive && !overview && !stopped && visible && !document.hidden && !pollRequest) {
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
    } else if (visible) poll();
})();
