// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
    const form = document.getElementById('enrollment-form');
    const id = document.getElementById('worker-id');
    const invitation = document.getElementById('enrollment-token');
    const engineType = document.getElementById('worker-engine-type');
    const engineURL = document.getElementById('worker-engine-url');
    const submit = document.getElementById('enroll-worker');
    const status = document.getElementById('enrollment-status');
    const result = document.getElementById('enrollment-result');
    const token = document.getElementById('enrolled-worker-token');
    let registered = false;
    let sending = false;
    let leaving = false;
    let pending;
    engineType.addEventListener('change', () => {
        if (['http://127.0.0.1:11434', 'http://127.0.0.1:8080'].includes(engineURL.value)) {
            engineURL.value = engineType.value === 'openai' ? 'http://127.0.0.1:8080' : 'http://127.0.0.1:11434';
        }
    });

    function hideToken() {
        invitation.value = '';
        token.textContent = '';
        document.getElementById('copy-worker-token').disabled = true;
        status.textContent = registered ? 'Worker token hidden. If you did not save it, contact the hub operator; do not enroll the same ID again.' : 'Enrollment token cleared.';
    }

    form.addEventListener('submit', async event => {
        event.preventDefault();
        if (registered || sending || !form.reportValidity()) return;
        sending = true;
        submit.disabled = true;
        status.textContent = 'Registering worker…';
        const body = JSON.stringify({ worker_id: id.value, enrollment_token: invitation.value });
        invitation.value = '';
        const controller = new AbortController();
        pending = controller;
        const deadline = setTimeout(() => controller.abort(), 15000);
        try {
            const response = await fetch('/join', {
                method: 'POST', credentials: 'omit', cache: 'no-store', redirect: 'error',
                headers: { 'Content-Type': 'application/json', 'X-Chair-Enroll': '1' },
                body, signal: controller.signal,
            });
            if (!response.ok) {
                const messages = {
                    400: 'Invalid worker ID or request. Use a unique name, not a token.',
                    401: 'Invalid enrollment token. Ask your hub operator for the correct token.',
                    409: 'Worker ID already registered or cluster capacity reached. Contact the hub operator.',
                    429: 'Enrollment request limit reached. Wait one minute before trying again.',
                };
                status.textContent = messages[response.status] || 'Enrollment unavailable. Contact the operator before retrying.';
                return;
            }
            const data = await response.json();
            if (typeof data.worker_id !== 'string' || typeof data.worker_token !== 'string') throw new Error('invalid enrollment response');
            registered = true;
            id.disabled = true;
            invitation.disabled = true;
            engineType.disabled = true;
            engineURL.disabled = true;
            if (leaving) { hideToken(); return; }
            token.textContent = data.worker_token;
            result.hidden = false;
            const quote = value => `'${value.replaceAll("'", "'\\''")}'`;
            document.getElementById('worker-start-command').textContent = `chair worker --hub ${quote(location.origin)} --worker-id ${quote(data.worker_id)} --worker-token-file ${quote(`${data.worker_id}.token`)} --engine-type ${quote(engineType.value)} --engine ${quote(engineURL.value)}`;
            status.textContent = 'Worker registered. Save the token now; it is shown only in this response. No request is automatically retried.';
        } catch {
            status.textContent = 'Enrollment response lost or unavailable. Registration may already have succeeded. Contact the hub operator before retrying.';
        } finally {
            clearTimeout(deadline);
            sending = false;
            submit.disabled = registered;
        }
    });
    document.getElementById('copy-worker-token').addEventListener('click', async () => {
        if (!token.textContent) return;
        try {
            await navigator.clipboard.writeText(token.textContent);
            status.textContent = 'Worker token copied. Save it privately before hiding it or leaving this page.';
        } catch {
            status.textContent = 'Copy failed. Select the visible worker token manually and save it privately.';
        }
    });
    document.getElementById('hide-worker-token').addEventListener('click', hideToken);
    window.addEventListener('pagehide', () => { leaving = true; if (pending) pending.abort(); hideToken(); });
    window.addEventListener('pageshow', () => { leaving = false; });
})();
