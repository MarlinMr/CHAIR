// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(() => {
    const form = document.getElementById('test-form');
    const model = document.getElementById('test-model');
    const prompt = document.getElementById('test-prompt');
    const send = document.getElementById('test-send');
    const cancel = document.getElementById('test-cancel');
    const refresh = document.getElementById('test-model-refresh');
    const clear = document.getElementById('test-clear');
    const status = document.getElementById('test-status');
    const output = document.getElementById('test-output');
    let active;
    let loading = false;
    const updateControls = () => {
        model.disabled = Boolean(active) || loading || !model.value;
        prompt.disabled = Boolean(active);
        send.disabled = Boolean(active) || loading || !model.value;
        cancel.disabled = !active;
        refresh.disabled = Boolean(active) || loading;
        clear.disabled = Boolean(active);
    };
    async function loadModels() {
        if (active || loading) return;
        loading = true;
        updateControls();
        const previous = model.value;
        status.textContent = 'Loading available models…';
        try {
            const response = await fetch('/dashboard/test/models', { credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(10000) });
            if (!response.ok) throw new Error('models unavailable');
            const inventory = await response.json();
            if (!Array.isArray(inventory.data)) throw new Error('invalid inventory');
            model.replaceChildren();
            for (const item of inventory.data) {
                if (typeof item.id !== 'string') continue;
                model.add(new Option(item.id, item.id));
            }
            if (Array.from(model.options).some(option => option.value === previous)) model.value = previous;
            status.textContent = model.value ? 'Ready. Enter a short prompt and select Send prompt.' : 'No models are available. Connect an engine, then refresh models.';
        } catch {
            model.replaceChildren(new Option('Models unavailable', ''));
            status.textContent = 'Cannot load models. Check your dashboard login and worker connections, then refresh models.';
        } finally {
            loading = false;
            updateControls();
        }
    }
    refresh.addEventListener('click', loadModels);
    cancel.addEventListener('click', () => active?.abort());
    clear.addEventListener('click', () => {
        prompt.value = '';
        output.textContent = '';
        status.textContent = 'Cleared.';
        prompt.focus();
    });
    window.addEventListener('pagehide', () => { active?.abort(); prompt.value = ''; output.textContent = ''; });
    form.addEventListener('submit', async event => {
        event.preventDefault();
        if (active || !form.reportValidity() || !prompt.value.trim()) return;
        const controller = new AbortController();
        active = controller;
        document.dispatchEvent(new CustomEvent('chair:test-activity', { detail: { active: true } }));
        updateControls();
        output.textContent = '';
        status.textContent = 'Sending…';
        let timedOut = false;
        const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 65000);
        let reader;
        try {
            const response = await fetch('/dashboard/test/run', {
                method: 'POST', credentials: 'same-origin', cache: 'no-store', redirect: 'error', signal: controller.signal,
                headers: { 'Content-Type': 'application/json', 'X-Chair-Test': '1' },
                body: JSON.stringify({ model: model.value, prompt: prompt.value })
            });
            if (!response.ok) {
                status.textContent = `Test failed (HTTP ${response.status}). Check login, model availability, engine support, and hub capacity. No request was retried.`;
                return;
            }
            if (!response.headers.get('Content-Type')?.startsWith('text/event-stream') || !response.body) throw new Error('unexpected response');
            reader = response.body.getReader();
            const decoder = new TextDecoder();
            let pending = '';
            let done = false;
            status.textContent = 'Generating…';
            while (!done) {
                const chunk = await reader.read();
                if (chunk.done) break;
                pending += decoder.decode(chunk.value, { stream: true });
                if (pending.length > 131072) throw new Error('event too large');
                let boundary;
                while ((boundary = /\r?\n\r?\n/.exec(pending))) {
                    const packet = pending.slice(0, boundary.index);
                    pending = pending.slice(boundary.index + boundary[0].length);
                    const data = packet.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
                    if (!data) continue;
                    if (data === '[DONE]') { done = true; break; }
                    const event = JSON.parse(data);
                    if (event.error) throw new Error('engine stream error');
                    const content = event.choices?.[0]?.delta?.content;
                    if (typeof content === 'string') {
                        const remaining = 65536 - output.textContent.length;
                        output.textContent += content.slice(0, remaining);
                        if (content.length > remaining) throw new Error('output limit');
                    }
                }
            }
            status.textContent = done ? (output.textContent ? 'Completed.' : 'Completed without text. Check the engine/model settings.') : 'Stream ended before completion. Partial output may be shown; no request was retried.';
        } catch {
            status.textContent = controller.signal.aborted ? (timedOut ? 'Test timed out. Partial output may be shown.' : 'Canceled. Partial output may be shown.') : 'Test failed or the stream was interrupted. Partial output may be shown; no request was retried.';
        } finally {
            clearTimeout(timeout);
            if (reader) { try { await reader.cancel(); } catch { /* Request may already be canceled. */ } }
            controller.abort();
            active = undefined;
            document.dispatchEvent(new CustomEvent('chair:test-activity', { detail: { active: false } }));
            updateControls();
        }
    });
    loadModels();
})();
