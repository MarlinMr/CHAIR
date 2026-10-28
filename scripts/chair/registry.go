// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const (
	workerIDHeader    = "X-Chair-Worker-ID"
	inventoryInterval = 2 * time.Second
	inventoryTimeout  = 3 * time.Second
	inventoryLifetime = 10 * time.Second
	maxInventoryBytes = 1 << 20
	maxModels         = 4096
)

func canonicalModel(name string) string {
	if strings.Contains(name, "@") || strings.LastIndex(name, ":") > strings.LastIndex(name, "/") {
		return name
	}
	return name + ":latest"
}

func validModel(name string) bool {
	if name == "" || len(name) > 512 {
		return false
	}
	for _, c := range name {
		if c <= ' ' || c == 127 {
			return false
		}
	}
	return true
}

// Keep Ollama's metadata intact when merging inventories. Model names are the
// capability gate; fields such as size and digest remain client-facing data.
func fetchInventory(ctx context.Context, worker *workerConnection) (map[string]json.RawMessage, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://worker/api/tags", nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{
		Transport:     worker.transport,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, errors.New("engine inventory unavailable")
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxInventoryBytes+1))
	if err != nil || len(body) > maxInventoryBytes {
		return nil, errors.New("engine inventory unreadable or oversized")
	}
	var inventory struct {
		Models *[]json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &inventory); err != nil || inventory.Models == nil || len(*inventory.Models) > maxModels {
		return nil, errors.New("invalid engine inventory")
	}
	models := make(map[string]json.RawMessage, len(*inventory.Models))
	for _, raw := range *inventory.Models {
		var model struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &model); err != nil || !validModel(model.Name) {
			return nil, errors.New("invalid engine model")
		}
		models[canonicalModel(model.Name)] = raw
	}
	return models, nil
}

func (h *hub) refreshInventory(parent context.Context, worker *workerConnection) {
	ctx, cancel := context.WithTimeout(parent, inventoryTimeout)
	defer cancel()
	models, err := fetchInventory(ctx, worker)
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed || h.workers[worker.id] != worker {
		return
	}
	// Failed or malformed reports withdraw all capabilities until a good report
	// arrives. Do not keep routing against a cached positive inventory forever.
	worker.models = models
	worker.updatedAt = time.Time{}
	if err == nil {
		worker.updatedAt = time.Now()
	}
}

func (w *workerConnection) ready(now time.Time) bool {
	return !w.session.IsClosed() && !w.updatedAt.IsZero() && now.Sub(w.updatedAt) < inventoryLifetime
}

func (h *hub) serveModels(w http.ResponseWriter) {
	h.mu.Lock()
	workers := make([]*workerConnection, 0, len(h.workers))
	if !h.closed {
		for _, worker := range h.workers {
			if worker.ready(time.Now()) {
				workers = append(workers, worker)
			}
		}
	}
	sort.Slice(workers, func(i, j int) bool { return workers[i].id < workers[j].id })
	merged := make(map[string]json.RawMessage)
	for _, worker := range workers {
		for key, model := range worker.models {
			if _, exists := merged[key]; !exists {
				merged[key] = model
			}
		}
	}
	h.mu.Unlock()
	if len(workers) == 0 {
		http.Error(w, "no worker with a fresh engine inventory", http.StatusServiceUnavailable)
		return
	}
	keys := make([]string, 0, len(merged))
	for key := range merged {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	models := make([]json.RawMessage, 0, len(keys))
	for _, key := range keys {
		models = append(models, merged[key])
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(struct {
		Models []json.RawMessage `json:"models"`
	}{models})
}

// Reserve before unlocking so concurrent bursts see work that has just been
// assigned, not only work whose response headers have already arrived.
func (h *hub) reserveWorker(model string) *workerConnection {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return nil
	}
	now := time.Now()
	var selected *workerConnection
	for _, worker := range h.workers {
		if !worker.ready(now) {
			continue
		}
		if _, owns := worker.models[canonicalModel(model)]; !owns {
			continue
		}
		if selected == nil || worker.inflight < selected.inflight ||
			(worker.inflight == selected.inflight && (worker.assigned < selected.assigned ||
				(worker.assigned == selected.assigned && worker.id < selected.id))) {
			selected = worker
		}
	}
	if selected != nil {
		h.assignment++
		selected.assigned = h.assignment
		selected.inflight++
	}
	return selected
}

func (h *hub) routeInference(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBytes))
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "cannot read request body", http.StatusBadRequest)
		}
		return
	}
	var request struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &request); err != nil || !validModel(request.Model) {
		http.Error(w, "request must be a JSON object with a nonempty model", http.StatusBadRequest)
		return
	}
	worker := h.reserveWorker(request.Model)
	if worker == nil {
		http.Error(w, "no connected worker advertises the requested model", http.StatusBadGateway)
		return
	}
	defer func() {
		h.mu.Lock()
		worker.inflight--
		h.mu.Unlock()
	}()
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.TransferEncoding = nil
	r.GetBody = nil
	// This hostname is logical. The transport opens a stream in this worker's
	// outbound connection, never a connection to an advertised worker address.
	inferenceProxy(&url.URL{Scheme: "http", Host: "worker"}, worker.transport).ServeHTTP(w, r)
}
