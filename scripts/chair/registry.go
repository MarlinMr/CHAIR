// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
	"unicode"
)

const (
	workerIDHeader    = "X-Chair-Worker-ID"
	inventoryInterval = 2 * time.Second
	inventoryTimeout  = 3 * time.Second
	inventoryLifetime = 10 * time.Second
	maxInventoryBytes = 1 << 20
	maxModels         = 4096
)

type inventoryFailure struct {
	reason string
	status int
}

func (e *inventoryFailure) Error() string { return e.reason }

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
		if unicode.IsSpace(c) || unicode.IsControl(c) {
			return false
		}
	}
	return true
}

// Keep each engine's model metadata. Only Ollama names get implicit :latest
// normalization; OpenAI-compatible IDs are exact, including vendor aliases.
func fetchInventory(ctx context.Context, worker *workerConnection) (map[string]json.RawMessage, error) {
	path := "/api/tags"
	if worker.engineType == engineOpenAI {
		path = "/v1/models"
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://worker"+path, nil)
	if err != nil {
		return nil, &inventoryFailure{reason: "inventory_request_failed"}
	}
	client := &http.Client{
		Transport:     worker.transport,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, &inventoryFailure{reason: "engine_unreachable"}
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, &inventoryFailure{reason: "inventory_http_error", status: response.StatusCode}
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxInventoryBytes+1))
	if err != nil || len(body) > maxInventoryBytes {
		return nil, &inventoryFailure{reason: "inventory_unreadable_or_oversized"}
	}
	if worker.engineType == engineOpenAI {
		return parseOpenAIInventory(body)
	}
	var inventory struct {
		Models *[]json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &inventory); err != nil || inventory.Models == nil || len(*inventory.Models) > maxModels {
		return nil, &inventoryFailure{reason: "invalid_inventory"}
	}
	models := make(map[string]json.RawMessage, len(*inventory.Models))
	for _, raw := range *inventory.Models {
		var model struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &model); err != nil || !validModel(model.Name) {
			return nil, &inventoryFailure{reason: "invalid_inventory_model"}
		}
		models[canonicalModel(model.Name)] = raw
	}
	return models, nil
}

func parseOpenAIInventory(body []byte) (map[string]json.RawMessage, error) {
	var inventory struct {
		Object string             `json:"object"`
		Data   *[]json.RawMessage `json:"data"`
	}
	if json.Unmarshal(body, &inventory) != nil || inventory.Object != "list" || inventory.Data == nil || len(*inventory.Data) > maxModels {
		return nil, &inventoryFailure{reason: "invalid_inventory"}
	}
	models := make(map[string]json.RawMessage, len(*inventory.Data))
	for _, raw := range *inventory.Data {
		var model openAIModel
		if json.Unmarshal(raw, &model) != nil || model.Object != "model" || !validModel(model.ID) {
			return nil, &inventoryFailure{reason: "invalid_inventory_model"}
		}
		if _, duplicate := models[model.ID]; duplicate {
			return nil, &inventoryFailure{reason: "invalid_inventory_model"}
		}
		models[model.ID] = raw
	}
	return models, nil
}

func (h *hub) refreshInventory(parent context.Context, worker *workerConnection) {
	ctx, cancel := context.WithTimeout(parent, inventoryTimeout)
	defer cancel()
	models, err := fetchInventory(ctx, worker)
	h.mu.Lock()
	if h.closed || h.workers[worker.id] != worker || parent.Err() != nil {
		h.mu.Unlock()
		return
	}
	wasFailed := worker.inventoryFailed
	hadInventory := !worker.updatedAt.IsZero()
	changed := !sameModelSet(worker.models, models)
	// Failed or malformed reports withdraw all capabilities until a good report
	// arrives. Do not keep routing against a cached positive inventory forever.
	worker.models = models
	worker.inventoryFailed = err != nil
	worker.updatedAt = time.Time{}
	if err == nil {
		worker.updatedAt = time.Now()
	}
	h.mu.Unlock()
	if err != nil {
		reason, status := "inventory_failed", 0
		var failure *inventoryFailure
		if errors.As(err, &failure) {
			reason, status = failure.reason, failure.status
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			reason = "inventory_timeout"
		}
		level := slog.LevelWarn
		if wasFailed {
			level = slog.LevelDebug
		}
		h.logger.LogAttrs(parent, level, "worker inventory unavailable", slog.String("worker_id", worker.id), slog.String("reason", reason), slog.Int("status", status))
	} else if wasFailed {
		h.logger.Info("worker inventory recovered", "worker_id", worker.id, "models", len(models))
	} else if !hadInventory || changed {
		h.logger.Info("worker inventory updated", "worker_id", worker.id, "models", len(models))
	} else {
		h.logger.Debug("worker inventory refreshed", "worker_id", worker.id, "models", len(models))
	}
}

func sameModelSet(left, right map[string]json.RawMessage) bool {
	if len(left) != len(right) {
		return false
	}
	for name := range left {
		if _, exists := right[name]; !exists {
			return false
		}
	}
	return true
}

func (w *workerConnection) ready(now time.Time) bool {
	return !w.session.IsClosed() && !w.updatedAt.IsZero() && now.Sub(w.updatedAt) < inventoryLifetime
}

func (h *hub) serveModels(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	workers := make([]*workerConnection, 0, len(h.workers))
	if !h.closed {
		for _, worker := range h.workers {
			if worker.ready(time.Now()) && engineSupportsPath(worker.engineType, r.URL.Path) {
				workers = append(workers, worker)
			}
		}
	}
	sort.Slice(workers, func(i, j int) bool { return workers[i].id < workers[j].id })
	merged := make(map[string]json.RawMessage)
	for _, worker := range workers {
		for key, model := range worker.models {
			if _, exists := merged[key]; !exists {
				if r.URL.Path == "/v1/models" && worker.engineType == engineOllama {
					model, _ = json.Marshal(openAIModel{ID: key, Object: "model", OwnedBy: "ollama"})
				}
				merged[key] = model
			}
		}
	}
	h.mu.Unlock()
	if len(workers) == 0 {
		writeAPIError(w, r, http.StatusServiceUnavailable, "no worker with a fresh engine inventory", "inventory_unavailable")
		return
	}
	keys := make([]string, 0, len(merged))
	for key := range merged {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if r.URL.Path == "/v1/models" {
		models := make([]json.RawMessage, 0, len(keys))
		for _, key := range keys {
			models = append(models, merged[key])
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			Object string            `json:"object"`
			Data   []json.RawMessage `json:"data"`
		}{Object: "list", Data: models})
		return
	}
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
func (h *hub) reserveWorker(model, path string) *workerConnection {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closed {
		return nil
	}
	now := time.Now()
	// An exact advertised OpenAI ID must not lose to Ollama's convenience alias.
	exact := false
	for _, worker := range h.workers {
		if worker.ready(now) && engineSupportsPath(worker.engineType, path) {
			if _, owns := worker.models[model]; owns {
				exact = true
				break
			}
		}
	}
	var selected *workerConnection
	for _, worker := range h.workers {
		if !worker.ready(now) || !engineSupportsPath(worker.engineType, path) {
			continue
		}
		key := model
		if worker.engineType == engineOllama && !exact {
			key = canonicalModel(model)
		}
		if _, owns := worker.models[key]; !owns {
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
			writeAPIError(w, r, http.StatusRequestEntityTooLarge, "request body too large", "request_too_large")
		} else {
			writeAPIError(w, r, http.StatusBadRequest, "cannot read request body", "invalid_request")
		}
		return
	}
	var request struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &request); err != nil || !validModel(request.Model) {
		writeAPIError(w, r, http.StatusBadRequest, "request must be a JSON object with a nonempty model", "invalid_request")
		return
	}
	worker := h.reserveWorker(request.Model, r.URL.Path)
	requestID := h.requestID.Add(1)
	logger := h.logger.With("request_id", requestID, "model", request.Model, "method", r.Method, "path", r.URL.Path)
	if worker == nil {
		logger.Warn("inference rejected", "reason", "no_eligible_worker")
		writeAPIError(w, r, http.StatusBadGateway, "no connected worker advertises the requested model", "model_unavailable")
		return
	}
	started := time.Now()
	h.mu.Lock()
	h.jobs[requestID] = activeJob{ID: requestID, WorkerID: worker.id, Model: request.Model, Path: r.URL.Path, StartedAt: started.UTC()}
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		worker.inflight--
		delete(h.jobs, requestID)
		h.mu.Unlock()
	}()
	logger = logger.With("worker_id", worker.id)
	logger.Debug("inference dispatched")
	logged := &requestLogWriter{ResponseWriter: w}
	finished := false
	defer func() {
		outcome := "completed"
		switch {
		case r.Context().Err() != nil:
			outcome = "canceled"
		case !finished:
			outcome = "interrupted"
			logger.Warn("inference interrupted", "reason", "stream_interrupted")
		case logged.status >= http.StatusBadRequest:
			outcome = "failed"
		}
		logger.Debug("inference finished", "outcome", outcome, "status", logged.status, "duration_ms", time.Since(started).Milliseconds(), "response_bytes", logged.bytes)
	}()
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.TransferEncoding = nil
	r.GetBody = nil
	// This hostname is logical. The transport opens a stream in this worker's
	// outbound connection, never a connection to an advertised worker address.
	inferenceProxy(&url.URL{Scheme: "http", Host: "worker"}, worker.transport, logger).ServeHTTP(logged, r)
	finished = true
}
