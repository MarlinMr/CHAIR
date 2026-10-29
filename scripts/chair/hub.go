// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/hashicorp/yamux"
)

type workerConnection struct {
	id              string
	session         *yamux.Session
	transport       *http.Transport
	models          map[string]json.RawMessage
	updatedAt       time.Time
	inflight        int
	assigned        uint64
	inventoryFailed bool
	hardware        workerHardware
}

type hub struct {
	clientToken            string
	workerTokens           map[string]string
	mu                     sync.Mutex
	workers                map[string]*workerConnection
	assignment             uint64
	closed                 bool
	requests               chan struct{}
	logger                 *slog.Logger
	requestID              atomic.Uint64
	jobs                   map[uint64]activeJob
	dashboardToken         string
	dashboardPublic        bool
	allowClientTokenReveal bool
	downloadsRoot          *os.Root
}

func newHub(clientToken string, workerTokens map[string]string, maxConcurrentRequests int) *hub {
	return &hub{
		clientToken:  clientToken,
		workerTokens: workerTokens,
		workers:      make(map[string]*workerConnection),
		requests:     make(chan struct{}, maxConcurrentRequests),
		logger:       slog.Default(),
		jobs:         make(map[uint64]activeJob),
	}
}

func (h *hub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/" || strings.HasPrefix(r.URL.Path, "/dashboard/") {
		h.serveDashboard(w, r)
		return
	}
	// This experiment requires end-to-end TLS, even behind a reverse proxy.
	// Forwarded headers are not an authentication or transport-security signal.
	if r.TLS == nil || r.Header.Get("Origin") != "" {
		writeAPIError(w, r, http.StatusForbidden, "TLS required; browser requests are unsupported", "transport_forbidden")
		return
	}
	if r.URL.Path == workerPath {
		h.connectWorker(w, r)
		return
	}
	if !authenticated(r, h.clientToken) {
		h.logger.Warn("request rejected", "reason", "client_authentication_failed")
		writeAPIError(w, r, http.StatusUnauthorized, "unauthorized", "invalid_api_key")
		return
	}
	if !allowedRequest(r) {
		writeAPIError(w, r, http.StatusNotFound, "unsupported inference route", "unsupported_route")
		return
	}
	if r.ContentLength > maxRequestBytes {
		writeAPIError(w, r, http.StatusRequestEntityTooLarge, "request body too large", "request_too_large")
		return
	}
	if r.URL.Path == "/api/tags" || r.URL.Path == "/v1/models" {
		h.serveModels(w, r)
		return
	}
	select {
	case h.requests <- struct{}{}:
		defer func() { <-h.requests }()
	default:
		h.logger.Warn("inference rejected", "reason", "concurrency_limit")
		writeAPIError(w, r, http.StatusServiceUnavailable, "hub concurrency limit reached", "concurrency_limit")
		return
	}
	h.routeInference(w, r)
}

func (h *hub) connectWorker(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(workerIDHeader)
	token, known := h.workerTokens[id]
	if !known || !authenticated(r, token) {
		h.logger.Warn("worker connection rejected", "reason", "worker_authentication_failed")
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if r.Method != http.MethodGet || r.URL.RawQuery != "" {
		http.Error(w, "unsupported worker request", http.StatusBadRequest)
		return
	}
	if !hasProtocol(r) {
		h.logger.Warn("worker connection rejected", "worker_id", id, "reason", "unsupported_protocol")
		http.Error(w, "unsupported worker protocol", http.StatusUpgradeRequired)
		return
	}
	hardware, err := decodeHardware(r.Header.Get(hardwareHeader))
	if err != nil {
		http.Error(w, "invalid hardware metadata", http.StatusBadRequest)
		return
	}
	ws, err := websocket.Accept(w, r, &websocket.AcceptOptions{Subprotocols: []string{protocol}})
	if err != nil {
		return
	}
	defer ws.CloseNow()
	session, err := yamux.Client(websocket.NetConn(r.Context(), ws, websocket.MessageBinary), sessionConfig())
	if err != nil {
		return
	}
	defer session.Close()
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return session.Open()
		},
		// One HTTP request per stream prevents net/http replaying a request
		// after a reused connection fails. This prototype never retries jobs.
		DisableKeepAlives:      true,
		ResponseHeaderTimeout:  5 * time.Minute,
		MaxResponseHeaderBytes: 64 << 10,
	}
	defer transport.CloseIdleConnections()
	worker := &workerConnection{id: id, session: session, transport: transport, hardware: hardware}
	h.mu.Lock()
	existing := h.workers[id]
	if h.closed || (existing != nil && !existing.session.IsClosed()) {
		h.mu.Unlock()
		h.logger.Warn("worker connection rejected", "worker_id", id, "reason", "identity_unavailable")
		_ = ws.Close(websocket.StatusPolicyViolation, "worker identity already connected or hub closed")
		return
	}
	h.workers[id] = worker
	connected := len(h.workers)
	h.mu.Unlock()
	h.logger.Info("worker connected", "worker_id", id, "connected_workers", connected)
	defer func() {
		h.mu.Lock()
		if h.workers[id] == worker {
			delete(h.workers, id)
		}
		connected := len(h.workers)
		h.mu.Unlock()
		h.logger.Info("worker disconnected", "worker_id", id, "connected_workers", connected)
	}()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	go func() {
		select {
		case <-session.CloseChan():
			cancel()
		case <-ctx.Done():
		}
	}()
	h.refreshInventory(ctx, worker)
	ticker := time.NewTicker(inventoryInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.refreshInventory(ctx, worker)
		}
	}
}

func hasProtocol(r *http.Request) bool {
	for _, value := range strings.Split(r.Header.Get("Sec-WebSocket-Protocol"), ",") {
		if strings.TrimSpace(value) == protocol {
			return true
		}
	}
	return false
}

func (h *hub) Close() error {
	h.mu.Lock()
	h.closed = true
	workers := make([]*workerConnection, 0, len(h.workers))
	for _, worker := range h.workers {
		workers = append(workers, worker)
	}
	h.mu.Unlock()
	results := make(chan error, len(workers))
	for _, worker := range workers {
		go func() { results <- worker.session.Close() }()
	}
	var result error
	for range workers {
		result = errors.Join(result, <-results)
	}
	return result
}

func connectToHub(ctx context.Context, client *http.Client, address *url.URL, id, token string, target *url.URL, logger *slog.Logger, hardware workerHardware) error {
	logger = logger.With("worker_id", id)
	metadata, err := json.Marshal(hardware)
	if err != nil {
		return fmt.Errorf("cannot encode hardware metadata")
	}
	ws, response, err := websocket.Dial(ctx, address.String(), &websocket.DialOptions{
		HTTPClient:   client,
		HTTPHeader:   http.Header{"Authorization": {"Bearer " + token}, workerIDHeader: {id}, hardwareHeader: {base64.RawURLEncoding.EncodeToString(metadata)}},
		Subprotocols: []string{protocol},
	})
	if err != nil {
		status := 0
		if response != nil {
			status = response.StatusCode
		}
		if ctx.Err() == nil {
			logger.Warn("hub connection failed", "reason", connectionReason(err), "status", status)
		}
		if response != nil && response.Body != nil {
			_ = response.Body.Close()
		}
		// Do not return raw library errors: they can include URLs or headers.
		return fmt.Errorf("worker connection failed")
	}
	defer ws.CloseNow()
	if ws.Subprotocol() != protocol {
		logger.Warn("hub connection rejected", "reason", "unsupported_protocol")
		return fmt.Errorf("hub selected an unsupported protocol")
	}
	session, err := yamux.Server(websocket.NetConn(ctx, ws, websocket.MessageBinary), sessionConfig())
	if err != nil {
		return fmt.Errorf("cannot create worker session")
	}
	defer session.Close()
	logger.Info("connected to hub")
	defer logger.Info("disconnected from hub")
	return serveEngine(ctx, session, target, logger)
}
