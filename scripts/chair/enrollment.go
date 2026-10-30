// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const enrollmentAttemptsPerMinute = 30

type workerEnrollment struct {
	token    string
	root     *os.Root
	filename string
	mu       sync.Mutex
	limitMu  sync.Mutex
	window   time.Time
	attempts int
}

func (h *hub) enableEnrollment(token, workersPath string) error {
	if validated, err := readToken(bytes.NewBufferString(token)); err != nil || validated != token {
		return fmt.Errorf("invalid enrollment token")
	}
	h.mu.Lock()
	reserved := token == h.clientToken || token == h.dashboardToken
	for _, workerToken := range h.workerTokens {
		reserved = reserved || token == workerToken
	}
	current := maps.Clone(h.workerTokens)
	h.mu.Unlock()
	if reserved {
		return fmt.Errorf("enrollment token must differ from client, dashboard, and worker tokens")
	}
	path, err := filepath.Abs(workersPath)
	if err != nil {
		return fmt.Errorf("invalid worker credentials path")
	}
	root, err := os.OpenRoot(filepath.Dir(path))
	if err != nil {
		return fmt.Errorf("cannot open worker credentials directory")
	}
	state := &workerEnrollment{token: token, root: root, filename: filepath.Base(path)}
	disk, err := state.readCredentials(h.clientToken)
	if err != nil || !maps.Equal(disk, current) {
		_ = root.Close()
		return fmt.Errorf("worker credentials must be a regular, unchanged registry file")
	}
	h.enrollment = state
	return nil
}

func (e *workerEnrollment) readCredentials(clientToken string) (map[string]string, error) {
	before, err := e.root.Lstat(e.filename)
	if err != nil || !before.Mode().IsRegular() {
		return nil, fmt.Errorf("invalid registry file")
	}
	file, err := e.root.Open(e.filename)
	if err != nil {
		return nil, fmt.Errorf("cannot open registry")
	}
	defer file.Close()
	after, err := file.Stat()
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) {
		return nil, fmt.Errorf("registry changed")
	}
	return readWorkerTokens(file, clientToken)
}

func (e *workerEnrollment) allowAttempt(now time.Time) bool {
	e.limitMu.Lock()
	defer e.limitMu.Unlock()
	if e.window.IsZero() || now.Sub(e.window) >= time.Minute {
		e.window, e.attempts = now, 0
	}
	if e.attempts >= enrollmentAttemptsPerMinute {
		return false
	}
	e.attempts++
	return true
}

func (e *workerEnrollment) saveCredentials(tokens map[string]string) error {
	config := workerCredentials{Workers: make([]workerCredential, 0, len(tokens))}
	ids := make([]string, 0, len(tokens))
	for id := range tokens {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		config.Workers = append(config.Workers, workerCredential{ID: id, Token: tokens[id]})
	}
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("cannot encode registry")
	}
	data = append(data, '\n')
	if len(data) > maxInventoryBytes {
		return fmt.Errorf("registry capacity exceeded")
	}
	name := ".chair-workers-" + rand.Text() + ".tmp"
	file, err := e.root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return fmt.Errorf("cannot create registry replacement")
	}
	defer e.root.Remove(name)
	if err := file.Chmod(0600); err != nil {
		_ = file.Close()
		return fmt.Errorf("cannot protect registry replacement")
	}
	_, writeErr := file.Write(data)
	syncErr := file.Sync()
	closeErr := file.Close()
	if errors.Join(writeErr, syncErr, closeErr) != nil {
		return fmt.Errorf("cannot write registry replacement")
	}
	if err := e.root.Rename(name, e.filename); err != nil {
		return fmt.Errorf("cannot replace registry")
	}
	// Rename is the commit point. Directory-sync support varies by platform;
	// never report failure after commit and leave a live/disk split registry.
	if directory, err := e.root.Open("."); err == nil {
		_ = directory.Sync()
		_ = directory.Close()
	}
	return nil
}

func (h *hub) registerWorker(id string) (string, int) {
	e := h.enrollment
	e.mu.Lock()
	defer e.mu.Unlock()
	h.mu.Lock()
	current, closed := maps.Clone(h.workerTokens), h.closed
	h.mu.Unlock()
	if closed {
		return "", http.StatusServiceUnavailable
	}
	// IDs are public/logged metadata; reject accidental credential pastes.
	for _, secret := range []string{h.clientToken, h.dashboardToken, e.token} {
		if secret != "" && strings.Contains(id, secret) {
			return "", http.StatusBadRequest
		}
	}
	for _, secret := range current {
		if strings.Contains(id, secret) {
			return "", http.StatusBadRequest
		}
	}
	if _, exists := current[id]; exists {
		return "", http.StatusConflict
	}
	if len(current) >= 256 {
		return "", http.StatusConflict
	}
	disk, err := e.readCredentials(h.clientToken)
	if err != nil || !maps.Equal(disk, current) {
		h.logger.Warn("worker enrollment failed", "reason", "registry_changed_or_unreadable")
		return "", http.StatusServiceUnavailable
	}
	var token string
	for range 5 {
		var data [32]byte
		if _, err := rand.Read(data[:]); err != nil {
			return "", http.StatusServiceUnavailable
		}
		token = hex.EncodeToString(data[:])
		_, named := current[token]
		duplicate := token == h.clientToken || token == h.dashboardToken || token == e.token || token == id || named
		for _, existing := range current {
			duplicate = duplicate || token == existing
		}
		if !duplicate {
			break
		}
		token = ""
	}
	if token == "" {
		return "", http.StatusServiceUnavailable
	}
	current[id] = token
	if err := e.saveCredentials(current); err != nil {
		h.logger.Warn("worker enrollment failed", "reason", "registry_write_failed")
		return "", http.StatusServiceUnavailable
	}
	h.mu.Lock()
	h.workerTokens = current
	h.mu.Unlock()
	h.logger.Info("worker enrolled", "worker_id", id, "authorized_workers", len(current))
	return token, http.StatusCreated
}

func (h *hub) serveEnrollment(w http.ResponseWriter, r *http.Request) {
	if !h.enrollment.allowAttempt(time.Now()) {
		w.Header().Set("Retry-After", "60")
		http.Error(w, "enrollment request limit reached; try again later", http.StatusTooManyRequests)
		return
	}
	kind, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || kind != "application/json" {
		http.Error(w, "JSON enrollment request required", http.StatusUnsupportedMediaType)
		return
	}
	if err := http.NewResponseController(w).SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil && !errors.Is(err, http.ErrNotSupported) {
		http.Error(w, "enrollment unavailable", http.StatusServiceUnavailable)
		return
	}
	defer func() { _ = http.NewResponseController(w).SetReadDeadline(time.Time{}) }()
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 16<<10))
	if err != nil {
		http.Error(w, "enrollment body unreadable or too large", http.StatusBadRequest)
		return
	}
	var input struct {
		WorkerID        string `json:"worker_id"`
		EnrollmentToken string `json:"enrollment_token"`
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&input) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		http.Error(w, "invalid enrollment request", http.StatusBadRequest)
		return
	}
	expected := sha256.Sum256([]byte(h.enrollment.token))
	actual := sha256.Sum256([]byte(input.EnrollmentToken))
	if subtle.ConstantTimeCompare(expected[:], actual[:]) != 1 {
		http.Error(w, "invalid enrollment credential", http.StatusUnauthorized)
		return
	}
	if !validWorkerID(input.WorkerID) {
		http.Error(w, "worker ID must contain 1 to 64 ASCII letters, digits, dots, underscores, or hyphens", http.StatusBadRequest)
		return
	}
	token, status := h.registerWorker(input.WorkerID)
	if status != http.StatusCreated {
		message := "enrollment unavailable; contact the hub operator before retrying"
		if status == http.StatusConflict {
			message = "worker ID is already registered or worker capacity is reached; contact the hub operator"
		}
		if status == http.StatusBadRequest {
			message = "worker ID must not contain a credential"
		}
		http.Error(w, message, status)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(struct {
		WorkerID    string `json:"worker_id"`
		WorkerToken string `json:"worker_token"`
	}{WorkerID: input.WorkerID, WorkerToken: token})
}
