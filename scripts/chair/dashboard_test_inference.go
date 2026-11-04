// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

const testPromptLimit = 4096

type inferenceTestContextKey struct{}

var inferenceTestTemplate = template.Must(template.ParseFS(dashboardAssets, "web/test.html", "web/ambient.html"))

func (h *hub) configureInferenceTest(enabled bool) error {
	if enabled && h.dashboardToken == "" {
		return fmt.Errorf("--allow-inference-test requires --dashboard-token-file")
	}
	h.allowInferenceTest = enabled
	return nil
}

// Called only after dashboard TLS, authentication, same-origin, and explicit
// action checks. No client token is exposed or injected into a browser request.
func (h *hub) serveInferenceTest(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/dashboard/test":
		var page bytes.Buffer
		if err := inferenceTestTemplate.Execute(&page, nil); err != nil {
			http.Error(w, "test page unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(page.Bytes())
	case "/dashboard/test.js":
		script, err := dashboardAssets.ReadFile("web/test.js")
		if err != nil {
			http.Error(w, "test script unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		_, _ = w.Write(script)
	case "/dashboard/test/models":
		request := r.Clone(r.Context())
		request.URL.Path = "/v1/models"
		h.serveModels(w, request)
	case "/dashboard/test/run":
		h.runInferenceTest(w, r)
	}
}

func (h *hub) runInferenceTest(w http.ResponseWriter, r *http.Request) {
	kind, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || kind != "application/json" {
		http.Error(w, "JSON test request required", http.StatusUnsupportedMediaType)
		return
	}
	controller := http.NewResponseController(w)
	_ = controller.SetReadDeadline(time.Now().Add(5 * time.Second))
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 16<<10))
	_ = controller.SetReadDeadline(time.Time{})
	if err != nil {
		var limit *http.MaxBytesError
		if errors.As(err, &limit) {
			http.Error(w, "test request too large", http.StatusRequestEntityTooLarge)
		} else {
			http.Error(w, "cannot read test request", http.StatusBadRequest)
		}
		return
	}
	fields, valid := responseObject(body)
	for key := range fields {
		valid = valid && (key == "model" || key == "prompt")
	}
	var input struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
	}
	if !valid || !utf8.Valid(body) || json.Unmarshal(body, &input) != nil || !validModel(input.Model) || strings.TrimSpace(input.Prompt) == "" || utf8.RuneCountInString(input.Prompt) > testPromptLimit {
		http.Error(w, "choose a model and enter a prompt of 1 to 4096 characters; only model and prompt are accepted", http.StatusBadRequest)
		return
	}
	// Construct the small, fixed chat request server-side: callers cannot add
	// tools, upstream URLs, history, token overrides, or arbitrary API routes.
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	requestBody, err := json.Marshal(struct {
		Model     string    `json:"model"`
		Messages  []message `json:"messages"`
		Stream    bool      `json:"stream"`
		MaxTokens int       `json:"max_tokens"`
	}{Model: input.Model, Messages: []message{{Role: "user", Content: input.Prompt}}, Stream: true, MaxTokens: 256})
	if err != nil {
		http.Error(w, "cannot prepare test request", http.StatusInternalServerError)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), time.Minute)
	defer cancel()
	// Context cancellation stops upstream work; a write deadline also bounds a
	// browser that stops reading while the proxy is writing a stream chunk.
	_ = controller.SetWriteDeadline(time.Now().Add(65 * time.Second))
	defer controller.SetWriteDeadline(time.Time{})
	request := r.Clone(context.WithValue(ctx, inferenceTestContextKey{}, true))
	request.URL.Path = "/v1/chat/completions"
	request.Header = http.Header{"Content-Type": {"application/json"}, "Accept": {"text/event-stream"}}
	request.Body = io.NopCloser(bytes.NewReader(requestBody))
	request.ContentLength = int64(len(requestBody))
	request.TransferEncoding = nil
	request.GetBody = nil
	h.serveInference(w, request)
}
