// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const (
	engineTypeHeader = "X-Chair-Engine-Type"
	engineOllama     = "ollama"
	engineOpenAI     = "openai"
)

type engineSettings struct {
	kind  string
	token string
}

func validEngineType(kind string) bool { return kind == engineOllama || kind == engineOpenAI }

func engineSupportsPath(kind, path string) bool {
	switch path {
	case "/api/tags", "/api/chat", "/api/generate":
		return kind == engineOllama
	case "/v1/models", "/v1/chat/completions":
		return validEngineType(kind)
	default:
		return false
	}
}

func engineTokenFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("cannot open engine token file")
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 4097))
	if err != nil || len(data) > 4096 {
		return "", fmt.Errorf("engine token unreadable or oversized")
	}
	token := strings.TrimSpace(string(data))
	if token == "" {
		return "", fmt.Errorf("engine token must be nonempty")
	}
	for _, char := range token {
		if char < 33 || char > 126 {
			return "", fmt.Errorf("engine token must contain non-whitespace ASCII characters")
		}
	}
	return token, nil
}

// Applied only to worker-to-loopback requests, after the proxy strips incoming
// credentials. This key never travels in worker handshake metadata to the hub.
type engineTransport struct {
	base  http.RoundTripper
	token string
}

func (t engineTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	request := r.Clone(r.Context())
	request.Header.Del("Authorization")
	if t.token != "" {
		request.Header.Set("Authorization", "Bearer "+t.token)
	}
	return t.base.RoundTrip(request)
}
