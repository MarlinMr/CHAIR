// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

func messagesPath(path string) bool {
	return path == "/v1/messages" || strings.HasPrefix(path, "/v1/messages/")
}

func authenticatedMessages(r *http.Request, token string) bool {
	// Do not accept a second credential when the supplied first one is invalid.
	// Keep x-api-key scoped to this client dialect, not worker/other API auth.
	bearer, key := r.Header.Values("Authorization"), r.Header.Values("X-Api-Key")
	if len(bearer) > 1 || len(key) > 1 || len(bearer)+len(key) == 0 {
		return false
	}
	if len(bearer) > 0 && !authenticated(r, token) {
		return false
	}
	if len(key) > 0 {
		expected := sha256.Sum256([]byte(token))
		actual := sha256.Sum256([]byte(key[0]))
		if subtle.ConstantTimeCompare(expected[:], actual[:]) != 1 {
			return false
		}
	}
	return true
}

func allowedHubRequest(r *http.Request) bool {
	if r.URL.Path == "/v1/messages" {
		return r.Method == http.MethodPost && (r.URL.RawQuery == "" || r.URL.RawQuery == "beta=true")
	}
	return allowedRequest(r)
}

func writeMessagesError(w http.ResponseWriter, status int, message string) {
	kind := "invalid_request_error"
	switch {
	case status == http.StatusUnauthorized:
		kind = "authentication_error"
	case status == http.StatusForbidden:
		kind = "permission_error"
	case status == http.StatusNotFound:
		kind = "not_found_error"
	case status == http.StatusTooManyRequests:
		kind = "rate_limit_error"
	case status == http.StatusServiceUnavailable:
		kind = "overloaded_error"
	case status >= http.StatusInternalServerError:
		kind = "api_error"
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = writeMessagesErrorJSON(w, kind, message)
}

func writeMessagesErrorJSON(w io.Writer, kind, message string) error {
	return json.NewEncoder(w).Encode(struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}{Type: "error", Error: struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	}{kind, message}})
}
