// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"net/http"
	"strings"
)

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type openAIError struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}

// Hub/transport errors use the client's API dialect. Upstream engine responses
// are forwarded unchanged, including their own error bodies and SSE events.
func writeAPIError(w http.ResponseWriter, r *http.Request, status int, message, code string) {
	if !strings.HasPrefix(r.URL.Path, "/v1/") {
		http.Error(w, message, status)
		return
	}
	kind := "invalid_request_error"
	switch {
	case status == http.StatusUnauthorized:
		kind = "authentication_error"
	case status == http.StatusForbidden:
		kind = "permission_error"
	case status >= http.StatusInternalServerError:
		kind = "server_error"
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(struct {
		Error openAIError `json:"error"`
	}{
		Error: openAIError{Message: message, Type: kind, Code: code},
	})
}
