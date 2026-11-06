// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func enableTestInference(t *testing.T, h *hub) {
	t.Helper()
	require.NoError(t, h.enableDashboard(testDashboardToken))
	require.NoError(t, h.configureInferenceTest(true))
}

func dashboardTestRequest(t *testing.T, fixture transportFixture, body string) *http.Request {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, fixture.server.URL+"/dashboard/test/run", strings.NewReader(body))
	require.NoError(t, err)
	request.SetBasicAuth("chair", testDashboardToken)
	request.Header.Set("Origin", fixture.server.URL)
	request.Header.Set("X-Chair-Test", "1")
	request.Header.Set("Content-Type", "application/json")
	return request
}

func TestDashboardInference_RequiresExplicitOptInAndDashboardCredentials(t *testing.T) {
	h := newHub(testClientToken, nil, 1)
	require.EqualError(t, h.configureInferenceTest(true), "--allow-inference-test requires --dashboard-token-file")
	require.NoError(t, h.configureDashboard(true, false, ""))
	page := httptest.NewRecorder()
	h.ServeHTTP(page, httptest.NewRequest(http.MethodGet, "https://hub/", nil))
	assert.NotContains(t, page.Body.String(), `href="/dashboard/test"`)
	for _, path := range []string{"/dashboard/test", "/dashboard/test.js", "/dashboard/test/models", "/dashboard/test/run"} {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()
			h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub"+path, nil))
			assert.Equal(t, http.StatusNotFound, response.Code)
		})
	}
}

func TestDashboardInference_ProtectsActionsEvenOnPublicDashboard(t *testing.T) {
	test := func(name, address, method, credential, origin, marker, site string, expected int) {
		t.Run(name, func(t *testing.T) {
			h := configuredDashboard(t, true, false, "")
			require.NoError(t, h.configureInferenceTest(true))
			request := httptest.NewRequest(method, address, strings.NewReader(`{"model":"coder","prompt":"hello"}`))
			if credential != "" {
				request.SetBasicAuth("chair", credential)
			}
			request.Header.Set("Origin", origin)
			request.Header.Set("X-Chair-Test", marker)
			request.Header.Set("Sec-Fetch-Site", site)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
			for _, token := range []string{testClientToken, testWorkerToken, testDashboardToken} {
				assert.NotContains(t, response.Body.String(), token)
			}
		})
	}
	test("anonymous page", "https://hub/dashboard/test", http.MethodGet, "", "", "", "", http.StatusUnauthorized)
	test("anonymous models", "https://hub/dashboard/test/models", http.MethodGet, "", "", "", "", http.StatusUnauthorized)
	test("anonymous script", "https://hub/dashboard/test.js", http.MethodGet, "", "", "", "", http.StatusUnauthorized)
	test("anonymous send", "https://hub/dashboard/test/run", http.MethodPost, "", "https://hub", "1", "", http.StatusUnauthorized)
	test("client key is not dashboard login", "https://hub/dashboard/test/run", http.MethodPost, testClientToken, "https://hub", "1", "", http.StatusUnauthorized)
	test("worker key is not dashboard login", "https://hub/dashboard/test/run", http.MethodPost, testWorkerToken, "https://hub", "1", "", http.StatusUnauthorized)
	test("TLS required", "http://hub/dashboard/test/run", http.MethodPost, testDashboardToken, "https://hub", "1", "", http.StatusForbidden)
	test("explicit origin required", "https://hub/dashboard/test/run", http.MethodPost, testDashboardToken, "", "1", "", http.StatusForbidden)
	test("foreign origin", "https://hub/dashboard/test/run", http.MethodPost, testDashboardToken, "https://evil", "1", "", http.StatusForbidden)
	test("explicit action required", "https://hub/dashboard/test/run", http.MethodPost, testDashboardToken, "https://hub", "", "", http.StatusForbidden)
	test("cross-site action", "https://hub/dashboard/test/run", http.MethodPost, testDashboardToken, "https://hub", "1", "cross-site", http.StatusForbidden)
	test("query rejected", "https://hub/dashboard/test/run?route=/api/chat", http.MethodPost, testDashboardToken, "https://hub", "1", "", http.StatusForbidden)
	test("GET cannot send", "https://hub/dashboard/test/run", http.MethodGet, testDashboardToken, "", "", "", http.StatusMethodNotAllowed)
	test("authenticated page", "https://hub/dashboard/test", http.MethodGet, testDashboardToken, "", "", "", http.StatusOK)
	test("authenticated request reaches normal routing", "https://hub/dashboard/test/run", http.MethodPost, testDashboardToken, "https://hub", "1", "", http.StatusBadGateway)
}

func TestDashboardInference_ConstructsBoundedChatAndIsolatesCredentials(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableTestInference(t, fixture.hub)
	var logs logBuffer
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	seen := make(chan string, 1)
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "Bearer "+testEngineToken, r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("Cookie"))
		assert.Empty(t, r.Header.Get("Origin"))
		assert.Empty(t, r.Header.Get("X-Chair-Test"))
		body, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		seen <- string(body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Set-Cookie", "unsafe=engine-cookie")
		w.Header().Set("Location", "https://untrusted.example")
		w.Header().Set("Cache-Control", "public")
		_, err = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"PRIVATE_RESPONSE\"}}]}\n\ndata: [DONE]\n\n")
		assert.NoError(t, err)
	}), engineSettings{kind: engineOpenAI, token: testEngineToken})
	request := dashboardTestRequest(t, fixture, `{"model":"coder","prompt":"PRIVATE_PROMPT"}`)
	request.Header.Set("Cookie", "private-browser-cookie")

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	actual, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Contains(t, string(actual), "PRIVATE_RESPONSE")
	assert.JSONEq(t, `{"model":"coder","messages":[{"role":"user","content":"PRIVATE_PROMPT"}],"stream":true,"max_tokens":256}`, <-seen)
	assert.Empty(t, response.Header.Get("Set-Cookie"))
	assert.Empty(t, response.Header.Get("Location"))
	assert.Equal(t, "no-store", response.Header.Get("Cache-Control"))
	for _, secret := range []string{"PRIVATE_PROMPT", "PRIVATE_RESPONSE", "private-browser-cookie", testClientToken, testDashboardToken, testEngineToken, testWorkerToken} {
		assert.NotContains(t, logs.text(), secret)
	}
}

func TestDashboardInference_RejectsInvalidOrUnboundedInput(t *testing.T) {
	h := configuredDashboard(t, true, false, "")
	require.NoError(t, h.configureInferenceTest(true))
	test := func(name, body, contentType string, status int) {
		t.Run(name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "https://hub/dashboard/test/run", strings.NewReader(body))
			request.SetBasicAuth("chair", testDashboardToken)
			request.Header.Set("Origin", "https://hub")
			request.Header.Set("X-Chair-Test", "1")
			request.Header.Set("Content-Type", contentType)
			response := httptest.NewRecorder()
			h.ServeHTTP(response, request)
			assert.Equal(t, status, response.Code)
		})
	}
	test("empty prompt", `{"model":"coder","prompt":" "}`, "application/json", http.StatusBadRequest)
	test("unknown model field", `{"prompt":"hello"}`, "application/json", http.StatusBadRequest)
	test("no tools", `{"model":"coder","prompt":"hi","tools":[]}`, "application/json", http.StatusBadRequest)
	test("no token overrides", `{"model":"coder","prompt":"hi","max_tokens":9999}`, "application/json", http.StatusBadRequest)
	test("duplicate prompt", `{"model":"coder","prompt":"hi","prompt":"other"}`, "application/json", http.StatusBadRequest)
	test("no trailing data", `{"model":"coder","prompt":"hi"}{}`, "application/json", http.StatusBadRequest)
	test("invalid UTF-8", "{\"model\":\"coder\",\"prompt\":\"\xff\"}", "application/json", http.StatusBadRequest)
	test("non-JSON content", `{"model":"coder","prompt":"hi"}`, "text/plain", http.StatusUnsupportedMediaType)
	test("prompt character limit", `{"model":"coder","prompt":"`+strings.Repeat("x", 4097)+`"}`, "application/json", http.StatusBadRequest)
	test("request byte limit", strings.Repeat("x", 16385), "application/json", http.StatusRequestEntityTooLarge)
}

func TestDashboardInference_CancellationStopsEngineAndReleasesJob(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 1)
	enableTestInference(t, fixture.hub)
	canceled := make(chan struct{})
	startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, "data: {}\n\n")
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
		close(canceled)
	}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	response, err := fixture.client.Do(dashboardTestRequest(t, fixture, `{"model":"coder:latest","prompt":"hello"}`).WithContext(ctx))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	assert.Len(t, fixture.hub.dashboardSnapshot().Jobs, 1)

	limited, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder:latest"}`))
	require.NoError(t, err)
	require.NoError(t, limited.Body.Close())
	assert.Equal(t, http.StatusServiceUnavailable, limited.StatusCode, "test jobs share normal API admission limits")
	cancel()
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("dashboard cancellation did not reach the engine")
	}
	require.Eventually(t, func() bool { return len(fixture.hub.dashboardSnapshot().Jobs) == 0 }, time.Second, 10*time.Millisecond)
}

func TestDashboardInference_ListsFreshModelsWithoutClientToken(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 1)
	enableTestInference(t, fixture.hub)
	startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{"coder"}, workerReply(t, testWorkerID)))
	request, err := http.NewRequest(http.MethodGet, fixture.server.URL+"/dashboard/test/models", nil)
	require.NoError(t, err)
	request.SetBasicAuth("chair", testDashboardToken)
	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	var inventory struct {
		Data []openAIModel `json:"data"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&inventory))
	require.Len(t, inventory.Data, 1)
	assert.Equal(t, "coder:latest", inventory.Data[0].ID)
}
