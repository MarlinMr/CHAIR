// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func enableScores(t *testing.T, h *hub) string {
	t.Helper()
	s, path := testScoreStore(t)
	h.stats = s
	return path
}

func TestScoreboard_StreamsUnchangedAndPersistsOnlyAggregates(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	path := enableScores(t, fixture.hub)
	body := `{"model":"coder","messages":[{"role":"user","content":"PRIVATE_PROMPT"}],"stream":true,"stream_options":{"include_usage":true}}`
	first := "data: {\"choices\":[{\"delta\":{\"content\":\"PRIVATE_RESPONSE\"}}]}\n\n"
	last := "data: {\"usage\":{\"completion_tokens\":23}}\n\ndata: [DONE]\n\n"
	finish, release := context.WithCancel(context.Background())
	defer release()
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		input, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		assert.Equal(t, body, string(input), "collecting usage must not rewrite inference requests")
		w.Header().Set("Content-Type", "text/event-stream")
		_, err = io.WriteString(w, first)
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-finish.Done():
		case <-r.Context().Done():
			return
		}
		_, err = io.WriteString(w, last)
		assert.NoError(t, err)
	}), engineSettings{kind: engineOpenAI})
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", body))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	firstBytes := make([]byte, len(first))
	_, err = io.ReadFull(response.Body, firstBytes)
	require.NoError(t, err)
	assert.Equal(t, first, string(firstBytes), "first content must arrive before generation finishes")
	view, err := fixture.hub.stats.snapshot("")
	require.NoError(t, err)
	assert.Empty(t, view.Requests)
	release()
	remaining, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Equal(t, last, string(remaining))
	view, err = fixture.hub.stats.snapshot("")
	require.NoError(t, err)
	require.Len(t, view.Tokens, 1)
	assert.Equal(t, testWorkerID, view.Tokens[0].Worker)
	assert.Equal(t, int64(23), view.Tokens[0].Tokens)
	require.NoError(t, fixture.hub.Close())
	reopened, err := openScoreStore(path)
	require.NoError(t, err)
	defer func() { assert.NoError(t, reopened.close()) }()
	view, err = reopened.snapshot("")
	require.NoError(t, err)
	assert.Equal(t, int64(1), view.Requests[0].Completed)
	stored, err := os.ReadFile(path)
	require.NoError(t, err)
	for _, private := range []string{"PRIVATE_PROMPT", "PRIVATE_RESPONSE", testClientToken, testWorkerToken} {
		assert.NotContains(t, string(stored), private)
	}
}

func TestScoreboard_ExcludesBrowserTestsAndRejectedModels(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	enableTestInference(t, fixture.hub)
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, err := io.WriteString(w, `{"usage":{"completion_tokens":10}}`)
		assert.NoError(t, err)
	}), engineSettings{kind: engineOpenAI})
	response, err := fixture.client.Do(dashboardTestRequest(t, fixture, `{"model":"coder","prompt":"hello"}`))
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	response, err = fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"absent"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
	view, err := fixture.hub.stats.snapshot("")
	require.NoError(t, err)
	assert.Empty(t, view.Requests)
	assert.Empty(t, view.Tokens)
}

func TestScoreboard_RecordsCancellationBeforeResponseHeaders(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	started := make(chan struct{})
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(_ http.ResponseWriter, r *http.Request) {
		close(started)
		<-r.Context().Done()
	}), engineSettings{kind: engineOpenAI})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	request := openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder"}`).WithContext(ctx)
	go func() {
		response, err := fixture.client.Do(request)
		if response != nil {
			assert.NoError(t, response.Body.Close())
		}
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("engine did not receive the inference request")
	}
	cancel()
	require.Error(t, <-done)
	require.Eventually(t, func() bool {
		view, err := fixture.hub.stats.snapshot("")
		return err == nil && len(view.Requests) == 1 && view.Requests[0].Canceled == 1
	}, time.Second, 10*time.Millisecond)
	view, err := fixture.hub.stats.snapshot("")
	require.NoError(t, err)
	assert.Zero(t, view.Requests[0].Failed)
	assert.Zero(t, view.Requests[0].Completed)
}

func TestScoreboard_CountsNativeOutputAgainstCanonicalModel(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{"coder"}, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, err := io.WriteString(w, "{\"response\":\"private\",\"done\":false}\n{\"done\":true,\"eval_count\":20}\n")
		assert.NoError(t, err)
	}))
	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"coder","prompt":"hello"}`))
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Eventually(t, func() bool {
		view, err := fixture.hub.stats.snapshot("coder:latest")
		return err == nil && len(view.Tokens) == 1
	}, time.Second, 10*time.Millisecond, "the hub finalizes accounting after forwarding the last response bytes")
	view, err := fixture.hub.stats.snapshot("coder:latest")
	require.NoError(t, err)
	require.Len(t, view.Tokens, 1)
	assert.Equal(t, int64(20), view.Tokens[0].Tokens)
}

func TestScoreboard_DashboardRoutesPreserveAccessRestrictions(t *testing.T) {
	test := func(name, method, address string, enabled, public, login bool, status int) {
		t.Run(name, func(t *testing.T) {
			h := configuredDashboard(t, public, false, "")
			if enabled {
				enableScores(t, h)
			}
			request := httptest.NewRequest(method, address, nil)
			if login {
				request.SetBasicAuth("chair", testDashboardToken)
			}
			response := httptest.NewRecorder()
			h.ServeHTTP(response, request)
			assert.Equal(t, status, response.Code)
			for _, secret := range []string{testClientToken, testWorkerToken, testDashboardToken} {
				assert.NotContains(t, response.Body.String(), secret)
			}
		})
	}
	test("disabled by default", http.MethodGet, "https://hub/dashboard/scoreboard", false, true, false, http.StatusNotFound)
	test("private page", http.MethodGet, "https://hub/dashboard/scoreboard", true, false, false, http.StatusUnauthorized)
	test("private data", http.MethodGet, "https://hub/dashboard/scoreboard/status", true, false, false, http.StatusUnauthorized)
	test("public page", http.MethodGet, "https://hub/dashboard/scoreboard", true, true, false, http.StatusOK)
	test("authenticated data", http.MethodGet, "https://hub/dashboard/scoreboard/status", true, false, true, http.StatusOK)
	test("TLS required", http.MethodGet, "http://hub/dashboard/scoreboard", true, true, false, http.StatusForbidden)
	test("no mutation", http.MethodPost, "https://hub/dashboard/scoreboard", true, true, false, http.StatusMethodNotAllowed)
	test("model filter", http.MethodGet, "https://hub/dashboard/scoreboard/status?model=org%2Fcoder", true, true, false, http.StatusOK)
	test("duplicate filter rejected", http.MethodGet, "https://hub/dashboard/scoreboard/status?model=a&model=b", true, true, false, http.StatusForbidden)
	test("unknown filter rejected", http.MethodGet, "https://hub/dashboard/scoreboard/status?file=private", true, true, false, http.StatusForbidden)
}

func TestScores_RejectSymlinkedDatabase(t *testing.T) {
	s, path := testScoreStore(t)
	require.NoError(t, s.close())
	link := filepath.Join(t.TempDir(), "link.db")
	if err := os.Symlink(path, link); err != nil {
		t.Skip("symlink creation unavailable on this platform")
	}
	_, err := openScoreStore(link)
	assert.Error(t, err)
}

func TestScoreboard_CountsFailedAndCanceledWorkWithoutTokenCredit(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	started := make(chan struct{})
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		if strings.Contains(string(body), "cancel-me") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, err := io.WriteString(w, "data: {}\n\n")
			assert.NoError(t, err)
			assert.NoError(t, http.NewResponseController(w).Flush())
			close(started)
			<-r.Context().Done()
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}), engineSettings{kind: engineOpenAI})
	failed, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder"}`))
	require.NoError(t, err)
	require.NoError(t, failed.Body.Close())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder","messages":[{"role":"user","content":"cancel-me"}]}`).WithContext(ctx))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	<-started
	cancel()
	require.Eventually(t, func() bool {
		view, err := fixture.hub.stats.snapshot("")
		return err == nil && len(view.Requests) == 1 && view.Requests[0].Canceled == 1
	}, time.Second, 10*time.Millisecond)
	view, err := fixture.hub.stats.snapshot("")
	require.NoError(t, err)
	assert.Equal(t, int64(1), view.Requests[0].Failed)
	assert.Zero(t, view.Requests[0].Completed)
	assert.Empty(t, view.Tokens)
}
