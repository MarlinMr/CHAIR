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
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type logBuffer struct {
	mu   sync.Mutex
	data bytes.Buffer
}

func (b *logBuffer) Write(body []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.data.Write(body)
}

func (b *logBuffer) text() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.data.String()
}

type logEvent struct {
	Level         string `json:"level"`
	Message       string `json:"msg"`
	WorkerID      string `json:"worker_id"`
	Models        int    `json:"models"`
	Reason        string `json:"reason"`
	Status        int    `json:"status"`
	RequestID     uint64 `json:"request_id"`
	Model         string `json:"model"`
	Outcome       string `json:"outcome"`
	Duration      int64  `json:"duration_ms"`
	ResponseBytes int64  `json:"response_bytes"`
}

func (b *logBuffer) events() ([]logEvent, error) {
	decoder := json.NewDecoder(strings.NewReader(b.text()))
	var events []logEvent
	for {
		var event logEvent
		if err := decoder.Decode(&event); err == io.EOF {
			return events, nil
		} else if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
}

func TestCommandLogger_FiltersConfiguredLevels(t *testing.T) {
	test := func(name string, debug, info, warn bool) {
		t.Run(name, func(t *testing.T) {
			logger, err := commandLogger(name, io.Discard)
			require.NoError(t, err)

			assert.Equal(t, debug, logger.Enabled(context.Background(), slog.LevelDebug))
			assert.Equal(t, info, logger.Enabled(context.Background(), slog.LevelInfo))
			assert.Equal(t, warn, logger.Enabled(context.Background(), slog.LevelWarn))
			assert.True(t, logger.Enabled(context.Background(), slog.LevelError))
		})
	}
	test("debug", true, true, true)
	test("info", false, true, true)
	test("warn", false, false, true)
	test("error", false, false, false)
}

func TestCommands_RejectInvalidLogLevelWithoutEchoingInput(t *testing.T) {
	test := func(command string) {
		t.Run(command, func(t *testing.T) {
			err := run(context.Background(), []string{command, "--log-level", "private-invalid-input"})

			require.EqualError(t, err, "--log-level must be debug, info, warn, or error")
		})
	}
	test("hub")
	test("worker")
}

func TestLogging_InfoReportsWorkerLifecycleWithoutDebugTraffic(t *testing.T) {
	var output logBuffer
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&output, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cancel := startTestWorker(t, fixture, testWorkerID, testWorkerToken,
		testEngine(t, []string{"test-model:latest"}, workerReply(t, testWorkerID)))
	require.Equal(t, testWorkerID, servedBy(t, fixture, "test-model"))

	cancel()
	require.Eventually(t, func() bool {
		events, err := output.events()
		if err != nil {
			return false
		}
		for _, event := range events {
			if event.Message == "worker disconnected" {
				return true
			}
		}
		return false
	}, 3*time.Second, 10*time.Millisecond)

	events, err := output.events()
	require.NoError(t, err)
	messages := make(map[string]logEvent)
	for _, event := range events {
		assert.NotEqual(t, "DEBUG", event.Level)
		messages[event.Message] = event
	}
	assert.Equal(t, testWorkerID, messages["worker connected"].WorkerID)
	assert.Equal(t, 1, messages["worker inventory updated"].Models)
	assert.Equal(t, testWorkerID, messages["worker disconnected"].WorkerID)
	assert.NotContains(t, messages, "inference dispatched")
}

func TestLogging_DebugReportsRoutingWithoutInferenceContentsOrCredentials(t *testing.T) {
	var output logBuffer
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&output, &slog.HandlerOptions{Level: slog.LevelDebug}))
	startTestWorker(t, fixture, testWorkerID, testWorkerToken,
		testEngine(t, []string{"test-model:latest"}, func(w http.ResponseWriter, _ *http.Request) {
			assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"response": "PRIVATE_RESPONSE_CONTENT"}))
		}))
	request := fixture.request(t, context.Background(), `{"model":"test-model","prompt":"PRIVATE_PROMPT_CONTENT","messages":[{"content":"PRIVATE_MESSAGE_CONTENT"}]}`)
	request.Header.Set("Cookie", "PRIVATE_COOKIE_CONTENT")
	request.Header.Set("X-Trace-Secret", "PRIVATE_HEADER_CONTENT")

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	_, readErr := io.ReadAll(response.Body)
	require.NoError(t, errors.Join(readErr, response.Body.Close()))
	require.Equal(t, http.StatusOK, response.StatusCode)

	events, err := output.events()
	require.NoError(t, err)
	var finished logEvent
	for _, event := range events {
		if event.Message == "inference finished" {
			finished = event
		}
	}
	assert.Equal(t, "DEBUG", finished.Level)
	assert.Equal(t, uint64(1), finished.RequestID)
	assert.Equal(t, testWorkerID, finished.WorkerID)
	assert.Equal(t, "test-model", finished.Model)
	assert.Equal(t, http.StatusOK, finished.Status)
	assert.Equal(t, "completed", finished.Outcome)
	assert.GreaterOrEqual(t, finished.Duration, int64(0))
	assert.Greater(t, finished.ResponseBytes, int64(0))
	for _, secret := range []string{testClientToken, testWorkerToken, "PRIVATE_PROMPT_CONTENT", "PRIVATE_MESSAGE_CONTENT", "PRIVATE_RESPONSE_CONTENT", "PRIVATE_COOKIE_CONTENT", "PRIVATE_HEADER_CONTENT"} {
		assert.NotContains(t, output.text(), secret)
	}
}

func TestLogging_InventoryFailuresWarnOnceAndReportRecovery(t *testing.T) {
	var output logBuffer
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&output, &slog.HandlerOptions{Level: slog.LevelDebug}))
	var failed atomic.Bool
	healthy := testEngine(t, []string{"test-model:latest"}, workerReply(t, testWorkerID))
	startTestWorker(t, fixture, testWorkerID, testWorkerToken, func(w http.ResponseWriter, r *http.Request) {
		if failed.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, err := io.WriteString(w, "PRIVATE_INVENTORY_FAILURE_BODY")
			assert.NoError(t, err)
			return
		}
		healthy(w, r)
	})
	fixture.hub.mu.Lock()
	worker := fixture.hub.workers[testWorkerID]
	fixture.hub.mu.Unlock()

	failed.Store(true)
	fixture.hub.refreshInventory(context.Background(), worker)
	fixture.hub.refreshInventory(context.Background(), worker)
	failed.Store(false)
	fixture.hub.refreshInventory(context.Background(), worker)

	events, err := output.events()
	require.NoError(t, err)
	warnings, repeated, recovered := 0, 0, 0
	for _, event := range events {
		if event.Message == "worker inventory unavailable" {
			assert.Equal(t, "inventory_http_error", event.Reason)
			assert.Equal(t, http.StatusServiceUnavailable, event.Status)
			if event.Level == "WARN" {
				warnings++
			} else if event.Level == "DEBUG" {
				repeated++
			}
		}
		if event.Message == "worker inventory recovered" {
			recovered++
		}
	}
	assert.Equal(t, 1, warnings)
	assert.GreaterOrEqual(t, repeated, 1)
	assert.Equal(t, 1, recovered)
	assert.NotContains(t, output.text(), "PRIVATE_INVENTORY_FAILURE_BODY")
}

type sensitiveTransportError struct{}

func (sensitiveTransportError) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errors.New("PRIVATE_LIBRARY_ERROR_CONTENT")
}

func TestLogging_ForwardingFailureNeverLogsRawLibraryError(t *testing.T) {
	var output logBuffer
	logger := slog.New(slog.NewJSONHandler(&output, &slog.HandlerOptions{Level: slog.LevelDebug}))
	proxy := inferenceProxy(&url.URL{Scheme: "http", Host: "worker"}, sensitiveTransportError{}, logger)
	request := httptest.NewRequest(http.MethodPost, "http://hub/api/generate", strings.NewReader(`{"model":"test-model"}`))
	response := httptest.NewRecorder()

	proxy.ServeHTTP(response, request)

	assert.Equal(t, http.StatusBadGateway, response.Code)
	events, err := output.events()
	require.NoError(t, err)
	require.Len(t, events, 1)
	assert.Equal(t, "inference forwarding failed", events[0].Message)
	assert.Equal(t, "upstream_connection_failed", events[0].Reason)
	assert.NotContains(t, output.text(), "PRIVATE_LIBRARY_ERROR_CONTENT")
}

func TestLogging_CancellationRemainsVisibleAtDebug(t *testing.T) {
	var output logBuffer
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&output, &slog.HandlerOptions{Level: slog.LevelDebug}))
	startTestWorker(t, fixture, testWorkerID, testWorkerToken,
		testEngine(t, []string{"test-model:latest"}, func(w http.ResponseWriter, r *http.Request) {
			assert.NoError(t, json.NewEncoder(w).Encode(map[string]bool{"done": false}))
			assert.NoError(t, http.NewResponseController(w).Flush())
			<-r.Context().Done()
		}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	response, err := fixture.client.Do(fixture.request(t, ctx, `{"model":"test-model"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })

	cancel()

	require.Eventually(t, func() bool {
		events, err := output.events()
		if err != nil {
			return false
		}
		for _, event := range events {
			if event.Message == "inference finished" && event.Outcome == "canceled" {
				return true
			}
		}
		return false
	}, 3*time.Second, 10*time.Millisecond)
}

func TestLogging_AuthenticationFailureDoesNotReflectUntrustedIdentity(t *testing.T) {
	var output logBuffer
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	h.logger = slog.New(slog.NewJSONHandler(&output, &slog.HandlerOptions{Level: slog.LevelDebug}))
	request := httptest.NewRequest(http.MethodGet, "https://hub/worker?private=query", nil)
	request.Header.Set(workerIDHeader, "PRIVATE_UNTRUSTED_ID")
	request.Header.Set("Authorization", "Bearer PRIVATE_UNTRUSTED_TOKEN")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusUnauthorized, response.Code)
	events, err := output.events()
	require.NoError(t, err)
	require.Len(t, events, 1)
	assert.Equal(t, "worker_authentication_failed", events[0].Reason)
	assert.Empty(t, events[0].WorkerID)
	assert.NotContains(t, output.text(), "PRIVATE_UNTRUSTED_ID")
	assert.NotContains(t, output.text(), "PRIVATE_UNTRUSTED_TOKEN")
	assert.NotContains(t, output.text(), "private=query")
}
