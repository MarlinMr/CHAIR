// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func startResponsesTransport(t *testing.T, concurrency int, handler http.HandlerFunc) transportFixture {
	t.Helper()
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, concurrency)
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, handler), engineSettings{kind: engineOpenAI, responses: true, token: testEngineToken})
	return fixture
}

func TestResponses_PreservesStatelessToolRequestsAndEngineCredentials(t *testing.T) {
	requestBody := `{"model":"coder","store":false,"input":[{"role":"user","content":"PRIVATE_RESPONSE_INPUT"},{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":"file content"}],"tools":[{"type":"function","name":"read_file","parameters":{"type":"object"}}],"include":["reasoning.encrypted_content"],"stream":false}`
	reply := `{"id":"resp_test","object":"response","status":"completed","output":[{"type":"function_call","call_id":"call_2","name":"read_file","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5}}`
	seen := make(chan string, 1)
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	var logs logBuffer
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/responses", r.URL.Path)
		assert.Equal(t, "Bearer "+testEngineToken, r.Header.Get("Authorization"))
		assert.Empty(t, r.Header.Get("Cookie"))
		assert.Empty(t, r.Header.Get(responsesHeader))
		body, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		seen <- string(body)
		w.Header().Set("Content-Type", "application/json")
		_, err = io.WriteString(w, reply)
		assert.NoError(t, err)
	}), engineSettings{kind: engineOpenAI, responses: true, token: testEngineToken})
	request := openAIRequest(t, fixture, "/v1/responses", requestBody)
	request.Header.Set("Cookie", "private-cookie")

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Equal(t, requestBody, <-seen)
	assert.Equal(t, reply, string(body))
	for _, secret := range []string{"PRIVATE_RESPONSE_INPUT", "file content", "private-cookie", testEngineToken, testClientToken, testWorkerToken} {
		assert.NotContains(t, logs.text(), secret)
	}
}

func TestResponses_RejectsStatefulOrAmbiguousRequestsBeforeDispatch(t *testing.T) {
	var calls atomic.Int32
	fixture := startResponsesTransport(t, 2, func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); w.WriteHeader(http.StatusOK) })
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", body))
			require.NoError(t, err)
			defer func() { assert.NoError(t, response.Body.Close()) }()
			assert.Equal(t, http.StatusBadRequest, response.StatusCode)
			var result struct {
				Error openAIError `json:"error"`
			}
			require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
			assert.Equal(t, "unsupported_response_state", result.Error.Code)
		})
	}
	test("storage defaults are unsafe", `{"model":"coder","input":"hello"}`)
	test("stored responses", `{"model":"coder","store":true}`)
	test("null storage", `{"model":"coder","store":null}`)
	test("mistyped storage", `{"model":"coder","store":"false"}`)
	test("background job", `{"model":"coder","store":false,"background":true}`)
	test("response ID follow-up", `{"model":"coder","store":false,"previous_response_id":"resp_1"}`)
	test("empty response ID", `{"model":"coder","store":false,"previous_response_id":""}`)
	test("conversation ID", `{"model":"coder","store":false,"conversation":"conv_1"}`)
	test("conversation object", `{"model":"coder","store":false,"conversation":{"id":"conv_1"}}`)
	test("stored input reference", `{"model":"coder","store":false,"input":[{"type":"item_reference","id":"msg_1"}]}`)
	test("duplicate storage", `{"model":"coder","store":true,"store":false}`)
	test("case variant storage", `{"model":"coder","store":false,"Store":true}`)
	test("case variant background", `{"model":"coder","store":false,"Background":true}`)
	test("duplicate input type", `{"model":"coder","store":false,"input":[{"type":"item_reference","type":"message"}]}`)
	test("escaped duplicate control", `{"model":"coder","store":true,"st\u006fre":false}`)
	assert.Zero(t, calls.Load())
}

func TestResponses_AcceptsExplicitStatelessDefaults(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) { assert.True(t, validStatelessResponse([]byte(body))) })
	}
	test("inline text", `{"store":false,"input":"hello"}`)
	test("null affinity fields", `{"store":false,"previous_response_id":null,"conversation":null,"background":null}`)
	test("explicit foreground", `{"store":false,"background":false,"input":[]}`)
	test("inline reasoning", `{"store":false,"input":[{"type":"reasoning","encrypted_content":"opaque"}]}`)
}

func TestResponses_RejectsMalformedJSON(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) { assert.False(t, validStatelessResponse([]byte(body))) })
	}
	test("null object", `null`)
	test("truncated object", `{"store":false`)
	test("trailing object", `{"store":false}{}`)
	test("non-object input item", `{"store":false,"input":["msg_1"]}`)
	test("mistyped input type", `{"store":false,"input":[{"type":123}]}`)
}

func TestResponses_RoutesOnlyToWorkersWithExplicitCapability(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, 4)
	startTestEngineWorker(t, fixture, "gpu-a", testWorkerToken, testOpenAIEngine(t, []string{"coder", "chat-only"}, func(_ http.ResponseWriter, _ *http.Request) { t.Error("Responses reached a chat-only engine") }), engineSettings{kind: engineOpenAI})
	startTestEngineWorker(t, fixture, "gpu-b", secondWorkerToken, testOpenAIEngine(t, []string{"coder"}, workerReply(t, "gpu-b")), engineSettings{kind: engineOpenAI, responses: true})
	for range 4 {
		response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", `{"model":"coder","store":false,"input":"hello"}`))
		require.NoError(t, err)
		var result struct {
			Worker string `json:"worker"`
		}
		require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
		require.NoError(t, response.Body.Close())
		assert.Equal(t, "gpu-b", result.Worker)
	}
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", `{"model":"chat-only","store":false}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
}

func TestResponses_StreamsEventsAndCancelsWithoutDisconnectingWorker(t *testing.T) {
	first := "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n"
	canceled := make(chan struct{})
	fixture := startResponsesTransport(t, 1, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, first)
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
		close(canceled)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", `{"model":"coder","store":false,"stream":true}`).WithContext(ctx))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	reader := bufio.NewReader(response.Body)
	var received strings.Builder
	for range 3 {
		line, err := reader.ReadString('\n')
		require.NoError(t, err)
		received.WriteString(line)
	}
	assert.Equal(t, first, received.String())
	assert.Len(t, fixture.hub.dashboardSnapshot().Jobs, 1)
	limited, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", `{"model":"coder","store":false}`))
	require.NoError(t, err)
	require.NoError(t, limited.Body.Close())
	assert.Equal(t, http.StatusServiceUnavailable, limited.StatusCode)
	cancel()
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("Responses cancellation did not reach the engine")
	}
	require.Eventually(t, func() bool { return len(fixture.hub.dashboardSnapshot().Jobs) == 0 }, time.Second, 10*time.Millisecond)
	assert.Equal(t, 1, fixture.hub.dashboardSnapshot().Connected)
}

func TestResponses_PreservesToolArgumentEventsAndCompletion(t *testing.T) {
	events := "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{}\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"output_tokens\":5}}}\n\n"
	fixture := startResponsesTransport(t, 2, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, events)
		assert.NoError(t, err)
	})
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", `{"model":"coder","store":false,"stream":true}`))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	assert.Equal(t, "text/event-stream", response.Header.Get("Content-Type"))
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Equal(t, events, string(body))
}

func TestResponses_PreservesUnsupportedEngineErrorsWithoutRetry(t *testing.T) {
	var calls atomic.Int32
	engineError := `{"error":{"message":"Responses is unsupported by this engine","type":"invalid_request_error"}}`
	fixture := startResponsesTransport(t, 2, func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, err := io.WriteString(w, engineError)
		assert.NoError(t, err)
	})
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/responses", `{"model":"coder","store":false}`))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Equal(t, http.StatusNotFound, response.StatusCode)
	assert.Equal(t, engineError, string(body))
	assert.Equal(t, int32(1), calls.Load())
}

func TestResponses_RejectsUnsupportedRoutesAndCredentials(t *testing.T) {
	test := func(name, method, path, token string, status int) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, nil, 1)
			request := httptest.NewRequest(method, "https://hub"+path, strings.NewReader(`{"model":"coder","store":false}`))
			request.Header.Set("Authorization", "Bearer "+token)
			response := httptest.NewRecorder()
			h.ServeHTTP(response, request)
			assert.Equal(t, status, response.Code)
		})
	}
	test("worker credential is not a client key", http.MethodPost, "/v1/responses", testWorkerToken, http.StatusUnauthorized)
	test("retrieval", http.MethodGet, "/v1/responses/resp_1", testClientToken, http.StatusNotFound)
	test("deletion", http.MethodDelete, "/v1/responses/resp_1", testClientToken, http.StatusNotFound)
	test("background cancellation", http.MethodPost, "/v1/responses/resp_1/cancel", testClientToken, http.StatusNotFound)
	test("input items", http.MethodGet, "/v1/responses/resp_1/input_items", testClientToken, http.StatusNotFound)
	test("compaction", http.MethodPost, "/v1/responses/compact", testClientToken, http.StatusNotFound)
	test("query string", http.MethodPost, "/v1/responses?stream=true", testClientToken, http.StatusNotFound)
}

func TestResponsesCapability_RequiresOpenAIEngineAndExplicitOptIn(t *testing.T) {
	test := func(name string, settings engineSettings, expected bool) {
		t.Run(name, func(t *testing.T) { assert.Equal(t, expected, settings.supportsPath("/v1/responses")) })
	}
	test("enabled OpenAI engine", engineSettings{kind: engineOpenAI, responses: true}, true)
	test("disabled by default", engineSettings{kind: engineOpenAI}, false)
	test("Ollama excluded", engineSettings{kind: engineOllama, responses: true}, false)
}

func TestResponsesCapability_RejectsInvalidWorkerDeclarations(t *testing.T) {
	test := func(name, kind, capability string) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			request := httptest.NewRequest(http.MethodGet, "https://hub/worker", nil)
			request.Header.Set("Authorization", "Bearer "+testWorkerToken)
			request.Header.Set(workerIDHeader, testWorkerID)
			request.Header.Set("Sec-WebSocket-Protocol", protocol)
			request.Header.Set(engineTypeHeader, kind)
			request.Header.Set(responsesHeader, capability)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, http.StatusBadRequest, response.Code)
		})
	}
	test("Ollama cannot declare Responses", engineOllama, "true")
	test("capability is not arbitrary text", engineOpenAI, "enabled")
}

func TestResponsesCapability_WorkerRejectsRequestsWhenDisabled(t *testing.T) {
	var calls atomic.Int32
	fixture := startOpenAITransport(t, []string{"coder"}, func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); w.WriteHeader(http.StatusOK) })
	fixture.hub.mu.Lock()
	worker := fixture.hub.workers[testWorkerID]
	fixture.hub.mu.Unlock()
	require.NotNil(t, worker)
	// Bypass hub routing to prove the worker independently enforces its opt-in.
	request, err := http.NewRequest(http.MethodPost, "http://worker/v1/responses", strings.NewReader(`{"model":"coder","store":false}`))
	require.NoError(t, err)
	response, err := worker.transport.RoundTrip(request)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusNotFound, response.StatusCode)
	assert.Zero(t, calls.Load())
}

func TestWorkerConfiguration_RejectsResponsesForOllama(t *testing.T) {
	err := runWorker(context.Background(), []string{"--worker-id", testWorkerID, "--enable-responses"})
	require.EqualError(t, err, "--enable-responses requires --engine-type openai")
}
