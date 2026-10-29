// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func openAIRequest(t *testing.T, fixture transportFixture, path, body string) *http.Request {
	t.Helper()
	method := http.MethodPost
	if path == "/v1/models" {
		method = http.MethodGet
	}
	request, err := http.NewRequest(method, fixture.server.URL+path, strings.NewReader(body))
	require.NoError(t, err)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	request.Header.Set("Content-Type", "application/json")
	return request
}

func TestOpenAI_ListsDeduplicatedModelsAcrossWorkers(t *testing.T) {
	fixture := startTwoWorkers(t, []string{"shared", "only-a:7b"}, []string{"shared:latest", "only-b:8b"},
		workerReply(t, "gpu-a"), workerReply(t, "gpu-b"))

	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/models", ""))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	assert.Equal(t, "application/json", response.Header.Get("Content-Type"))
	var list struct {
		Object string        `json:"object"`
		Data   []openAIModel `json:"data"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&list))
	assert.Equal(t, "list", list.Object)
	assert.Equal(t, []openAIModel{
		{ID: "only-a:7b", Object: "model", OwnedBy: "ollama"},
		{ID: "only-b:8b", Object: "model", OwnedBy: "ollama"},
		{ID: "shared:latest", Object: "model", OwnedBy: "ollama"},
	}, list.Data)
}

func TestOpenAI_ForwardsChatAndToolCallsUnchangedToModelOwner(t *testing.T) {
	requestBody := `{
  "model":"only-b:8b",
  "messages":[{"role":"user","content":"Read a file"},{"role":"assistant","tool_calls":[{"id":"call_old","type":"function","function":{"name":"read_file","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_old","content":"file content"}],
  "tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
  "tool_choice":"auto", "temperature":0.2, "stream":false, "extra_option":{"preserve":true}
}`
	responseBody := `{"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"only-b:8b","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	type forwardedRequest struct{ path, body, authorization, cookie string }
	forwarded := make(chan forwardedRequest, 1)
	wrongWorkerCalls := make(chan struct{}, 1)
	fixture := startTwoWorkers(t, []string{"only-a:7b"}, []string{"only-b:8b"},
		func(w http.ResponseWriter, _ *http.Request) {
			wrongWorkerCalls <- struct{}{}
			w.WriteHeader(http.StatusNoContent)
		},
		func(w http.ResponseWriter, r *http.Request) {
			body, err := io.ReadAll(r.Body)
			assert.NoError(t, err)
			forwarded <- forwardedRequest{r.URL.Path, string(body), r.Header.Get("Authorization"), r.Header.Get("Cookie")}
			w.Header().Set("Content-Type", "application/json")
			_, err = io.WriteString(w, responseBody)
			assert.NoError(t, err)
		})
	request := openAIRequest(t, fixture, "/v1/chat/completions", requestBody)
	request.Header.Set("Cookie", "private-client-cookie")

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	actual, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Equal(t, responseBody, string(actual), "engine response bytes must be preserved")
	assert.Equal(t, forwardedRequest{path: "/v1/chat/completions", body: requestBody}, <-forwarded)
	assert.Empty(t, wrongWorkerCalls)
	var result struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(actual, &result))
	require.Len(t, result.Choices, 1)
	assert.Equal(t, "tool_calls", result.Choices[0].FinishReason)
	require.Len(t, result.Choices[0].Message.ToolCalls, 1)
	assert.Equal(t, "read_file", result.Choices[0].Message.ToolCalls[0].Function.Name)
	assert.Equal(t, `{"path":"README.md"}`, result.Choices[0].Message.ToolCalls[0].Function.Arguments)
}

func TestOpenAI_StreamsToolCallDeltasAndDoneWithoutBuffering(t *testing.T) {
	finish, release := context.WithCancel(context.Background())
	defer release()
	first := `data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{"}}]},"finish_reason":null}]}` + "\n\n"
	tail := `data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":"tool_calls"}]}` + "\n\ndata: [DONE]\n\n"
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, first)
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-finish.Done():
			_, err := io.WriteString(w, tail)
			assert.NoError(t, err)
		case <-r.Context().Done():
		}
	})
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"test-model","messages":[],"stream":true,"stream_options":{"include_usage":true}}`))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	assert.Equal(t, "text/event-stream", response.Header.Get("Content-Type"))
	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	require.NoError(t, err)
	blank, err := reader.ReadString('\n')
	require.NoError(t, err)
	assert.Equal(t, first, line+blank, "first delta must arrive while inference is still blocked")
	require.True(t, json.Valid([]byte(strings.TrimSpace(strings.TrimPrefix(line, "data: ")))), "tool delta must remain valid JSON")
	release()
	remaining, err := io.ReadAll(reader)
	require.NoError(t, err)
	assert.Equal(t, tail, string(remaining), "preserve tool-call fragments, finish reason, and [DONE]")
}

func TestOpenAI_ClientCancellationStopsEngineGeneration(t *testing.T) {
	canceled := make(chan struct{})
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, "data: {\"choices\":[]}\n\n")
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
		close(canceled)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request := openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"test-model","messages":[],"stream":true}`)
	response, err := fixture.client.Do(request.WithContext(ctx))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()

	cancel()

	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("OpenAI stream cancellation did not reach the engine")
	}
}

func TestOpenAI_ReturnsStructuredHubErrors(t *testing.T) {
	test := func(name, method, path, token, body string, expectedStatus int, expectedType, expectedCode string) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			request := httptest.NewRequest(method, "https://hub"+path, strings.NewReader(body))
			request.Header.Set("Authorization", "Bearer "+token)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			require.Equal(t, expectedStatus, response.Code)
			assert.Equal(t, "application/json", response.Header().Get("Content-Type"))
			var result struct {
				Error openAIError `json:"error"`
			}
			require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
			assert.Equal(t, expectedType, result.Error.Type)
			assert.Equal(t, expectedCode, result.Error.Code)
			assert.NotEmpty(t, result.Error.Message)
			assert.Nil(t, result.Error.Param)
		})
	}
	test("client token required", http.MethodPost, "/v1/chat/completions", "", `{}`, http.StatusUnauthorized, "authentication_error", "invalid_api_key")
	test("worker token cannot act as client", http.MethodGet, "/v1/models", testWorkerToken, "", http.StatusUnauthorized, "authentication_error", "invalid_api_key")
	test("missing model", http.MethodPost, "/v1/chat/completions", testClientToken, `{}`, http.StatusBadRequest, "invalid_request_error", "invalid_request")
	test("no eligible owner", http.MethodPost, "/v1/chat/completions", testClientToken, `{"model":"absent"}`, http.StatusBadGateway, "server_error", "model_unavailable")
	test("no fresh inventories", http.MethodGet, "/v1/models", testClientToken, "", http.StatusServiceUnavailable, "server_error", "inventory_unavailable")
	test("Responses API not exposed", http.MethodPost, "/v1/responses", testClientToken, `{}`, http.StatusNotFound, "invalid_request_error", "unsupported_route")
}

func TestOpenAI_PreservesEngineErrors(t *testing.T) {
	errorBody := `{"error":{"message":"engine rejected tool choice","type":"invalid_request_error","param":"tool_choice","code":"unsupported_value"}}`
	fixture := startTransport(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, err := io.WriteString(w, errorBody)
		assert.NoError(t, err)
	})

	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"test-model","messages":[]}`))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	assert.Equal(t, http.StatusBadRequest, response.StatusCode)
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	assert.Equal(t, errorBody, string(body))
}

func TestOpenAI_ReturnsStructuredProxyConnectionError(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	proxy := inferenceProxy(&url.URL{Scheme: "http", Host: "worker"}, sensitiveTransportError{}, logger)
	request := httptest.NewRequest(http.MethodPost, "http://hub/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	response := httptest.NewRecorder()

	proxy.ServeHTTP(response, request)

	require.Equal(t, http.StatusBadGateway, response.Code)
	var result struct {
		Error openAIError `json:"error"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
	assert.Equal(t, "upstream_connection_failed", result.Error.Code)
	assert.Equal(t, "server_error", result.Error.Type)
	assert.NotContains(t, result.Error.Message, "PRIVATE_LIBRARY_ERROR_CONTENT")
}

func TestOpenAI_ConcurrencyLimitUsesStructuredError(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
	h.requests <- struct{}{}
	request := httptest.NewRequest(http.MethodPost, "https://hub/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	require.Equal(t, http.StatusServiceUnavailable, response.Code)
	var result struct {
		Error openAIError `json:"error"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
	assert.Equal(t, "concurrency_limit", result.Error.Code)
}

func TestOpenAI_ModelListExcludesDisconnectedWorkers(t *testing.T) {
	fixture := startTwoWorkers(t, []string{"only-a:latest"}, []string{"only-b:latest"}, workerReply(t, "gpu-a"), workerReply(t, "gpu-b"))
	fixture.cancel()
	request := openAIRequest(t, fixture, "/v1/models", "")

	require.Eventually(t, func() bool {
		response, err := fixture.client.Do(request.Clone(context.Background()))
		if err != nil {
			return false
		}
		body, readErr := io.ReadAll(response.Body)
		if errors.Join(readErr, response.Body.Close()) != nil {
			return false
		}
		var list struct {
			Data []openAIModel `json:"data"`
		}
		return json.Unmarshal(body, &list) == nil && len(list.Data) == 1 && list.Data[0].ID == "only-b:latest"
	}, 3*time.Second, 10*time.Millisecond)
}

func TestOpenAI_HealthyEmptyInventoryReturnsEmptyDataArray(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{}, workerReply(t, testWorkerID)))

	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/models", ""))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	var list struct {
		Object string        `json:"object"`
		Data   []openAIModel `json:"data"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&list))
	assert.Equal(t, "list", list.Object)
	assert.Equal(t, []openAIModel{}, list.Data)
}

func TestWorkerProtocol_RejectsWorkersWithoutOpenAIRoutes(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	request := httptest.NewRequest(http.MethodGet, "https://hub/worker", nil)
	request.Header.Set(workerIDHeader, testWorkerID)
	request.Header.Set("Authorization", "Bearer "+testWorkerToken)
	request.Header.Set("Sec-WebSocket-Protocol", "chair-transport-v1")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusUpgradeRequired, response.Code)
}
