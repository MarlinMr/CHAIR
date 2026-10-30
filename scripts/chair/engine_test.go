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
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testEngineToken = "local-engine-test-key"

func testOpenAIEngine(t *testing.T, models []string, handler http.HandlerFunc) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			data := make([]openAIModel, 0, len(models))
			for _, model := range models {
				data = append(data, openAIModel{ID: model, Object: "model", OwnedBy: "test-openai"})
			}
			w.Header().Set("Content-Type", "application/json")
			assert.NoError(t, json.NewEncoder(w).Encode(struct {
				Object string        `json:"object"`
				Data   []openAIModel `json:"data"`
			}{Object: "list", Data: data}))
			return
		}
		handler(w, r)
	}
}

func startOpenAITransport(t *testing.T, models []string, handler http.HandlerFunc) transportFixture {
	t.Helper()
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	fixture.cancel = startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, models, handler), engineSettings{kind: engineOpenAI})
	return fixture
}

func TestOpenAIEngine_ListsExactIDsAndPreservesVendorMetadata(t *testing.T) {
	test := func(name, id, owner, model string) {
		t.Run(name, func(t *testing.T) {
			fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
			startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/v1/models", r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				_, err := io.WriteString(w, `{"object":"list","data":[`+model+`]}`)
				assert.NoError(t, err)
			}, engineSettings{kind: engineOpenAI})

			response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/models", ""))
			require.NoError(t, err)
			defer func() { assert.NoError(t, response.Body.Close()) }()
			require.Equal(t, http.StatusOK, response.StatusCode)
			var list struct {
				Object string            `json:"object"`
				Data   []json.RawMessage `json:"data"`
			}
			require.NoError(t, json.NewDecoder(response.Body).Decode(&list))
			assert.Equal(t, "list", list.Object)
			require.Len(t, list.Data, 1)
			assert.JSONEq(t, model, string(list.Data[0]))
			var item openAIModel
			require.NoError(t, json.Unmarshal(list.Data[0], &item))
			assert.Equal(t, id, item.ID)
			assert.Equal(t, owner, item.OwnedBy)
		})
	}
	test("llama.cpp alias", "coder", "llamacpp", `{"id":"coder","object":"model","created":123,"owned_by":"llamacpp","meta":{"n_ctx_train":32768}}`)
	test("vLLM served model", "org/coder", "vllm", `{"id":"org/coder","object":"model","created":456,"owned_by":"vllm","root":"org/coder","parent":null,"max_model_len":32768,"permission":[]}`)
}

func TestOpenAIEngine_DoesNotAddOrRemoveOllamaTags(t *testing.T) {
	fixture := startOpenAITransport(t, []string{"coder", "tagged:custom", "org/coder"}, workerReply(t, testWorkerID))
	test := func(name, model string, expected int) {
		t.Run(name, func(t *testing.T) {
			body, err := json.Marshal(struct {
				Model string `json:"model"`
			}{Model: model})
			require.NoError(t, err)
			response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", string(body)))
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, expected, response.StatusCode)
		})
	}
	test("exact untagged ID", "coder", http.StatusOK)
	test("exact tagged ID", "tagged:custom", http.StatusOK)
	test("namespaced repository ID", "org/coder", http.StatusOK)
	test("no implicit latest tag", "coder:latest", http.StatusBadGateway)
	test("tag is not removed", "tagged", http.StatusBadGateway)
}

func TestMixedEngines_KeepNativeRequestsAndInventoriesOnOllama(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, 8)
	startTestEngineWorker(t, fixture, "gpu-a", testWorkerToken, testOpenAIEngine(t, []string{"shared:latest", "only-openai"}, workerReply(t, "gpu-a")), engineSettings{kind: engineOpenAI})
	startTestWorker(t, fixture, "gpu-b", secondWorkerToken, testEngine(t, []string{"shared", "only-ollama"}, workerReply(t, "gpu-b")))

	for range 4 {
		assert.Equal(t, "gpu-b", servedBy(t, fixture, "shared"))
	}
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"shared:latest","messages":[]}`))
	require.NoError(t, err)
	var selected struct {
		Worker string `json:"worker"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&selected))
	require.NoError(t, response.Body.Close())
	assert.Equal(t, "gpu-a", selected.Worker, "OpenAI requests can use the OpenAI worker, not just Ollama")
	request, err := http.NewRequest(http.MethodGet, fixture.server.URL+"/api/tags", nil)
	require.NoError(t, err)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	tags, err := fixture.client.Do(request)
	require.NoError(t, err)
	defer func() { assert.NoError(t, tags.Body.Close()) }()
	require.Equal(t, http.StatusOK, tags.StatusCode)
	var inventory struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	require.NoError(t, json.NewDecoder(tags.Body).Decode(&inventory))
	names := []string{}
	for _, model := range inventory.Models {
		names = append(names, model.Name)
	}
	assert.ElementsMatch(t, []string{"shared", "only-ollama"}, names)
	assert.Equal(t, engineOpenAI, dashboardEngineType(fixture.hub, "gpu-a"))
}

func dashboardEngineType(h *hub, id string) string {
	for _, worker := range h.dashboardSnapshot().Workers {
		if worker.ID == id {
			return worker.EngineType
		}
	}
	return ""
}

func TestOpenAIEngine_RejectsNativeRequestsWithoutDispatch(t *testing.T) {
	var calls atomic.Int32
	fixture := startOpenAITransport(t, []string{"coder"}, func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); w.WriteHeader(http.StatusOK) })

	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"coder","prompt":"hello"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
	assert.Zero(t, calls.Load())
}

func TestMixedEngines_PreferExactIDsOverImplicitOllamaAliases(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, 8)
	startTestEngineWorker(t, fixture, "gpu-a", testWorkerToken, testOpenAIEngine(t, []string{"coder"}, workerReply(t, "gpu-a")), engineSettings{kind: engineOpenAI})
	startTestWorker(t, fixture, "gpu-b", secondWorkerToken, testEngine(t, []string{"coder"}, workerReply(t, "gpu-b")))
	for range 4 {
		response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder"}`))
		require.NoError(t, err)
		var result struct {
			Worker string `json:"worker"`
		}
		require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
		require.NoError(t, response.Body.Close())
		assert.Equal(t, "gpu-a", result.Worker)
	}
	assert.Equal(t, "gpu-b", servedBy(t, fixture, "coder"), "native API still accepts Ollama's implicit tag")
}

func TestOpenAIEngine_AddsOnlyItsLocalCredentialAndPreservesToolRequests(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 4)
	var logs logBuffer
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	body := `{"model":"coder","messages":[{"role":"user","content":"PRIVATE_PROMPT"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}],"tool_choice":"auto","stream":false}`
	reply := `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`
	seen := make(chan string, 1)
	engine := testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		data, err := io.ReadAll(r.Body)
		assert.NoError(t, err)
		seen <- string(data)
		w.Header().Set("Content-Type", "application/json")
		_, err = io.WriteString(w, reply)
		assert.NoError(t, err)
	})
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer "+testEngineToken, r.Header.Get("Authorization"), "inventory and inference both use the local engine key")
		assert.Empty(t, r.Header.Get("Cookie"))
		assert.Empty(t, r.Header.Get(workerIDHeader))
		assert.Empty(t, r.Header.Get(engineTypeHeader))
		engine(w, r)
	}, engineSettings{kind: engineOpenAI, token: testEngineToken})
	request := openAIRequest(t, fixture, "/v1/chat/completions", body)
	request.Header.Set("Cookie", "private-client-cookie")

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	actual, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Equal(t, http.StatusOK, response.StatusCode)
	assert.Equal(t, body, <-seen)
	assert.Equal(t, reply, string(actual))
	for _, secret := range []string{testEngineToken, testClientToken, testWorkerToken, "PRIVATE_PROMPT", "private-client-cookie"} {
		assert.NotContains(t, logs.text(), secret)
	}
}

func TestOpenAIEngine_StreamsToolDeltasAndDoneBeforeGenerationCompletes(t *testing.T) {
	finish, release := context.WithCancel(context.Background())
	defer release()
	first := "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]}}]}\n\n"
	tail := "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n"
	fixture := startOpenAITransport(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
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
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder","messages":[],"stream":true}`))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	reader := bufio.NewReader(response.Body)
	line, err := reader.ReadString('\n')
	require.NoError(t, err)
	blank, err := reader.ReadString('\n')
	require.NoError(t, err)
	assert.Equal(t, first, line+blank)
	release()
	remaining, err := io.ReadAll(reader)
	require.NoError(t, err)
	assert.Equal(t, tail, string(remaining))
}

func TestOpenAIEngine_CancellationReachesEngineWithoutDisconnectingWorker(t *testing.T) {
	canceled := make(chan struct{})
	fixture := startOpenAITransport(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, "data: {}\n\n")
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
		close(canceled)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder","stream":true}`).WithContext(ctx))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	cancel()
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("engine generation was not canceled")
	}
	assert.Equal(t, 1, fixture.hub.dashboardSnapshot().Connected)
}

func TestOpenAIInventory_RejectsMalformedModelLists(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) { _, err := parseOpenAIInventory([]byte(body)); assert.Error(t, err) })
	}
	test("missing list", `{}`)
	test("null list", `{"object":"list","data":null}`)
	test("wrong object", `{"object":"model","data":[]}`)
	test("missing ID", `{"object":"list","data":[{"object":"model"}]}`)
	test("duplicate IDs", `{"object":"list","data":[{"object":"model","id":"coder"},{"object":"model","id":"coder"}]}`)
	test("whitespace ID", `{"object":"list","data":[{"object":"model","id":"not a model"}]}`)
	test("Unicode control in ID", `{"object":"list","data":[{"object":"model","id":"bad\u0085name"}]}`)
	test("invalid metadata", `{"object":"list","data":[{"object":"model","id":"coder","created":"yesterday"}]}`)
}

func TestEngineTokenFile_AcceptsShortKeysWithoutRelaxingWorkerCredentials(t *testing.T) {
	path := filepath.Join(t.TempDir(), "engine.token")
	require.NoError(t, os.WriteFile(path, []byte("short-local-key\n"), 0600))
	key, err := engineTokenFile(path)
	require.NoError(t, err)
	assert.Equal(t, "short-local-key", key)
	_, err = tokenFile(path)
	assert.Error(t, err)
}

func TestEngineTokenFile_RejectsUnsafeOrUnboundedKeys(t *testing.T) {
	test := func(name, data string) {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "engine.token")
			require.NoError(t, os.WriteFile(path, []byte(data), 0600))
			_, err := engineTokenFile(path)
			assert.Error(t, err)
		})
	}
	test("empty", "")
	test("embedded newline", "secret\nkey")
	test("non-ASCII", "secret-é")
	test("oversized", strings.Repeat("k", 4097))
}

func TestWorkerProtocol_RequiresExplicitSupportedEngineType(t *testing.T) {
	test := func(name, kind, version string, expected int) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			request := httptest.NewRequest(http.MethodGet, "https://hub/worker", nil)
			request.Header.Set("Authorization", "Bearer "+testWorkerToken)
			request.Header.Set(workerIDHeader, testWorkerID)
			request.Header.Set(engineTypeHeader, kind)
			request.Header.Set("Sec-WebSocket-Protocol", version)
			response := httptest.NewRecorder()
			h.ServeHTTP(response, request)
			assert.Equal(t, expected, response.Code)
		})
	}
	test("missing engine type", "", protocol, http.StatusBadRequest)
	test("unsupported type", "vllm-auto", protocol, http.StatusBadRequest)
	test("transport v2 cannot advertise engine types", engineOllama, "chair-transport-v2", http.StatusUpgradeRequired)
}

func TestOpenAIEngine_WithdrawsModelsOnInventoryFailureAndRecovers(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 4)
	var healthy atomic.Bool
	healthy.Store(true)
	var logs logBuffer
	fixture.hub.logger = slog.New(slog.NewJSONHandler(&logs, nil))
	engine := testOpenAIEngine(t, []string{"coder"}, workerReply(t, testWorkerID))
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" && !healthy.Load() {
			http.Error(w, "PRIVATE_ENGINE_DIAGNOSTIC", http.StatusServiceUnavailable)
			return
		}
		engine(w, r)
	}, engineSettings{kind: engineOpenAI})

	healthy.Store(false)

	require.Eventually(t, func() bool { return fixture.hub.dashboardSnapshot().Ready == 0 }, 4*time.Second, 10*time.Millisecond)
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
	assert.NotContains(t, logs.text(), "PRIVATE_ENGINE_DIAGNOSTIC")
	healthy.Store(true)
	require.Eventually(t, func() bool { return fixture.hub.dashboardSnapshot().Ready == 1 }, 4*time.Second, 10*time.Millisecond)
	response, err = fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusOK, response.StatusCode)
}

func TestOpenAIInventory_AllowsAHealthyEmptyList(t *testing.T) {
	models, err := parseOpenAIInventory([]byte(`{"object":"list","data":[]}`))
	require.NoError(t, err)
	assert.Equal(t, map[string]json.RawMessage{}, models)
}

func TestModelSet_IgnoresChangingVendorTimestampsButDetectsNewIDs(t *testing.T) {
	before := map[string]json.RawMessage{"coder": json.RawMessage(`{"id":"coder","created":123}`)}
	refreshed := map[string]json.RawMessage{"coder": json.RawMessage(`{"id":"coder","created":456}`)}
	replacement := map[string]json.RawMessage{"other": json.RawMessage(`{"id":"other","created":456}`)}

	assert.True(t, sameModelSet(before, refreshed))
	assert.False(t, sameModelSet(before, replacement))
}

func TestEngineRoutes_AllowOnlyTheConfiguredAPISubset(t *testing.T) {
	test := func(name, kind, path string, expected bool) {
		t.Run(name, func(t *testing.T) { assert.Equal(t, expected, engineSupportsPath(kind, path)) })
	}
	test("OpenAI model discovery", engineOpenAI, "/v1/models", true)
	test("OpenAI chat", engineOpenAI, "/v1/chat/completions", true)
	test("Ollama OpenAI chat", engineOllama, "/v1/chat/completions", true)
	test("Ollama native chat", engineOllama, "/api/chat", true)
	test("no native API for generic engines", engineOpenAI, "/api/chat", false)
	test("no native inventory for generic engines", engineOpenAI, "/api/tags", false)
	test("no vLLM administrative RPC", engineOpenAI, "/collective_rpc", false)
	test("no llama.cpp slots", engineOpenAI, "/slots", false)
	test("no Responses API", engineOpenAI, "/v1/responses", false)
	test("no unknown engine", "unsupported", "/v1/models", false)
}

func TestWorkerConfiguration_RejectsReusingWorkerCredentialsForTheEngine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "worker.token")
	require.NoError(t, os.WriteFile(path, []byte(testWorkerToken), 0600))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := runWorker(ctx, []string{"--hub", "https://hub", "--worker-id", testWorkerID, "--worker-token-file", path, "--engine-type", engineOpenAI, "--engine-token-file", path})

	require.Error(t, err)
	assert.Equal(t, "engine token must differ from the worker credential", err.Error())
	assert.NotContains(t, err.Error(), testWorkerToken)
}

func TestWorkerConfiguration_RejectsUnsupportedEngineTypesBeforeConnecting(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := runWorker(ctx, []string{"--hub", "https://hub", "--worker-id", testWorkerID, "--engine-type", "automatic"})
	require.Error(t, err)
	assert.Equal(t, "--engine-type must be ollama or openai", err.Error())
}
