// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const secondWorkerToken = "second-worker-test-credential-not-for-real-use"

func startTwoWorkers(t *testing.T, firstModels, secondModels []string, first, second http.HandlerFunc) transportFixture {
	t.Helper()
	fixture := startHubFixture(t, map[string]string{
		"gpu-a": testWorkerToken,
		"gpu-b": secondWorkerToken,
	}, defaultMaxConcurrentRequests)
	fixture.cancel = startTestWorker(t, fixture, "gpu-a", testWorkerToken, testEngine(t, firstModels, first))
	startTestWorker(t, fixture, "gpu-b", secondWorkerToken, testEngine(t, secondModels, second))
	return fixture
}

func workerReply(t *testing.T, id string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"worker": id}))
	}
}

func servedBy(t *testing.T, fixture transportFixture, model string) string {
	t.Helper()
	body, err := json.Marshal(map[string]string{"model": model})
	require.NoError(t, err)
	response, err := fixture.client.Do(fixture.request(t, context.Background(), string(body)))
	require.NoError(t, err)
	responseBody, readErr := io.ReadAll(response.Body)
	require.NoError(t, errors.Join(readErr, response.Body.Close()))
	require.Equal(t, http.StatusOK, response.StatusCode)
	var result struct {
		Worker string `json:"worker"`
	}
	require.NoError(t, json.Unmarshal(responseBody, &result))
	return result.Worker
}

type listedModel struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

func listedModels(t *testing.T, fixture transportFixture) []listedModel {
	t.Helper()
	models, err := fetchListedModels(fixture)
	require.NoError(t, err)
	return models
}

func fetchListedModels(fixture transportFixture) ([]listedModel, error) {
	request, err := http.NewRequest(http.MethodGet, fixture.server.URL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	response, err := fixture.client.Do(request)
	if err != nil {
		return nil, err
	}
	body, readErr := io.ReadAll(response.Body)
	if err := errors.Join(readErr, response.Body.Close()); err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("model listing returned HTTP %d", response.StatusCode)
	}
	var inventory struct {
		Models []listedModel `json:"models"`
	}
	if err := json.Unmarshal(body, &inventory); err != nil {
		return nil, err
	}
	return inventory.Models, nil
}

func TestMultiWorker_RoutesOnlyToOwnersOfRequestedModel(t *testing.T) {
	fixture := startTwoWorkers(t,
		[]string{"llama:latest"}, []string{"qwen:7b"},
		workerReply(t, "gpu-a"), workerReply(t, "gpu-b"))

	assert.Equal(t, "gpu-b", servedBy(t, fixture, "qwen:7b"))
	assert.Equal(t, "gpu-a", servedBy(t, fixture, "llama"))
}

func TestMultiWorker_MergesDistinctModelsAndPreservesMetadata(t *testing.T) {
	fixture := startTwoWorkers(t,
		[]string{"shared:latest", "llama:8b"}, []string{"qwen:7b", "shared:latest"},
		workerReply(t, "gpu-a"), workerReply(t, "gpu-b"))

	assert.Equal(t, []listedModel{
		{Name: "llama:8b", Digest: "test-digest"},
		{Name: "qwen:7b", Digest: "test-digest"},
		{Name: "shared:latest", Digest: "test-digest"},
	}, listedModels(t, fixture))
}

func TestMultiWorker_RejectsUnknownModelsWithoutContactingEngines(t *testing.T) {
	calls := make(chan struct{}, 2)
	handler := func(w http.ResponseWriter, _ *http.Request) {
		calls <- struct{}{}
		w.WriteHeader(http.StatusNoContent)
	}
	fixture := startTwoWorkers(t, []string{"llama:8b"}, []string{"qwen:7b"}, handler, handler)

	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"not-installed"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
	assert.Empty(t, calls)
}

func TestMultiWorker_SpreadsSimultaneousRequestsBeforeResponseHeaders(t *testing.T) {
	release, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan string, maxConcurrentRequests)
	handler := func(id string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			started <- id
			select {
			case <-release.Done():
				assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"worker": id}))
			case <-r.Context().Done():
			}
		}
	}
	fixture := startTwoWorkers(t, []string{"shared:latest"}, []string{"shared:latest"}, handler("gpu-a"), handler("gpu-b"))
	type result struct {
		response *http.Response
		err      error
	}
	results := make(chan result, maxConcurrentRequests)
	for range maxConcurrentRequests {
		request := fixture.request(t, context.Background(), `{"model":"shared"}`)
		go func() {
			response, err := fixture.client.Do(request)
			results <- result{response, err}
		}()
	}

	counts := map[string]int{"gpu-a": 0, "gpu-b": 0}
	for range maxConcurrentRequests {
		select {
		case id := <-started:
			counts[id]++
		case <-time.After(4 * time.Second):
			t.Fatal("concurrent jobs did not reach both workers before response headers")
		}
	}
	assert.Equal(t, map[string]int{"gpu-a": 8, "gpu-b": 8}, counts)
	cancel()
	for range maxConcurrentRequests {
		result := <-results
		require.NoError(t, result.err)
		assert.Equal(t, http.StatusOK, result.response.StatusCode)
		assert.NoError(t, result.response.Body.Close())
	}
}

func TestMultiWorker_RotatesEquallyIdleOwners(t *testing.T) {
	fixture := startTwoWorkers(t,
		[]string{"shared:latest"}, []string{"shared:latest"},
		workerReply(t, "gpu-a"), workerReply(t, "gpu-b"))

	assert.Equal(t, "gpu-a", servedBy(t, fixture, "shared"))
	assert.Equal(t, "gpu-b", servedBy(t, fixture, "shared"))
	assert.Equal(t, "gpu-a", servedBy(t, fixture, "shared"))
	assert.Equal(t, "gpu-b", servedBy(t, fixture, "shared"))
}

func TestMultiWorker_DisconnectExcludesWorkerWithoutReplayingItsJob(t *testing.T) {
	secondCalls := make(chan struct{}, 2)
	fixture := startTwoWorkers(t, []string{"shared:latest"}, []string{"shared:latest"},
		func(w http.ResponseWriter, r *http.Request) {
			assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"worker": "gpu-a"}))
			assert.NoError(t, http.NewResponseController(w).Flush())
			<-r.Context().Done()
		},
		func(w http.ResponseWriter, _ *http.Request) {
			secondCalls <- struct{}{}
			assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"worker": "gpu-b"}))
		})
	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"shared"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })
	var first struct {
		Worker string `json:"worker"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&first))
	require.Equal(t, "gpu-a", first.Worker)

	fixture.cancel()
	_, err = io.ReadAll(response.Body)
	require.Error(t, err)
	assert.Empty(t, secondCalls, "a partial job must not be replayed on the surviving worker")
	assert.Equal(t, "gpu-b", servedBy(t, fixture, "shared"))
	assert.Len(t, secondCalls, 1)
}

func TestMultiWorker_RefreshesAddedAndRemovedModels(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken}, defaultMaxConcurrentRequests)
	initial := `{"models":[{"name":"old:latest","digest":"old-digest"}]}`
	var report atomic.Pointer[string]
	report.Store(&initial)
	startTestWorker(t, fixture, "gpu-a", testWorkerToken, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, err := io.WriteString(w, *report.Load())
			assert.NoError(t, err)
			return
		}
		workerReply(t, "gpu-a")(w, r)
	})
	require.Equal(t, []listedModel{{Name: "old:latest", Digest: "old-digest"}}, listedModels(t, fixture))
	updated := `{"models":[{"name":"new:latest","digest":"new-digest"}]}`

	report.Store(&updated)

	require.Eventually(t, func() bool {
		models, err := fetchListedModels(fixture)
		return err == nil && len(models) == 1 && models[0].Name == "new:latest"
	}, 5*time.Second, 20*time.Millisecond)
	assert.Equal(t, "gpu-a", servedBy(t, fixture, "new"))
	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"old"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
}

func TestMultiWorker_InvalidInventoryWithdrawsWorkerCapabilities(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, defaultMaxConcurrentRequests)
	var invalid atomic.Bool
	startTestWorker(t, fixture, "gpu-a", testWorkerToken, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" && invalid.Load() {
			_, err := io.WriteString(w, `{"models":null}`)
			assert.NoError(t, err)
			return
		}
		testEngine(t, []string{"only-a:latest"}, workerReply(t, "gpu-a"))(w, r)
	})
	startTestWorker(t, fixture, "gpu-b", secondWorkerToken,
		testEngine(t, []string{"only-b:latest"}, workerReply(t, "gpu-b")))

	invalid.Store(true)

	require.Eventually(t, func() bool {
		models, err := fetchListedModels(fixture)
		return err == nil && len(models) == 1 && models[0].Name == "only-b:latest"
	}, 5*time.Second, 20*time.Millisecond)
	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"only-a"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
	assert.Equal(t, "gpu-b", servedBy(t, fixture, "only-b"))
}

func TestMultiWorker_WorkerCannotAuthenticateAsAnotherIdentity(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, defaultMaxConcurrentRequests)
	request, err := http.NewRequest(http.MethodGet, fixture.server.URL+workerPath, nil)
	require.NoError(t, err)
	request.Header.Set(workerIDHeader, "gpu-b")
	request.Header.Set("Authorization", "Bearer "+testWorkerToken)
	request.Header.Set("Sec-WebSocket-Protocol", protocol)

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusUnauthorized, response.StatusCode)
}

func TestMultiWorker_AdmitsReconnectedIdentityAfterDisconnect(t *testing.T) {
	fixture := startTransport(t, workerReply(t, "before-reconnect"))
	require.Equal(t, "before-reconnect", servedBy(t, fixture, "test-model"))

	fixture.cancel()
	require.Eventually(t, func() bool {
		_, err := fetchListedModels(fixture)
		return err != nil
	}, 3*time.Second, 10*time.Millisecond)
	startTestWorker(t, fixture, testWorkerID, testWorkerToken,
		testEngine(t, []string{"test-model:latest"}, workerReply(t, "after-reconnect")))

	assert.Equal(t, "after-reconnect", servedBy(t, fixture, "test-model"))
}

func TestMultiWorker_EmptyInventoryServesNoInference(t *testing.T) {
	calls := make(chan struct{}, 1)
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{},
		func(w http.ResponseWriter, _ *http.Request) {
			calls <- struct{}{}
			w.WriteHeader(http.StatusNoContent)
		}))
	require.Equal(t, []listedModel{}, listedModels(t, fixture))

	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusBadGateway, response.StatusCode)
	assert.Empty(t, calls)
}

func TestInference_RejectsMissingOrMalformedModel(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken}, defaultMaxConcurrentRequests)

			response, err := fixture.client.Do(fixture.request(t, context.Background(), body))
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())

			assert.Equal(t, http.StatusBadRequest, response.StatusCode)
		})
	}
	test("missing", `{}`)
	test("empty", `{"model":""}`)
	test("wrong type", `{"model":42}`)
	test("malformed JSON", `{`)
	test("trailing JSON", `{"model":"qwen"}{}`)
	test("embedded whitespace", `{"model":"not a model"}`)
}

func TestCanonicalModel_NormalizesOnlyImplicitLatestTags(t *testing.T) {
	test := func(name, input, expected string) {
		t.Run(name, func(t *testing.T) { assert.Equal(t, expected, canonicalModel(input)) })
	}
	test("implicit tag", "llama", "llama:latest")
	test("explicit tag", "llama:8b", "llama:8b")
	test("registry port", "registry:5000/team/model", "registry:5000/team/model:latest")
	test("digest", "model@sha256:abc", "model@sha256:abc")
}

func TestWorkerCredentials_RejectsAmbiguousOrInvalidConfiguration(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			_, err := readWorkerTokens(strings.NewReader(body), testClientToken)
			assert.Error(t, err)
		})
	}
	test("empty list", `{"workers":[]}`)
	test("unknown fields", `{"workers":[],"typo":true}`)
	test("malformed JSON", `{`)
	test("short credential", `{"workers":[{"id":"gpu-a","token":"short"}]}`)
	test("invalid identity", `{"workers":[{"id":"gpu a","token":"`+testWorkerToken+`"}]}`)
	test("duplicate identity", `{"workers":[{"id":"gpu-a","token":"`+testWorkerToken+`"},{"id":"gpu-a","token":"`+secondWorkerToken+`"}]}`)
	test("duplicate credential", `{"workers":[{"id":"gpu-a","token":"`+testWorkerToken+`"},{"id":"gpu-b","token":"`+testWorkerToken+`"}]}`)
	test("client credential reused", `{"workers":[{"id":"gpu-a","token":"`+testClientToken+`"}]}`)
	test("trailing JSON", `{"workers":[{"id":"gpu-a","token":"`+testWorkerToken+`"}]}{}`)
}

func TestWorkerCredentials_LoadsDistinctIdentitiesAndCredentials(t *testing.T) {
	config := `{"workers":[{"id":"gpu-a","token":"` + testWorkerToken + `"},{"id":"gpu-b","token":"` + secondWorkerToken + `"}]}`

	tokens, err := readWorkerTokens(strings.NewReader(config), testClientToken)

	require.NoError(t, err)
	assert.Equal(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, tokens)
}
