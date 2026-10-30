// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testClientToken       = "client-test-credential-not-for-real-use"
	testWorkerToken       = "worker-test-credential-not-for-real-use"
	testWorkerID          = "gpu-a"
	maxConcurrentRequests = 16
)

type transportFixture struct {
	hub    *hub
	server *httptest.Server
	client *http.Client
	cancel context.CancelFunc
}

func startTransport(t *testing.T, handler http.HandlerFunc) transportFixture {
	t.Helper()
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	fixture.cancel = startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{"test-model:latest"}, handler))
	return fixture
}

func startHubFixture(t *testing.T, workerTokens map[string]string, concurrency int) transportFixture {
	t.Helper()
	h := newHub(testClientToken, workerTokens, concurrency)
	server := httptest.NewTLSServer(h)
	t.Cleanup(server.Close)
	t.Cleanup(func() { assert.NoError(t, h.Close()) })
	client := server.Client()
	client.Timeout = 5 * time.Second
	return transportFixture{hub: h, server: server, client: client}
}

func testEngine(t *testing.T, models []string, inference http.HandlerFunc) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			inventory := make([]map[string]string, 0, len(models))
			for _, name := range models {
				inventory = append(inventory, map[string]string{"name": name, "digest": "test-digest"})
			}
			w.Header().Set("Content-Type", "application/json")
			assert.NoError(t, json.NewEncoder(w).Encode(map[string][]map[string]string{"models": inventory}))
			return
		}
		inference(w, r)
	}
}

func startTestWorker(t *testing.T, fixture transportFixture, id, token string, handler http.HandlerFunc) context.CancelFunc {
	t.Helper()
	return startTestEngineWorker(t, fixture, id, token, handler, engineSettings{kind: engineOllama})
}

func startTestEngineWorker(t *testing.T, fixture transportFixture, id, token string, handler http.HandlerFunc, settings engineSettings) context.CancelFunc {
	t.Helper()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read the real HTTP body before generating so net/http can watch for
		// disconnects, then allow each fake engine to inspect the request too.
		body, err := io.ReadAll(r.Body)
		if !assert.NoError(t, err) {
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		handler(w, r)
	}))
	t.Cleanup(engine.Close)
	address, err := hubURL(fixture.server.URL)
	require.NoError(t, err)
	target, err := engineURL(engine.URL)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- connectToHub(ctx, fixture.client, address, id, token, target, fixture.hub.logger, workerHardware{OS: "linux", Arch: "amd64", CPU: "Test CPU", LogicalCPUs: 8, MemoryBytes: 32 << 30, GPUs: []workerGPU{{Name: "Test GPU", MemoryMiB: 24576}}}, settings)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("worker did not stop after cancellation")
		}
	})
	// Wait for this particular worker's first report, not another worker's
	// aggregate model response. This is a setup barrier, not a routing assertion.
	require.Eventually(t, func() bool {
		fixture.hub.mu.Lock()
		defer fixture.hub.mu.Unlock()
		worker := fixture.hub.workers[id]
		return worker != nil && worker.ready(time.Now())
	}, 5*time.Second, 10*time.Millisecond, "worker should report inventory through its outbound connection")
	return cancel
}

func (f transportFixture) request(t *testing.T, ctx context.Context, body string) *http.Request {
	t.Helper()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, f.server.URL+"/api/generate", strings.NewReader(body))
	require.NoError(t, err)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	request.Header.Set("Content-Type", "application/json")
	return request
}

func TestTransport_StreamsBeforeInferenceCompletes(t *testing.T) {
	finish := make(chan struct{})
	defer close(finish)
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"response": "first"}))
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-finish:
		case <-r.Context().Done():
		}
	})

	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model","stream":true}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })
	require.Equal(t, http.StatusOK, response.StatusCode)
	var chunk struct {
		Response string `json:"response"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&chunk))
	assert.Equal(t, "first", chunk.Response)
}

func TestTransport_ClientCancellationStopsEngineRequest(t *testing.T) {
	engineCanceled := make(chan struct{})
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]bool{"done": false}))
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
		close(engineCanceled)
	})
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	response, err := fixture.client.Do(fixture.request(t, ctx, `{"model":"test-model"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })

	cancel()

	select {
	case <-engineCanceled:
	case <-time.After(3 * time.Second):
		t.Fatal("client cancellation did not reach the engine")
	}
}

func TestTransport_DoesNotForwardCredentialsOrClientCookies(t *testing.T) {
	headers := make(chan http.Header, 1)
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		headers <- r.Header.Clone()
		w.WriteHeader(http.StatusNoContent)
	})
	request := fixture.request(t, context.Background(), `{"model":"test-model"}`)
	request.Header.Set("Cookie", "private-cookie=value")
	request.Header.Set("Proxy-Authorization", "private-proxy-credential")
	request.Header.Set("X-Forwarded-For", "private-address")
	request.Header.Set("X-NVPAIR-Engine-Identity-Probe", "1")

	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Equal(t, http.StatusNoContent, response.StatusCode)
	forwarded := <-headers
	assert.Equal(t, "application/json", forwarded.Get("Content-Type"))
	assert.Empty(t, forwarded.Get("Authorization"))
	assert.Empty(t, forwarded.Get("Proxy-Authorization"))
	assert.Empty(t, forwarded.Get("Cookie"))
	assert.Empty(t, forwarded.Get("X-Forwarded-For"))
	assert.Empty(t, forwarded.Get("X-NVPAIR-Engine-Identity-Probe"))
}

func TestTransport_MultiplexesIndependentRequests(t *testing.T) {
	finish := make(chan struct{})
	defer close(finish)
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]bool{"done": false}))
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-finish:
		case <-r.Context().Done():
		}
	})
	first, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, first.Body.Close()) })

	second, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, second.Body.Close()) })
	assert.Equal(t, http.StatusOK, second.StatusCode)
}

func TestTransport_WorkerDisconnectFailsRunningRequestWithoutReplay(t *testing.T) {
	calls := make(chan struct{}, 2)
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		calls <- struct{}{}
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]bool{"done": false}))
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
	})
	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })
	var first map[string]bool
	require.NoError(t, json.NewDecoder(response.Body).Decode(&first))
	require.Equal(t, map[string]bool{"done": false}, first)

	fixture.cancel()
	_, err = io.ReadAll(response.Body)
	assert.Error(t, err, "a partial response must not appear complete")
	assert.Len(t, calls, 1, "an interrupted request must not be replayed")

	request, err := http.NewRequest(http.MethodGet, fixture.server.URL+"/api/tags", nil)
	require.NoError(t, err)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	require.Eventually(t, func() bool {
		result, err := fixture.client.Do(request.Clone(context.Background()))
		if err != nil {
			return false
		}
		assert.NoError(t, result.Body.Close())
		return result.StatusCode == http.StatusServiceUnavailable
	}, 3*time.Second, 10*time.Millisecond)
}

func TestHub_RejectsUnauthorizedAndUnsupportedRequests(t *testing.T) {
	test := func(name, method, path, token string, expected int) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
			request := httptest.NewRequest(method, "https://hub.example"+path, nil)
			request.Header.Set("Authorization", "Bearer "+token)
			request.Header.Set(workerIDHeader, testWorkerID)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
		})
	}
	test("missing client credential", http.MethodPost, "/api/generate", "", http.StatusUnauthorized)
	test("worker cannot submit inference", http.MethodPost, "/api/generate", testWorkerToken, http.StatusUnauthorized)
	test("client cannot register worker", http.MethodGet, workerPath, testClientToken, http.StatusUnauthorized)
	test("worker protocol must be explicit", http.MethodGet, workerPath, testWorkerToken, http.StatusUpgradeRequired)
	test("engine administration is not exposed", http.MethodPost, "/api/pull", testClientToken, http.StatusNotFound)
	test("query strings cannot alter routing", http.MethodPost, "/api/generate?target=remote", testClientToken, http.StatusNotFound)
	test("no worker connected", http.MethodGet, "/api/tags", testClientToken, http.StatusServiceUnavailable)
}

func TestTransport_RejectsRequestsBeyondConcurrencyLimit(t *testing.T) {
	finish := make(chan struct{})
	defer close(finish)
	calls := make(chan struct{}, maxConcurrentRequests+1)
	fixture := startTransport(t, func(w http.ResponseWriter, r *http.Request) {
		calls <- struct{}{}
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]bool{"done": false}))
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-finish:
		case <-r.Context().Done():
		}
	})
	for range maxConcurrentRequests {
		response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
		require.NoError(t, err)
		t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })
		require.Equal(t, http.StatusOK, response.StatusCode)
	}

	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusServiceUnavailable, response.StatusCode)
	assert.Len(t, calls, maxConcurrentRequests)
}

func TestTransport_DuplicateIdentityCannotDisplaceActiveWorker(t *testing.T) {
	fixture := startTransport(t, func(w http.ResponseWriter, _ *http.Request) {
		assert.NoError(t, json.NewEncoder(w).Encode(map[string]string{"worker": "original"}))
	})
	address, err := hubURL(fixture.server.URL)
	require.NoError(t, err)
	target, err := engineURL("http://127.0.0.1:1")
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	err = connectToHub(ctx, fixture.client, address, testWorkerID, testWorkerToken, target, fixture.hub.logger, workerHardware{}, engineSettings{kind: engineOllama})

	require.Error(t, err)
	response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })
	require.Equal(t, http.StatusOK, response.StatusCode)
	var body struct {
		Worker string `json:"worker"`
	}
	require.NoError(t, json.NewDecoder(response.Body).Decode(&body))
	assert.Equal(t, "original", body.Worker)
}

func TestHub_RejectsOversizedDeclaredBody(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	request := httptest.NewRequest(http.MethodPost, "https://hub.example/api/generate", nil)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	request.ContentLength = maxRequestBytes + 1
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusRequestEntityTooLarge, response.Code)
}

func TestHubURL_RejectsInsecureOrAmbiguousAddresses(t *testing.T) {
	test := func(name, address string) {
		t.Run(name, func(t *testing.T) {
			_, err := hubURL(address)
			assert.Error(t, err)
		})
	}
	test("plaintext", "http://hub.example")
	test("embedded credentials", "https://user:password@hub.example")
	test("path prefix", "https://hub.example/admin")
	test("query string", "https://hub.example?token=secret")
	test("missing host", "https://")
}

func TestReadToken_AcceptsCredentialFileWithTrailingNewline(t *testing.T) {
	token, err := readToken(strings.NewReader(testClientToken + "\n"))

	require.NoError(t, err)
	assert.Equal(t, testClientToken, token)
}

func TestHub_RejectsPlaintextEvenWithForwardedHTTPS(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	request := httptest.NewRequest(http.MethodGet, "http://hub.example/api/tags", nil)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	request.Header.Set("X-Forwarded-Proto", "https")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusForbidden, response.Code)
}

func TestHub_RejectsBrowserOrigins(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	request := httptest.NewRequest(http.MethodGet, "https://hub.example/api/tags", nil)
	request.Header.Set("Authorization", "Bearer "+testClientToken)
	request.Header.Set("Origin", "https://other.example")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusForbidden, response.Code)
}

func TestWorker_RefusesUntrustedHubCertificate(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	server := httptest.NewTLSServer(h)
	t.Cleanup(server.Close)
	t.Cleanup(func() { assert.NoError(t, h.Close()) })
	address, err := hubURL(server.URL)
	require.NoError(t, err)
	target, err := engineURL("http://127.0.0.1:11434")
	require.NoError(t, err)
	client := &http.Client{
		Transport: &http.Transport{TLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12}},
		Timeout:   time.Second,
	}
	t.Cleanup(client.CloseIdleConnections)

	err = connectToHub(context.Background(), client, address, testWorkerID, testWorkerToken, target, h.logger, workerHardware{}, engineSettings{kind: engineOllama})

	require.EqualError(t, err, "worker connection failed")
}

func TestEngineURL_RejectsNonlocalOrAmbiguousTargets(t *testing.T) {
	test := func(name, address string) {
		t.Run(name, func(t *testing.T) {
			_, err := engineURL(address)
			assert.Error(t, err)
		})
	}
	test("remote address", "http://192.0.2.1:11434")
	test("hostname requires DNS", "http://localhost:11434")
	test("embedded credentials", "http://user:password@127.0.0.1:11434")
	test("path prefix", "http://127.0.0.1:11434/admin")
	test("query string", "http://127.0.0.1:11434?target=remote")
	test("wrong scheme", "https://127.0.0.1:11434")
}

func TestEngineURL_AcceptsLiteralLoopbackTargets(t *testing.T) {
	test := func(name, address, host string) {
		t.Run(name, func(t *testing.T) {
			target, err := engineURL(address)
			require.NoError(t, err)
			assert.Equal(t, "http", target.Scheme)
			assert.Equal(t, host, target.Host)
			assert.Empty(t, target.Path)
		})
	}
	test("IPv4", "http://127.0.0.1:11434", "127.0.0.1:11434")
	test("IPv6", "http://[::1]:11434/", "[::1]:11434")
}

func TestReadToken_RejectsInvalidCredentials(t *testing.T) {
	test := func(name, input string) {
		t.Run(name, func(t *testing.T) {
			_, err := readToken(strings.NewReader(input))
			assert.Error(t, err)
		})
	}
	test("empty", "")
	test("short", "short")
	test("embedded whitespace", "a-long-token-containing a space-is-invalid")
	test("non-ASCII", "a-long-token-containing-é-is-invalid")
	test("oversized", strings.Repeat("x", 4097))
}
