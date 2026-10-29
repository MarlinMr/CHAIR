// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
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

const testDashboardToken = "dashboard-test-credential-not-for-real-use"

func dashboardRequest(t *testing.T, fixture transportFixture, path string) *http.Request {
	t.Helper()
	request, err := http.NewRequest(http.MethodGet, fixture.server.URL+path, nil)
	require.NoError(t, err)
	request.SetBasicAuth("chair", testDashboardToken)
	return request
}

func dashboardData(t *testing.T, fixture transportFixture) dashboardView {
	t.Helper()
	response, err := fixture.client.Do(dashboardRequest(t, fixture, "/dashboard/status"))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	var view dashboardView
	require.NoError(t, json.NewDecoder(response.Body).Decode(&view))
	return view
}

func TestDashboard_DisabledUntilSeparateCredentialIsConfigured(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
	for _, path := range []string{"/", "/dashboard/status", "/dashboard/style.css"} {
		t.Run(path, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "https://hub"+path, nil)
			request.SetBasicAuth("chair", testDashboardToken)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, http.StatusNotFound, response.Code)
		})
	}
}

func TestDashboard_RequiresDedicatedReadOnlyCredential(t *testing.T) {
	test := func(name, username, password string) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			require.NoError(t, h.enableDashboard(testDashboardToken))
			request := httptest.NewRequest(http.MethodGet, "https://hub/dashboard/status", nil)
			request.SetBasicAuth(username, password)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, http.StatusUnauthorized, response.Code)
			assert.Equal(t, `Basic realm="CHAIR dashboard", charset="UTF-8"`, response.Header().Get("WWW-Authenticate"))
			assert.NotContains(t, response.Body.String(), testWorkerID)
		})
	}
	test("missing password", "chair", "")
	test("wrong username", "other", testDashboardToken)
	test("inference token is not a dashboard credential", "chair", testClientToken)
	test("worker token is not a dashboard credential", "chair", testWorkerToken)
}

func TestDashboardToken_CannotReuseInferenceOrWorkerCredentials(t *testing.T) {
	test := func(name, token string) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			assert.Error(t, h.enableDashboard(token))
		})
	}
	test("empty credential", "")
	test("inference credential", testClientToken)
	test("worker credential", testWorkerToken)
}

func TestDashboardCredential_DoesNotGrantInferenceAccess(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
	require.NoError(t, h.enableDashboard(testDashboardToken))
	request := httptest.NewRequest(http.MethodPost, "https://hub/v1/chat/completions", strings.NewReader(`{"model":"test-model"}`))
	request.Header.Set("Authorization", "Bearer "+testDashboardToken)
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusUnauthorized, response.Code)
}

func TestDashboard_ShowsAuthorizedWorkersAndFreshModels(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{"gpu-a": testWorkerToken, "gpu-b": secondWorkerToken}, 7)
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))
	startTestWorker(t, fixture, "gpu-a", testWorkerToken,
		testEngine(t, []string{"qwen:7b", "llama:8b"}, workerReply(t, "gpu-a")))

	view := dashboardData(t, fixture)

	assert.Equal(t, 2, view.Authorized)
	assert.Equal(t, 1, view.Connected)
	assert.Equal(t, 1, view.Ready)
	assert.Equal(t, 2, view.Models)
	assert.Equal(t, 7, view.Limit)
	require.Len(t, view.Workers, 2)
	assert.Equal(t, "gpu-a", view.Workers[0].ID)
	assert.Equal(t, "Ready", view.Workers[0].Status)
	assert.Equal(t, []string{"llama:8b", "qwen:7b"}, view.Workers[0].Models)
	assert.Equal(t, "gpu-b", view.Workers[1].ID)
	assert.Equal(t, "Disconnected", view.Workers[1].Status)
	assert.Empty(t, view.Workers[1].Models)
	assert.Empty(t, view.Jobs)
}

func TestDashboard_ShowsActiveJobMetadataWithoutInferenceContents(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 4)
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))
	startTestWorker(t, fixture, testWorkerID, testWorkerToken,
		testEngine(t, []string{"test-model:latest"}, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, err := io.WriteString(w, "data: PRIVATE_GENERATED_RESPONSE\n\n")
			assert.NoError(t, err)
			assert.NoError(t, http.NewResponseController(w).Flush())
			<-r.Context().Done()
		}))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	request := openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"test-model","messages":[{"role":"user","content":"PRIVATE_PROMPT_CONTENT"}],"stream":true}`)
	response, err := fixture.client.Do(request.WithContext(ctx))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()

	view := dashboardData(t, fixture)

	require.Len(t, view.Jobs, 1)
	assert.Equal(t, uint64(1), view.Jobs[0].ID)
	assert.Equal(t, testWorkerID, view.Jobs[0].WorkerID)
	assert.Equal(t, "test-model", view.Jobs[0].Model)
	assert.Equal(t, "/v1/chat/completions", view.Jobs[0].Path)
	assert.False(t, view.Jobs[0].StartedAt.IsZero())
	assert.Equal(t, 1, view.Workers[0].Inflight)
	jsonBody, err := json.Marshal(view)
	require.NoError(t, err)
	for _, secret := range []string{testClientToken, testWorkerToken, testDashboardToken, "PRIVATE_PROMPT_CONTENT", "PRIVATE_GENERATED_RESPONSE"} {
		assert.NotContains(t, string(jsonBody), secret)
	}
	cancel()
	require.Eventually(t, func() bool { return len(fixture.hub.dashboardSnapshot().Jobs) == 0 }, 3*time.Second, 10*time.Millisecond)
}

func TestDashboard_DoesNotRetainCompletedJobs(t *testing.T) {
	fixture := startTransport(t, workerReply(t, testWorkerID))
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))

	require.Equal(t, testWorkerID, servedBy(t, fixture, "test-model"))

	assert.Empty(t, dashboardData(t, fixture).Jobs)
}

func TestDashboard_HTMLAndStylesAreEmbeddedAndProtected(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 1)
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))
	response, err := fixture.client.Do(dashboardRequest(t, fixture, "/"))
	require.NoError(t, err)
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Equal(t, http.StatusOK, response.StatusCode)
	assert.Equal(t, "text/html; charset=utf-8", response.Header.Get("Content-Type"))
	assert.Equal(t, "no-store", response.Header.Get("Cache-Control"))
	assert.Contains(t, response.Header.Get("Content-Security-Policy"), "default-src 'none'")
	assert.Contains(t, response.Header.Get("Content-Security-Policy"), "frame-ancestors 'none'")
	assert.Contains(t, string(body), "CHAIR")
	assert.Contains(t, string(body), "Active jobs")
	assert.Contains(t, string(body), `http-equiv="refresh" content="3"`)
	assert.Contains(t, string(body), "No jobs in flight")
	stylesheet, err := fixture.client.Do(dashboardRequest(t, fixture, "/dashboard/style.css"))
	require.NoError(t, err)
	require.NoError(t, stylesheet.Body.Close())
	assert.Equal(t, http.StatusOK, stylesheet.StatusCode)
	assert.Equal(t, "text/css; charset=utf-8", stylesheet.Header.Get("Content-Type"))
}

func TestDashboard_CanPauseAutomaticRefresh(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 1)
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))

	response, err := fixture.client.Do(dashboardRequest(t, fixture, "/?refresh=off"))
	require.NoError(t, err)
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusOK, response.StatusCode)
	assert.Contains(t, string(body), "Resume live refresh")
	assert.NotContains(t, string(body), `http-equiv="refresh"`)
}

func TestDashboard_RejectsPlaintextCrossOriginAndMutationRequests(t *testing.T) {
	test := func(name, method, address, origin, fetchSite string, expected int) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			require.NoError(t, h.enableDashboard(testDashboardToken))
			request := httptest.NewRequest(method, address, nil)
			request.SetBasicAuth("chair", testDashboardToken)
			request.Header.Set("Origin", origin)
			request.Header.Set("Sec-Fetch-Site", fetchSite)
			request.Header.Set("X-Forwarded-Proto", "https")
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
		})
	}
	test("plaintext cannot spoof TLS", http.MethodGet, "http://hub/", "", "", http.StatusForbidden)
	test("cross-origin read", http.MethodGet, "https://hub/dashboard/status", "https://other.example", "", http.StatusForbidden)
	test("cross-site read", http.MethodGet, "https://hub/", "", "cross-site", http.StatusForbidden)
	test("credentials in query rejected", http.MethodGet, "https://hub/?token=secret", "", "", http.StatusForbidden)
	test("mutating requests rejected", http.MethodPost, "https://hub/", "https://hub", "same-origin", http.StatusMethodNotAllowed)
	test("same-origin browser read", http.MethodGet, "https://hub/dashboard/status", "https://hub", "same-origin", http.StatusOK)
	test("default TLS port normalized", http.MethodGet, "https://hub:443/dashboard/status", "https://hub", "same-origin", http.StatusOK)
}

func TestDashboard_EscapesUntrustedModelNames(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 1)
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))
	startTestWorker(t, fixture, testWorkerID, testWorkerToken,
		testEngine(t, []string{"model<script>alert(1)</script>"}, workerReply(t, testWorkerID)))

	response, err := fixture.client.Do(dashboardRequest(t, fixture, "/"))
	require.NoError(t, err)
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusOK, response.StatusCode)
	assert.NotContains(t, string(body), "<script>")
	assert.Contains(t, string(body), "&lt;script&gt;")
}

func TestDashboard_WithdrawsModelsWhenWorkerDisconnects(t *testing.T) {
	fixture := startTransport(t, workerReply(t, testWorkerID))
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))

	fixture.cancel()

	require.Eventually(t, func() bool {
		return fixture.hub.dashboardSnapshot().Connected == 0
	}, 3*time.Second, 10*time.Millisecond)
	view := dashboardData(t, fixture)
	assert.Zero(t, view.Ready)
	assert.Zero(t, view.Models)
	require.Len(t, view.Workers, 1)
	assert.Equal(t, "Disconnected", view.Workers[0].Status)
	assert.Empty(t, view.Workers[0].Models)
}

func TestDashboard_DoesNotRetainFailedJobs(t *testing.T) {
	fixture := startTransport(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "engine rejected request", http.StatusBadRequest)
	})
	require.NoError(t, fixture.hub.enableDashboard(testDashboardToken))

	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"test-model","messages":[]}`))
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Equal(t, http.StatusBadRequest, response.StatusCode)

	require.Eventually(t, func() bool {
		return len(fixture.hub.dashboardSnapshot().Jobs) == 0
	}, 3*time.Second, 10*time.Millisecond)
	assert.Empty(t, dashboardData(t, fixture).Jobs)
}
