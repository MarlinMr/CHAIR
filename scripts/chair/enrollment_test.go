// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testEnrollmentToken = "enrollment-test-credential-not-for-real-use"

type enrolledWorker struct {
	WorkerID    string `json:"worker_id"`
	WorkerToken string `json:"worker_token"`
}

func startEnrollmentFixture(t *testing.T, public bool) (transportFixture, string) {
	t.Helper()
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, defaultMaxConcurrentRequests)
	h.logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	require.NoError(t, h.enableDashboard(testDashboardToken))
	require.NoError(t, h.configureDashboard(public, false, ""))
	path := filepath.Join(t.TempDir(), "workers.json")
	require.NoError(t, os.WriteFile(path, []byte(`{"workers":[{"id":"gpu-a","token":"`+testWorkerToken+`"}]}`), 0600))
	require.NoError(t, h.enableEnrollment(testEnrollmentToken, path))
	t.Cleanup(func() { assert.NoError(t, h.enrollment.root.Close()) })
	server := httptest.NewTLSServer(h)
	t.Cleanup(server.Close)
	t.Cleanup(func() { assert.NoError(t, h.Close()) })
	return transportFixture{hub: h, server: server, client: server.Client()}, path
}

func enrollmentRequest(t *testing.T, fixture transportFixture, id, token string) *http.Request {
	t.Helper()
	body, err := json.Marshal(struct {
		ID    string `json:"worker_id"`
		Token string `json:"enrollment_token"`
	}{ID: id, Token: token})
	require.NoError(t, err)
	request, err := http.NewRequest(http.MethodPost, fixture.server.URL+"/join", bytes.NewReader(body))
	require.NoError(t, err)
	request.Header.Set("Origin", fixture.server.URL)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Chair-Enroll", "1")
	return request
}

func enrollWorker(t *testing.T, fixture transportFixture, id string) enrolledWorker {
	t.Helper()
	response, err := fixture.client.Do(enrollmentRequest(t, fixture, id, testEnrollmentToken))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusCreated, response.StatusCode)
	assert.Equal(t, "no-store", response.Header.Get("Cache-Control"))
	var result enrolledWorker
	require.NoError(t, json.NewDecoder(response.Body).Decode(&result))
	return result
}

func readRegistry(t *testing.T, path string) map[string]string {
	t.Helper()
	file, err := os.Open(path)
	require.NoError(t, err)
	defer func() { assert.NoError(t, file.Close()) }()
	tokens, err := readWorkerTokens(file, testClientToken)
	require.NoError(t, err)
	return tokens
}

func TestEnrollment_GeneratedWorkerCanServeInferenceWithoutHubRestart(t *testing.T) {
	fixture, _ := startEnrollmentFixture(t, true)

	worker := enrollWorker(t, fixture, "new-gpu")

	assert.Equal(t, "new-gpu", worker.WorkerID)
	assert.Len(t, worker.WorkerToken, 64)
	_, err := hex.DecodeString(worker.WorkerToken)
	require.NoError(t, err)
	for _, credential := range []string{testClientToken, testDashboardToken, testEnrollmentToken, testWorkerToken} {
		assert.NotEqual(t, credential, worker.WorkerToken)
	}
	startTestWorker(t, fixture, worker.WorkerID, worker.WorkerToken, testEngine(t, []string{"test-model"}, workerReply(t, worker.WorkerID)))
	assert.Equal(t, "new-gpu", servedBy(t, fixture, "test-model"))
	assert.Equal(t, 2, dashboardData(t, fixture).Authorized)
}

func TestEnrollment_PersistsWorkerCredentialsForTheNextHub(t *testing.T) {
	fixture, path := startEnrollmentFixture(t, false)
	worker := enrollWorker(t, fixture, "persistent-gpu")

	tokens := readRegistry(t, path)

	assert.Equal(t, map[string]string{testWorkerID: testWorkerToken, "persistent-gpu": worker.WorkerToken}, tokens)
	info, err := os.Stat(path)
	require.NoError(t, err)
	if runtime.GOOS != "windows" {
		assert.Equal(t, os.FileMode(0600), info.Mode().Perm())
	}
	restarted := startHubFixture(t, tokens, defaultMaxConcurrentRequests)
	startTestWorker(t, restarted, worker.WorkerID, worker.WorkerToken, testEngine(t, []string{"test-model"}, workerReply(t, worker.WorkerID)))
	assert.Equal(t, "persistent-gpu", servedBy(t, restarted, "test-model"))
}

func TestEnrollment_DoesNotReplaceAnExistingWorkerOrReturnItsToken(t *testing.T) {
	fixture, path := startEnrollmentFixture(t, true)
	worker := enrollWorker(t, fixture, "new-gpu")
	before, err := os.ReadFile(path)
	require.NoError(t, err)
	for _, id := range []string{testWorkerID, worker.WorkerID} {
		t.Run(id, func(t *testing.T) {
			response, err := fixture.client.Do(enrollmentRequest(t, fixture, id, testEnrollmentToken))
			require.NoError(t, err)
			body, err := io.ReadAll(response.Body)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, http.StatusConflict, response.StatusCode)
			assert.NotContains(t, string(body), testWorkerToken)
			assert.NotContains(t, string(body), worker.WorkerToken)
		})
	}
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, before, after)
}

func TestEnrollment_RequiresTheSharedEnrollmentCredential(t *testing.T) {
	fixture, path := startEnrollmentFixture(t, false)
	test := func(name, token string) {
		t.Run(name, func(t *testing.T) {
			request := enrollmentRequest(t, fixture, "untrusted-gpu", token)
			request.SetBasicAuth("chair", testDashboardToken)
			response, err := fixture.client.Do(request)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, http.StatusUnauthorized, response.StatusCode)
		})
	}
	test("missing credential", "")
	test("wrong credential", "wrong")
	test("inference credential", testClientToken)
	test("dashboard credential", testDashboardToken)
	test("worker credential", testWorkerToken)
	assert.Equal(t, map[string]string{testWorkerID: testWorkerToken}, readRegistry(t, path))
}

func TestEnrollment_KeepsTokensOutOfPagesStatusAndLogs(t *testing.T) {
	fixture, _ := startEnrollmentFixture(t, true)
	var logs bytes.Buffer
	fixture.hub.logger = slog.New(slog.NewTextHandler(&logs, nil))
	worker := enrollWorker(t, fixture, "private-gpu")
	for _, path := range []string{"/", "/join", "/dashboard/enrollment.js", "/dashboard/status"} {
		t.Run(path, func(t *testing.T) {
			response, err := fixture.client.Get(fixture.server.URL + path)
			require.NoError(t, err)
			body, err := io.ReadAll(response.Body)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, http.StatusOK, response.StatusCode)
			for _, secret := range []string{worker.WorkerToken, testEnrollmentToken, testClientToken, testDashboardToken, testWorkerToken} {
				assert.NotContains(t, string(body), secret)
				assert.NotContains(t, logs.String(), secret)
			}
		})
	}
}

func TestEnrollment_RejectsUnsafeOrMalformedRequests(t *testing.T) {
	test := func(name, method, address, origin, marker, contentType, body string, expected int) {
		t.Run(name, func(t *testing.T) {
			fixture, _ := startEnrollmentFixture(t, true)
			request := httptest.NewRequest(method, address, strings.NewReader(body))
			request.Header.Set("Origin", origin)
			request.Header.Set("X-Chair-Enroll", marker)
			request.Header.Set("Content-Type", contentType)
			response := httptest.NewRecorder()

			fixture.hub.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
			assert.NotContains(t, response.Body.String(), testEnrollmentToken)
		})
	}
	valid := `{"worker_id":"new-gpu","enrollment_token":"` + testEnrollmentToken + `"}`
	test("TLS required", http.MethodPost, "http://hub/join", "https://hub", "1", "application/json", valid, http.StatusForbidden)
	test("Origin required", http.MethodPost, "https://hub/join", "", "1", "application/json", valid, http.StatusForbidden)
	test("foreign Origin", http.MethodPost, "https://hub/join", "https://other.example", "1", "application/json", valid, http.StatusForbidden)
	test("explicit action required", http.MethodPost, "https://hub/join", "https://hub", "", "application/json", valid, http.StatusForbidden)
	test("query rejected", http.MethodPost, "https://hub/join?token=hidden", "https://hub", "1", "application/json", valid, http.StatusForbidden)
	test("form submission rejected", http.MethodPost, "https://hub/join", "https://hub", "1", "application/x-www-form-urlencoded", valid, http.StatusUnsupportedMediaType)
	test("malformed JSON", http.MethodPost, "https://hub/join", "https://hub", "1", "application/json", "{", http.StatusBadRequest)
	test("unknown field", http.MethodPost, "https://hub/join", "https://hub", "1", "application/json", `{"worker_token":"not-accepted"}`, http.StatusBadRequest)
	test("invalid worker ID", http.MethodPost, "https://hub/join", "https://hub", "1", "application/json", `{"worker_id":"gpu/../other","enrollment_token":"`+testEnrollmentToken+`"}`, http.StatusBadRequest)
	test("oversized body", http.MethodPost, "https://hub/join", "https://hub", "1", "application/json", strings.Repeat("a", (16<<10)+1), http.StatusBadRequest)
}

func TestEnrollment_LeavesPrivateClusterStatusProtected(t *testing.T) {
	fixture, _ := startEnrollmentFixture(t, false)
	for _, path := range []string{"/join", "/dashboard/enrollment.js", "/dashboard/style.css"} {
		response, err := fixture.client.Get(fixture.server.URL + path)
		require.NoError(t, err)
		require.NoError(t, response.Body.Close())
		assert.Equal(t, http.StatusOK, response.StatusCode)
	}
	response, err := fixture.client.Get(fixture.server.URL + "/dashboard/status")
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusUnauthorized, response.StatusCode)
}

func TestEnrollment_RefusesToOverwriteExternalRegistryChanges(t *testing.T) {
	fixture, path := startEnrollmentFixture(t, true)
	replacement := []byte(`{"workers":[]}`)
	require.NoError(t, os.WriteFile(path, replacement, 0600))

	response, err := fixture.client.Do(enrollmentRequest(t, fixture, "new-gpu", testEnrollmentToken))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusServiceUnavailable, response.StatusCode)
	assert.Equal(t, 1, fixture.hub.dashboardSnapshot().Authorized)
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, replacement, after)
}

func TestEnrollment_DoesNotAuthorizeWorkersWhenRegistryReplacementFails(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("requires POSIX directory permissions enforced for a non-root user")
	}
	fixture, path := startEnrollmentFixture(t, true)
	before, err := os.ReadFile(path)
	require.NoError(t, err)
	directory := filepath.Dir(path)
	t.Cleanup(func() { assert.NoError(t, os.Chmod(directory, 0700)) })
	require.NoError(t, os.Chmod(directory, 0500))

	response, err := fixture.client.Do(enrollmentRequest(t, fixture, "unsaved-gpu", testEnrollmentToken))
	require.NoError(t, err)
	body, err := io.ReadAll(response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusServiceUnavailable, response.StatusCode)
	assert.NotContains(t, string(body), "worker_token")
	assert.Equal(t, 1, fixture.hub.dashboardSnapshot().Authorized)
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Equal(t, before, after)
}

func TestEnrollment_BoundsRequestAttempts(t *testing.T) {
	fixture, _ := startEnrollmentFixture(t, true)
	for range enrollmentAttemptsPerMinute {
		response, err := fixture.client.Do(enrollmentRequest(t, fixture, "new-gpu", "wrong"))
		require.NoError(t, err)
		require.NoError(t, response.Body.Close())
		require.Equal(t, http.StatusUnauthorized, response.StatusCode)
	}
	response, err := fixture.client.Do(enrollmentRequest(t, fixture, "new-gpu", testEnrollmentToken))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	assert.Equal(t, http.StatusTooManyRequests, response.StatusCode)
	assert.Equal(t, "60", response.Header.Get("Retry-After"))
}

func TestEnrollment_SerializesConcurrentRegistrationsWithoutLosingWorkers(t *testing.T) {
	fixture, path := startEnrollmentFixture(t, true)
	type outcome struct {
		id     string
		worker enrolledWorker
		err    error
		status int
	}
	results := make(chan outcome, 8)
	var group sync.WaitGroup
	for index := range 8 {
		id := fmt.Sprintf("new-gpu-%d", index)
		request := enrollmentRequest(t, fixture, id, testEnrollmentToken)
		probe, err := http.NewRequest(http.MethodGet, fixture.server.URL+workerPath, nil)
		require.NoError(t, err)
		probe.Header.Set(workerIDHeader, id)
		group.Go(func() {
			response, err := fixture.client.Do(request)
			if err != nil {
				results <- outcome{id: id, err: err}
				return
			}
			var worker enrolledWorker
			err = errors.Join(json.NewDecoder(response.Body).Decode(&worker), response.Body.Close())
			if err == nil && response.StatusCode == http.StatusCreated {
				// Authentication can run while other registrations are persisting.
				probe.Header.Set("Authorization", "Bearer "+worker.WorkerToken)
				checked, probeErr := fixture.client.Do(probe)
				if probeErr != nil {
					err = probeErr
				} else {
					err = checked.Body.Close()
					if checked.StatusCode != http.StatusUpgradeRequired {
						err = errors.Join(err, fmt.Errorf("new worker did not pass authentication"))
					}
				}
			}
			results <- outcome{id: id, worker: worker, err: err, status: response.StatusCode}
		})
	}
	group.Wait()
	close(results)
	tokens := readRegistry(t, path)
	assert.Len(t, tokens, 9)
	seen := map[string]bool{testWorkerToken: true}
	for result := range results {
		require.NoError(t, result.err)
		require.Equal(t, http.StatusCreated, result.status)
		assert.Equal(t, result.id, result.worker.WorkerID)
		assert.Equal(t, result.worker.WorkerToken, tokens[result.id])
		assert.False(t, seen[result.worker.WorkerToken])
		seen[result.worker.WorkerToken] = true
	}
	assert.Equal(t, 9, fixture.hub.dashboardSnapshot().Authorized)
}

func TestEnrollment_IsDisabledByDefault(t *testing.T) {
	h := configuredDashboard(t, true, false, "")
	for _, path := range []string{"/join", "/dashboard/enrollment.js"} {
		response := httptest.NewRecorder()
		h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub"+path, nil))
		assert.Equal(t, http.StatusNotFound, response.Code)
	}
}

func TestEnrollment_RejectsReusedCredentials(t *testing.T) {
	test := func(name, token string) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			require.NoError(t, h.enableDashboard(testDashboardToken))
			assert.Error(t, h.enableEnrollment(token, filepath.Join(t.TempDir(), "workers.json")))
		})
	}
	test("client credential", testClientToken)
	test("worker credential", testWorkerToken)
	test("dashboard credential", testDashboardToken)
	test("empty credential", "")
}

func TestEnrollment_CanStartWithNoWorkersAndNoPublicDashboard(t *testing.T) {
	path := filepath.Join(t.TempDir(), "workers.json")
	require.NoError(t, os.WriteFile(path, []byte(`{"workers":[]}`), 0600))
	h := newHub(testClientToken, readRegistry(t, path), 1)
	require.NoError(t, h.enableEnrollment(testEnrollmentToken, path))
	t.Cleanup(func() { assert.NoError(t, h.enrollment.root.Close()) })
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "https://hub/join", strings.NewReader(`{"worker_id":"first-gpu","enrollment_token":"`+testEnrollmentToken+`"}`))
	request.Header.Set("Origin", "https://hub")
	request.Header.Set("X-Chair-Enroll", "1")
	request.Header.Set("Content-Type", "application/json")

	h.ServeHTTP(response, request)

	require.Equal(t, http.StatusCreated, response.Code)
	var worker enrolledWorker
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &worker))
	assert.Equal(t, map[string]string{"first-gpu": worker.WorkerToken}, readRegistry(t, path))
	private := httptest.NewRecorder()
	h.ServeHTTP(private, httptest.NewRequest(http.MethodGet, "https://hub/dashboard/status", nil))
	assert.Equal(t, http.StatusNotFound, private.Code)
}

func TestEnrollment_RejectsRegistrationsAboveWorkerCapacity(t *testing.T) {
	fixture, _ := startEnrollmentFixture(t, true)
	fixture.hub.mu.Lock()
	for index := range 255 {
		fixture.hub.workerTokens[fmt.Sprintf("existing-%d", index)] = fmt.Sprintf("existing-worker-credential-number-%d", index)
	}
	fixture.hub.mu.Unlock()

	response, err := fixture.client.Do(enrollmentRequest(t, fixture, "one-too-many", testEnrollmentToken))
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())

	assert.Equal(t, http.StatusConflict, response.StatusCode)
	assert.Equal(t, 256, fixture.hub.dashboardSnapshot().Authorized)
}

func TestEnrollment_RequestLimitResetsAfterOneMinute(t *testing.T) {
	state := workerEnrollment{}
	now := time.Now()
	for range enrollmentAttemptsPerMinute {
		require.True(t, state.allowAttempt(now))
	}
	assert.False(t, state.allowAttempt(now.Add(59*time.Second)))
	assert.True(t, state.allowAttempt(now.Add(time.Minute)))
}

func TestEnrollment_SharedCredentialCannotActAsWorkerOrInferenceClient(t *testing.T) {
	fixture, _ := startEnrollmentFixture(t, true)
	for _, path := range []string{"/worker", "/v1/models"} {
		t.Run(path, func(t *testing.T) {
			request, err := http.NewRequest(http.MethodGet, fixture.server.URL+path, nil)
			require.NoError(t, err)
			request.Header.Set("Authorization", "Bearer "+testEnrollmentToken)
			request.Header.Set(workerIDHeader, testWorkerID)
			response, err := fixture.client.Do(request)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, http.StatusUnauthorized, response.StatusCode)
		})
	}
}

func TestEnrollment_RejectsCredentialsPastedIntoPublicWorkerIDs(t *testing.T) {
	fixture, path := startEnrollmentFixture(t, true)
	var logs bytes.Buffer
	fixture.hub.logger = slog.New(slog.NewTextHandler(&logs, nil))
	test := func(name, id string) {
		t.Run(name, func(t *testing.T) {
			response, err := fixture.client.Do(enrollmentRequest(t, fixture, id, testEnrollmentToken))
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, http.StatusBadRequest, response.StatusCode)
			assert.NotContains(t, logs.String(), id)
		})
	}
	test("enrollment credential", testEnrollmentToken)
	test("client credential", testClientToken)
	test("dashboard credential", testDashboardToken)
	test("worker credential", testWorkerToken)
	assert.Equal(t, map[string]string{testWorkerID: testWorkerToken}, readRegistry(t, path))
}
