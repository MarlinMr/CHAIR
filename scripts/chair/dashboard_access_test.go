// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func configuredDashboard(t *testing.T, public, reveal bool, downloads string) *hub {
	t.Helper()
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
	require.NoError(t, h.enableDashboard(testDashboardToken))
	require.NoError(t, h.configureDashboard(public, reveal, downloads))
	if h.downloadsRoot != nil {
		t.Cleanup(func() { assert.NoError(t, h.downloadsRoot.Close()) })
	}
	return h
}

func TestPublicDashboard_AllowsReadOnlyResourcesWithoutCredentials(t *testing.T) {
	directory := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(directory, "chair-linux-amd64.tar.gz"), []byte("test archive"), 0600))
	h := configuredDashboard(t, true, true, directory)
	for _, path := range []string{"/", "/dashboard/status", "/dashboard/style.css", "/dashboard/dashboard.js", "/dashboard/readme", "/dashboard/downloads/chair-linux-amd64.tar.gz"} {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()

			h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub"+path, nil))

			assert.Equal(t, http.StatusOK, response.Code)
			assert.Empty(t, response.Header().Get("WWW-Authenticate"))
			for _, secret := range []string{testClientToken, testWorkerToken, testDashboardToken} {
				assert.NotContains(t, response.Body.String(), secret)
			}
		})
	}
}

func TestPublicDashboard_KeepsPrivilegedEndpointsAuthenticated(t *testing.T) {
	h := configuredDashboard(t, true, true, "")
	for _, path := range []string{"/v1/models", "/worker", "/dashboard/access", "/dashboard/client-token"} {
		t.Run(path, func(t *testing.T) {
			method := http.MethodGet
			if path == "/dashboard/client-token" {
				method = http.MethodPost
			}
			request := httptest.NewRequest(method, "https://hub"+path, nil)
			request.Header.Set("Origin", "https://hub")
			request.Header.Set("X-Chair-Token-Reveal", "1")
			// Inference rejects browser Origins independently of authentication.
			if path == "/v1/models" || path == "/worker" {
				request.Header.Del("Origin")
			}
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, http.StatusUnauthorized, response.Code)
			assert.NotContains(t, response.Body.String(), testClientToken)
		})
	}
}

func TestPublicDashboard_CanBeEnabledWithoutADashboardCredential(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
	require.NoError(t, h.configureDashboard(true, false, ""))
	response := httptest.NewRecorder()

	h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub/", nil))

	assert.Equal(t, http.StatusOK, response.Code)
	assert.Contains(t, response.Body.String(), "Public dashboard")
	assert.NotContains(t, response.Body.String(), "Reveal client token")
}

func TestPublicDashboard_AllowsExternalLinksButRejectsCrossOriginReads(t *testing.T) {
	test := func(name, mode, origin string, expected int) {
		t.Run(name, func(t *testing.T) {
			h := configuredDashboard(t, true, false, "")
			request := httptest.NewRequest(http.MethodGet, "https://hub/", nil)
			request.Header.Set("Sec-Fetch-Site", "cross-site")
			request.Header.Set("Sec-Fetch-Mode", mode)
			request.Header.Set("Sec-Fetch-Dest", "document")
			request.Header.Set("Origin", origin)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
		})
	}
	test("external navigation", "navigate", "", http.StatusOK)
	test("cross-site fetch", "cors", "", http.StatusForbidden)
	test("foreign Origin", "navigate", "https://other.example", http.StatusForbidden)
}

func TestPublicDashboard_RequiresLoginBeforeOfferingTokenRevealControls(t *testing.T) {
	h := configuredDashboard(t, true, true, "")
	public := httptest.NewRecorder()
	h.ServeHTTP(public, httptest.NewRequest(http.MethodGet, "https://hub/", nil))
	assert.Contains(t, public.Body.String(), `href="/dashboard/access"`)
	assert.NotContains(t, public.Body.String(), `id="reveal-token"`)
	request := httptest.NewRequest(http.MethodGet, "https://hub/dashboard/access?refresh=off", nil)
	request.SetBasicAuth("chair", testDashboardToken)
	authenticated := httptest.NewRecorder()

	h.ServeHTTP(authenticated, request)

	assert.Equal(t, http.StatusOK, authenticated.Code)
	assert.Contains(t, authenticated.Body.String(), `id="reveal-token"`)
	assert.Contains(t, authenticated.Body.String(), `href="/dashboard/access?refresh=off"`)
	assert.NotContains(t, authenticated.Body.String(), testClientToken)
}

func TestClientTokenReveal_IsDisabledByDefault(t *testing.T) {
	h := configuredDashboard(t, true, false, "")
	for _, path := range []string{"/dashboard/client-token", "/dashboard/access"} {
		t.Run(path, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "https://hub"+path, nil)
			request.SetBasicAuth("chair", testDashboardToken)
			request.Header.Set("Origin", "https://hub")
			request.Header.Set("X-Chair-Token-Reveal", "1")
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, http.StatusNotFound, response.Code)
			assert.NotContains(t, response.Body.String(), testClientToken)
		})
	}
}

func TestClientTokenReveal_RequiresAuthenticatedExplicitSameOriginPost(t *testing.T) {
	test := func(name, method, address, origin, marker, password, fetchSite string, expected int) {
		t.Run(name, func(t *testing.T) {
			h := configuredDashboard(t, true, true, "")
			request := httptest.NewRequest(method, address, nil)
			request.SetBasicAuth("chair", password)
			request.Header.Set("Origin", origin)
			request.Header.Set("X-Chair-Token-Reveal", marker)
			request.Header.Set("Sec-Fetch-Site", fetchSite)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
			assert.NotContains(t, response.Body.String(), testClientToken)
		})
	}
	test("GET cannot reveal", http.MethodGet, "https://hub/dashboard/client-token", "https://hub", "1", testDashboardToken, "same-origin", http.StatusMethodNotAllowed)
	test("Origin required", http.MethodPost, "https://hub/dashboard/client-token", "", "1", testDashboardToken, "same-origin", http.StatusForbidden)
	test("cross-origin rejected", http.MethodPost, "https://hub/dashboard/client-token", "https://other.example", "1", testDashboardToken, "cross-site", http.StatusForbidden)
	test("cross-site rejected even with matching Origin", http.MethodPost, "https://hub/dashboard/client-token", "https://hub", "1", testDashboardToken, "cross-site", http.StatusForbidden)
	test("explicit action header required", http.MethodPost, "https://hub/dashboard/client-token", "https://hub", "", testDashboardToken, "same-origin", http.StatusForbidden)
	test("dashboard credential required", http.MethodPost, "https://hub/dashboard/client-token", "https://hub", "1", "", "same-origin", http.StatusUnauthorized)
	test("worker credential rejected", http.MethodPost, "https://hub/dashboard/client-token", "https://hub", "1", testWorkerToken, "same-origin", http.StatusUnauthorized)
	test("client credential rejected", http.MethodPost, "https://hub/dashboard/client-token", "https://hub", "1", testClientToken, "same-origin", http.StatusUnauthorized)
	test("query strings rejected", http.MethodPost, "https://hub/dashboard/client-token?show=1", "https://hub", "1", testDashboardToken, "same-origin", http.StatusForbidden)
	test("plaintext rejected", http.MethodPost, "http://hub/dashboard/client-token", "https://hub", "1", testDashboardToken, "same-origin", http.StatusForbidden)
}

func TestClientTokenReveal_ReturnsOnlyTheClientTokenOnExplicitRequest(t *testing.T) {
	h := configuredDashboard(t, true, true, "")
	var logs bytes.Buffer
	h.logger = slog.New(slog.NewTextHandler(&logs, nil))
	request := httptest.NewRequest(http.MethodPost, "https://hub/dashboard/client-token", nil)
	request.SetBasicAuth("chair", testDashboardToken)
	request.Header.Set("Origin", "https://hub")
	request.Header.Set("X-Chair-Token-Reveal", "1")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	require.Equal(t, http.StatusOK, response.Code)
	var result map[string]string
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	assert.Equal(t, map[string]string{"client_token": testClientToken}, result)
	assert.Equal(t, "application/json", response.Header().Get("Content-Type"))
	assert.Equal(t, "no-store", response.Header().Get("Cache-Control"))
	assert.Equal(t, "no-cache", response.Header().Get("Pragma"))
	assert.Empty(t, response.Header().Get("Access-Control-Allow-Origin"))
	assert.NotContains(t, logs.String(), testClientToken)
}

func TestDashboardOptions_RejectsUnsafeOrUnavailableConfiguration(t *testing.T) {
	test := func(name string, public, reveal bool, directory string) {
		t.Run(name, func(t *testing.T) {
			h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
			assert.Error(t, h.configureDashboard(public, reveal, directory))
		})
	}
	test("reveal needs authentication", false, true, "")
	test("public reveal still needs authentication", true, true, "")
	test("downloads need an enabled dashboard", false, false, t.TempDir())
	test("downloads directory must exist", true, false, filepath.Join(t.TempDir(), "missing"))
}

func TestDashboardDownloads_ServesOnlyApprovedRegularArchives(t *testing.T) {
	directory := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(directory, "chair-linux-amd64.tar.gz"), []byte("test archive bytes"), 0600))
	require.NoError(t, os.WriteFile(filepath.Join(directory, "SHA256SUMS"), []byte("test checksums"), 0600))
	require.NoError(t, os.WriteFile(filepath.Join(directory, "client.token"), []byte(testClientToken), 0600))
	require.NoError(t, os.Mkdir(filepath.Join(directory, "chair-linux-arm64.tar.gz"), 0700))
	h := configuredDashboard(t, true, false, directory)
	assert.Equal(t, []dashboardDownload{{Name: "chair-linux-amd64.tar.gz", Label: "Linux · x64"}, {Name: "SHA256SUMS", Label: "SHA-256 checksums"}}, h.dashboardDownloads())
	response := httptest.NewRecorder()

	h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub/dashboard/downloads/chair-linux-amd64.tar.gz", nil))

	assert.Equal(t, http.StatusOK, response.Code)
	assert.Equal(t, "test archive bytes", response.Body.String())
	assert.Equal(t, `attachment; filename="chair-linux-amd64.tar.gz"`, response.Header().Get("Content-Disposition"))
	assert.Equal(t, "application/octet-stream", response.Header().Get("Content-Type"))
	for _, name := range []string{"client.token", "", "../client.token", "%2e%2e/client.token", "chair-linux-arm64.tar.gz", "chair-windows-amd64.zip"} {
		t.Run(name, func(t *testing.T) {
			response := httptest.NewRecorder()
			h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub/dashboard/downloads/"+name, nil))
			assert.Equal(t, http.StatusNotFound, response.Code)
			assert.NotContains(t, response.Body.String(), testClientToken)
		})
	}
}

func TestDashboardDownloads_DoesNotServeOrAdvertiseSymlinks(t *testing.T) {
	directory := t.TempDir()
	secret := filepath.Join(t.TempDir(), "client.token")
	require.NoError(t, os.WriteFile(secret, []byte(testClientToken), 0600))
	err := os.Symlink(secret, filepath.Join(directory, "chair-linux-amd64.tar.gz"))
	if err != nil && runtime.GOOS == "windows" {
		t.Skipf("creating a test symlink requires Windows symlink privileges: %v", err)
	}
	require.NoError(t, err)
	h := configuredDashboard(t, true, false, directory)
	response := httptest.NewRecorder()

	h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub/dashboard/downloads/chair-linux-amd64.tar.gz", nil))

	assert.Equal(t, http.StatusNotFound, response.Code)
	assert.Empty(t, h.dashboardDownloads())
	assert.NotContains(t, response.Body.String(), testClientToken)
}

func TestDashboardDownloads_RequiresLoginUnlessPublicAccessIsEnabled(t *testing.T) {
	directory := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(directory, "chair-linux-amd64.tar.gz"), []byte("test archive"), 0600))
	h := configuredDashboard(t, false, false, directory)
	for _, path := range []string{"/dashboard/readme", "/dashboard/downloads/chair-linux-amd64.tar.gz"} {
		t.Run(path, func(t *testing.T) {
			response := httptest.NewRecorder()
			h.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "https://hub"+path, nil))
			assert.Equal(t, http.StatusUnauthorized, response.Code)
		})
	}
}
