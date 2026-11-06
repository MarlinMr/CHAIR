// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAmbientAssets_KeepDashboardAccessAndSecurityPolicy(t *testing.T) {
	test := func(name, path, method, origin string, public, authenticated bool, expected int) {
		t.Run(name, func(t *testing.T) {
			h := configuredDashboard(t, public, false, "")
			request := httptest.NewRequest(method, "https://hub"+path, nil)
			request.Header.Set("Origin", origin)
			if authenticated {
				request.SetBasicAuth("chair", testDashboardToken)
			}
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
			assert.Equal(t, "no-store", response.Header().Get("Cache-Control"))
			assert.Contains(t, response.Header().Get("Content-Security-Policy"), "script-src 'self'")
			assert.NotContains(t, response.Header().Get("Content-Security-Policy"), "unsafe-inline")
			for _, secret := range []string{testClientToken, testDashboardToken, testWorkerToken} {
				assert.NotContains(t, response.Body.String(), secret)
			}
		})
	}
	test("public script", "/dashboard/ambient.js", http.MethodGet, "", true, false, http.StatusOK)
	test("public stylesheet", "/dashboard/ambient.css", http.MethodGet, "", true, false, http.StatusOK)
	test("private script requires login", "/dashboard/ambient.js", http.MethodGet, "", false, false, http.StatusUnauthorized)
	test("private stylesheet requires login", "/dashboard/ambient.css", http.MethodGet, "", false, false, http.StatusUnauthorized)
	test("authenticated script", "/dashboard/ambient.js", http.MethodGet, "", false, true, http.StatusOK)
	test("foreign-origin script rejected", "/dashboard/ambient.js", http.MethodGet, "https://evil", true, false, http.StatusForbidden)
	test("query rejected", "/dashboard/ambient.js?token=secret", http.MethodGet, "", true, false, http.StatusForbidden)
	test("POST rejected", "/dashboard/ambient.js", http.MethodPost, "", true, false, http.StatusMethodNotAllowed)
}

func TestAmbientCreature_IsSharedAcrossOverviewAndInferenceTest(t *testing.T) {
	h := configuredDashboard(t, false, false, "")
	require.NoError(t, h.configureInferenceTest(true))
	test := func(name, path string) {
		t.Run(name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "https://hub"+path, nil)
			request.SetBasicAuth("chair", testDashboardToken)
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			require.Equal(t, http.StatusOK, response.Code)
			assert.Contains(t, response.Body.String(), `id="ambient-creature"`)
			assert.Contains(t, response.Body.String(), `aria-hidden="true"`)
			assert.Contains(t, response.Body.String(), `href="/dashboard/ambient.css"`)
			assert.Contains(t, response.Body.String(), `src="/dashboard/ambient.js" defer`)
			assert.NotContains(t, response.Body.String(), `id="ambient-mode"`)
			assert.NotContains(t, response.Body.String(), `id="ambient-controls"`)
		})
	}
	test("overview", "/")
	test("test inference", "/dashboard/test")
}
