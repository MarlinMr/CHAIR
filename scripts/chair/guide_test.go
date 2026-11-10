// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGuideOrigin_PreservesValidHostnamesAndPorts(t *testing.T) {
	test := func(name, authority, expected string) {
		t.Run(name, func(t *testing.T) {
			origin, err := guideOrigin(authority)
			require.NoError(t, err)
			assert.Equal(t, expected, origin)
		})
	}
	test("standard HTTPS port", "hub.example.org", "https://hub.example.org")
	test("explicit port", "hub.example.org:8443", "https://hub.example.org:8443")
	test("IPv4", "192.0.2.10:443", "https://192.0.2.10:443")
	test("IPv6", "[2001:db8::1]:8443", "https://[2001:db8::1]:8443")
	test("IPv6 default port", "[::1]", "https://[::1]")
	test("local name", "localhost:8443", "https://localhost:8443")
}

func TestGuideOrigin_RejectsUnsafeOrAmbiguousAuthorities(t *testing.T) {
	test := func(name, authority string) {
		t.Run(name, func(t *testing.T) { _, err := guideOrigin(authority); assert.Error(t, err) })
	}
	test("empty host", "")
	test("shell interpolation", "$(touch.bad):8443")
	test("shell quote", "hub'evil:8443")
	test("HTML injection", `<script>alert(1)</script>`)
	test("credentials", "user@hub:8443")
	test("path", "hub/evil")
	test("query", "hub?evil=1")
	test("fragment", "hub#evil")
	test("empty port", "hub:")
	test("out of range port", "hub:65536")
	test("zero port", "hub:0")
	test("unbracketed IPv6", "::1")
	test("oversized DNS label", strings.Repeat("a", 64)+".test")
}

func TestGuide_UsesCurrentOriginWithoutCredentialsOrForwardedHeaders(t *testing.T) {
	h := configuredDashboard(t, true, true, "")
	request := httptest.NewRequest(http.MethodGet, "https://hub.example.org:9443/dashboard/readme.html", nil)
	request.Header.Set("X-Forwarded-Host", "untrusted.example")
	request.Header.Set("Forwarded", "host=untrusted.example;proto=http")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	require.Equal(t, http.StatusOK, response.Code)
	assert.Equal(t, "text/html; charset=utf-8", response.Header().Get("Content-Type"))
	assert.Equal(t, "no-store", response.Header().Get("Cache-Control"))
	assert.NotEmpty(t, response.Header().Get("Content-Security-Policy"))
	assert.Contains(t, response.Body.String(), "https://hub.example.org:9443/v1")
	assert.Contains(t, response.Body.String(), "--hub https://hub.example.org:9443")
	assert.Contains(t, response.Body.String(), "http://127.0.0.1:11434")
	for _, value := range []string{testClientToken, testWorkerToken, testDashboardToken, "untrusted.example", "https://hub.example.internal"} {
		assert.NotContains(t, response.Body.String(), value)
	}
}

func TestGuide_RejectsUnsafeHostBeforeRendering(t *testing.T) {
	h := configuredDashboard(t, true, false, "")
	request := httptest.NewRequest(http.MethodGet, "https://hub/dashboard/readme.html", nil)
	request.Host = "$(unsafe)"
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusBadRequest, response.Code)
	assert.NotContains(t, response.Body.String(), "$(unsafe)")
}

func TestGuide_PreservesDashboardAccessRestrictions(t *testing.T) {
	test := func(name, address, method, origin, site string, authenticated bool, expected int) {
		t.Run(name, func(t *testing.T) {
			h := configuredDashboard(t, false, false, "")
			request := httptest.NewRequest(method, address, nil)
			request.Header.Set("Origin", origin)
			request.Header.Set("Sec-Fetch-Site", site)
			if authenticated {
				request.SetBasicAuth("chair", testDashboardToken)
			}
			response := httptest.NewRecorder()

			h.ServeHTTP(response, request)

			assert.Equal(t, expected, response.Code)
		})
	}
	test("private guide requires login", "https://hub/dashboard/readme.html", http.MethodGet, "", "", false, http.StatusUnauthorized)
	test("authenticated guide", "https://hub/dashboard/readme.html", http.MethodGet, "", "", true, http.StatusOK)
	test("TLS required", "http://hub/dashboard/readme.html", http.MethodGet, "", "", true, http.StatusForbidden)
	test("POST rejected", "https://hub/dashboard/readme.html", http.MethodPost, "", "", true, http.StatusMethodNotAllowed)
	test("query rejected", "https://hub/dashboard/readme.html?host=evil", http.MethodGet, "", "", true, http.StatusForbidden)
	test("foreign origin rejected", "https://hub/dashboard/readme.html", http.MethodGet, "https://evil", "", true, http.StatusForbidden)
	test("cross-site fetch rejected", "https://hub/dashboard/readme.html", http.MethodGet, "", "cross-site", true, http.StatusForbidden)
}

func TestGuideRenderer_OmitsUnsafeHTMLAndLinkSchemes(t *testing.T) {
	source := []byte("## Test section\n\n<script>alert('unsafe')</script>\n\n[unsafe](javascript:alert%281%29)\n\n```html\n<script>example</script>\n```\n")
	page, err := renderGuide(source, "https://hub:8443")
	require.NoError(t, err)
	assert.NotContains(t, string(page), "<script>")
	assert.NotContains(t, string(page), `href="javascript:`)
	assert.Contains(t, string(page), "&lt;script&gt;example&lt;/script&gt;")
	assert.Contains(t, string(page), `href="#test-section"`)
}
