// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHub_EnforcesConfiguredConcurrencyLimit(t *testing.T) {
	test := func(name string, limit int) {
		t.Run(name, func(t *testing.T) {
			finish := make(chan struct{})
			defer close(finish)
			calls := make(chan struct{}, limit+1)
			fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, limit)
			startTestWorker(t, fixture, testWorkerID, testWorkerToken, testEngine(t, []string{"test-model:latest"},
				func(w http.ResponseWriter, r *http.Request) {
					calls <- struct{}{}
					assert.NoError(t, json.NewEncoder(w).Encode(map[string]bool{"done": false}))
					assert.NoError(t, http.NewResponseController(w).Flush())
					select {
					case <-finish:
					case <-r.Context().Done():
					}
				}))

			for range limit {
				response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, response.Body.Close()) })
				require.Equal(t, http.StatusOK, response.StatusCode)
			}
			response, err := fixture.client.Do(fixture.request(t, context.Background(), `{"model":"test-model"}`))
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())

			assert.Equal(t, http.StatusServiceUnavailable, response.StatusCode)
			assert.Len(t, calls, limit, "excess requests must not reach the engine")
		})
	}
	test("limit below default", 2)
	test("limit above sixteen", 24)
}

func TestHubCommand_RejectsNonpositiveConcurrencyLimit(t *testing.T) {
	test := func(name, value string) {
		t.Run(name, func(t *testing.T) {
			err := runHub(context.Background(), []string{"--max-concurrent-requests", value})

			require.EqualError(t, err, "--max-concurrent-requests must be positive")
		})
	}
	test("zero", "0")
	test("negative", "-1")
}
