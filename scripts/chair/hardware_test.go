// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLinuxHardware_ParsesCPUModelAndHostMemory(t *testing.T) {
	cpu, memory := linuxHardware([]byte("processor : 0\nmodel name : Example CPU\nmodel name : Example CPU\n"), []byte("MemTotal:       33554432 kB\nMemFree: 1234 kB\n"))

	assert.Equal(t, "Example CPU", cpu)
	assert.Equal(t, uint64(32<<30), memory)
}

func TestLinuxHardware_LeavesUnavailableValuesEmpty(t *testing.T) {
	test := func(name, memory string) {
		t.Run(name, func(t *testing.T) {
			cpu, ram := linuxHardware(nil, []byte(memory))
			assert.Empty(t, cpu)
			assert.Zero(t, ram)
		})
	}
	test("missing memory", "")
	test("unexpected units", "MemTotal: 1234 MB")
	test("invalid number", "MemTotal: unknown kB")
	test("overflowing memory", "MemTotal: 18446744073709551615 kB")
}

func TestNvidiaHardware_ListsGPUModelsAndDedicatedMemory(t *testing.T) {
	gpus := nvidiaHardware([]byte("NVIDIA RTX 4090, 24564\nNVIDIA A100-SXM4-80GB, 81920\n"))

	assert.Equal(t, []workerGPU{{Name: "NVIDIA RTX 4090", MemoryMiB: 24564}, {Name: "NVIDIA A100-SXM4-80GB", MemoryMiB: 81920}}, gpus)
}

func TestNvidiaHardware_DoesNotInventMissingGPUDetails(t *testing.T) {
	test := func(name, data string) {
		t.Run(name, func(t *testing.T) { assert.Empty(t, nvidiaHardware([]byte(data))) })
	}
	test("tool unavailable", "")
	test("invalid memory", "NVIDIA GPU, invalid")
	test("unexpected output", "driver error")
	test("control characters", "\"NVIDIA\nGPU\", 1234")
}

func TestNvidiaHardware_KeepsGPUNameWhenDedicatedMemoryIsUnavailable(t *testing.T) {
	gpus := nvidiaHardware([]byte("NVIDIA GB10, [N/A]\n"))

	assert.Equal(t, []workerGPU{{Name: "NVIDIA GB10"}}, gpus)
	assert.Contains(t, (workerHardware{GPUs: gpus}).summary(), "NVIDIA GB10 (VRAM unavailable)")
}

func TestHardwareMetadata_PreservesReportedFields(t *testing.T) {
	expected := workerHardware{OS: "linux", Arch: "arm64", CPU: "Example CPU", LogicalCPUs: 16, MemoryBytes: 64 << 30, GPUs: []workerGPU{{Name: "Example GPU", MemoryMiB: 24576}}}
	data, err := json.Marshal(expected)
	require.NoError(t, err)

	actual, err := decodeHardware(base64.RawURLEncoding.EncodeToString(data))

	require.NoError(t, err)
	assert.Equal(t, expected, actual)
}

func TestHardwareMetadata_RejectsMalformedOrUnboundedReports(t *testing.T) {
	test := func(name, jsonBody string) {
		t.Run(name, func(t *testing.T) {
			_, err := decodeHardware(base64.RawURLEncoding.EncodeToString([]byte(jsonBody)))
			assert.Error(t, err)
		})
	}
	test("invalid JSON", "{")
	test("negative CPU count", `{"logical_cpus":-1}`)
	test("unbounded CPU count", `{"logical_cpus":2000000}`)
	test("control characters", `{"cpu":"private\nline"}`)
	test("oversized model name", `{"cpu":"`+strings.Repeat("a", 257)+`"}`)
	test("too many GPUs", `{"gpus":[`+strings.Repeat(`{"name":"GPU"},`, 32)+`{"name":"GPU"}]}`)
	test("oversized header", `{"cpu":"`+strings.Repeat("a", 16384)+`"}`)
}

func TestHardwareMetadata_AllowsAbsentOptionalReport(t *testing.T) {
	info, err := decodeHardware("")

	require.NoError(t, err)
	assert.Empty(t, info.OS)
	assert.Contains(t, info.summary(), "RAM unavailable")
	assert.Contains(t, info.summary(), "GPU details unavailable")
}

func TestWorkerConnection_RejectsInvalidHardwareBeforeUpgrade(t *testing.T) {
	h := newHub(testClientToken, map[string]string{testWorkerID: testWorkerToken}, 1)
	request := httptest.NewRequest(http.MethodGet, "https://hub/worker", nil)
	request.Header.Set("Authorization", "Bearer "+testWorkerToken)
	request.Header.Set(workerIDHeader, testWorkerID)
	request.Header.Set("Sec-WebSocket-Protocol", protocol)
	request.Header.Set(engineTypeHeader, engineOllama)
	request.Header.Set(hardwareHeader, "not base64")
	response := httptest.NewRecorder()

	h.ServeHTTP(response, request)

	assert.Equal(t, http.StatusBadRequest, response.Code)
	assert.Empty(t, h.workers)
}

func TestHardwareOutput_RejectsDataBeyondTheOutputLimit(t *testing.T) {
	var output hardwareOutput
	n, err := output.Write([]byte(strings.Repeat("a", 64<<10)))
	require.NoError(t, err)
	assert.Equal(t, 64<<10, n)

	n, err = output.Write([]byte("b"))

	assert.Error(t, err)
	assert.Zero(t, n)
	assert.Equal(t, 64<<10, output.buffer.Len())
}
