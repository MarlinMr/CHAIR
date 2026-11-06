// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func observeUsage(t *testing.T, path, kind, body string) *usageObserver {
	t.Helper()
	u := newUsageObserver(path, kind)
	reader := &observedBody{ReadCloser: io.NopCloser(strings.NewReader(body)), usage: u}
	var forwarded strings.Builder
	buffer := make([]byte, 7) // Split JSON fields, UTF-8, CRLF, and SSE boundaries.
	_, err := io.CopyBuffer(&forwarded, reader, buffer)
	require.NoError(t, err)
	require.NoError(t, reader.Close())
	assert.Equal(t, body, forwarded.String())
	assert.True(t, u.eof)
	return u
}

func TestUsageObserver_ExtractsFinalOutputCountsWithoutChangingBodies(t *testing.T) {
	test := func(name, path, kind, body string, tokens int64) {
		t.Run(name, func(t *testing.T) {
			u := observeUsage(t, path, kind, body)
			require.NotNil(t, u.tokens)
			assert.Equal(t, tokens, *u.tokens)
			assert.True(t, u.terminal)
			assert.False(t, u.failed)
			assert.False(t, u.invalid)
		})
	}
	test("chat JSON", "/v1/chat/completions", "application/json", "{\n\"choices\":[],\n\"usage\":{\"prompt_tokens\":99,\"completion_tokens\":12}\n}", 12)
	test("chat SSE usage then done", "/v1/chat/completions", "text/event-stream", "data: {\"choices\":[{\"delta\":{\"content\":\"private café\"}}]}\r\n\r\ndata: {\"usage\":{\"completion_tokens\":13}}\r\n\r\ndata: [DONE]\r\n\r\n", 13)
	test("SSE multiline data", "/v1/chat/completions", "text/event-stream", "data: {\n"+"data: \"usage\": {\"completion_tokens\":3}}\n\ndata: [DONE]\n\n", 3)
	test("native generate NDJSON", "/api/generate", "application/x-ndjson", "{\"response\":\"private\",\"done\":false}\n{\"done\":true,\"eval_count\":14,\"prompt_eval_count\":700}\n", 14)
	test("native chat JSON", "/api/chat", "application/json", `{"done":true,"eval_count":15}`, 15)
	test("Responses JSON", "/v1/responses", "application/json", `{"status":"completed","usage":{"input_tokens":80,"output_tokens":16}}`, 16)
	test("Responses SSE", "/v1/responses", "text/event-stream", "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"output_tokens\":17}}}\n\n", 17)
	test("zero is reported usage", "/v1/chat/completions", "application/json", `{"usage":{"completion_tokens":0}}`, 0)
}

func TestUsageObserver_DoesNotEstimateMissingOrInvalidUsage(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			u := observeUsage(t, "/v1/chat/completions", "application/json", body)
			assert.True(t, u.tokens == nil || u.invalid)
		})
	}
	test("missing usage", `{"choices":[{"message":{"content":"many words must not become token estimates"}}]}`)
	test("null count", `{"usage":{"completion_tokens":null}}`)
	test("fractional count", `{"usage":{"completion_tokens":2.5}}`)
	test("string count", `{"usage":{"completion_tokens":"12"}}`)
	test("negative count", `{"usage":{"completion_tokens":-1}}`)
	test("unbounded count", `{"usage":{"completion_tokens":1000000001}}`)
	test("input counts are not output", `{"usage":{"prompt_tokens":400}}`)
}

func TestUsageObserver_DeduplicatesRepeatedUsageAndRejectsConflicts(t *testing.T) {
	u := observeUsage(t, "/v1/chat/completions", "text/event-stream", "data: {\"usage\":{\"completion_tokens\":5}}\n\ndata: {\"usage\":{\"completion_tokens\":5}}\n\ndata: [DONE]\n\n")
	require.NotNil(t, u.tokens)
	assert.Equal(t, int64(5), *u.tokens)
	assert.False(t, u.invalid)
	u = observeUsage(t, "/v1/chat/completions", "text/event-stream", "data: {\"usage\":{\"completion_tokens\":5}}\n\ndata: {\"usage\":{\"completion_tokens\":6}}\n\ndata: [DONE]\n\n")
	assert.True(t, u.invalid)
}

func TestUsageObserver_SkipsOversizedFramesAndRecoversAtNextEvent(t *testing.T) {
	body := "data: " + strings.Repeat("x", maxUsageFrame+20) + "\n\ndata: {\"usage\":{\"completion_tokens\":9}}\n\ndata: [DONE]\n\n"
	u := observeUsage(t, "/v1/chat/completions", "text/event-stream", body)
	require.NotNil(t, u.tokens)
	assert.Equal(t, int64(9), *u.tokens)
	assert.True(t, u.terminal)
	assert.Empty(t, u.body)
	assert.Empty(t, u.line)
	assert.Empty(t, u.event)
}

func TestUsageObserver_RecognizesIncompleteAndFailedStreams(t *testing.T) {
	u := observeUsage(t, "/v1/chat/completions", "text/event-stream", "data: {\"usage\":{\"completion_tokens\":5}}\n\n")
	assert.False(t, u.terminal, "a usage frame alone does not prove successful stream completion")
	u = observeUsage(t, "/v1/responses", "text/event-stream", "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"}}\n\n")
	assert.True(t, u.failed)
	assert.False(t, u.terminal)
}
