// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessagesRequest_TranslatesSystemToolsAndOrderedResults(t *testing.T) {
	_, body, err := translateMessagesRequest([]byte(`{
"model":"coder","max_tokens":100,"stream":true,
"system":[{"type":"text","text":"system one","cache_control":{"type":"ephemeral"}},{"type":"text","text":"system two"}],
"tools":[{"name":"Read","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
"tool_choice":{"type":"tool","name":"Read","disable_parallel_tool_use":true},
"messages":[{"role":"user","content":"read two files"},
{"role":"assistant","content":[{"type":"text","text":"Reading"},{"type":"tool_use","id":"call_a","name":"Read","input":{"path":"a"}},{"type":"tool_use","id":"call_b","name":"Read","input":{"path":"b"}}]},
{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_b","is_error":true,"content":[{"type":"text","text":"missing"}]},{"type":"tool_result","tool_use_id":"call_a","content":"file data"},{"type":"text","text":"continue"}]}]}`))
	require.NoError(t, err)
	var chat messagesChatRequest
	require.NoError(t, json.Unmarshal(body, &chat))
	assert.Equal(t, "coder", chat.Model)
	assert.Equal(t, int64(100), chat.MaxTokens)
	require.NotNil(t, chat.StreamOptions)
	assert.True(t, chat.StreamOptions.IncludeUsage)
	require.Len(t, chat.Messages, 6)
	assert.Equal(t, "system", chat.Messages[0].Role)
	require.NotNil(t, chat.Messages[0].Content)
	assert.Equal(t, "system one\nsystem two", *chat.Messages[0].Content)
	require.Len(t, chat.Messages[2].Tools, 2)
	assert.JSONEq(t, `{"path":"a"}`, chat.Messages[2].Tools[0].Function.Arguments)
	assert.Equal(t, "call_b", chat.Messages[3].ToolID)
	assert.Equal(t, "[tool error]\nmissing", *chat.Messages[3].Content)
	assert.Equal(t, "call_a", chat.Messages[4].ToolID)
	assert.Equal(t, "continue", *chat.Messages[5].Content)
	assert.JSONEq(t, `{"type":"function","function":{"name":"Read"}}`, string(chat.ToolChoice))
	require.NotNil(t, chat.Parallel)
	assert.False(t, *chat.Parallel)
}

func TestMessagesRequest_RejectsUnsupportedFeaturesAndBrokenToolHistory(t *testing.T) {
	test := func(name, fields string) {
		t.Run(name, func(t *testing.T) {
			_, _, err := translateMessagesRequest([]byte(`{"model":"coder","max_tokens":100,` + fields + `}`))
			assert.Error(t, err)
		})
	}
	test("images are not silently dropped", `"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"http://private"}}]}]`)
	test("adaptive thinking is not silently disabled", `"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"adaptive"}`)
	test("server tools are not mapped to client tools", `"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search","input_schema":{}}]`)
	test("unknown fields are rejected", `"messages":[{"role":"user","content":"hi"}],"context_management":{}`)
	test("orphan results are rejected", `"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"missing","content":"secret"}]}]`)
	test("unanswered calls are rejected", `"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Read","input":{}}]},{"role":"user","content":"new input"}]`)
	test("assistant prefilling is rejected", `"messages":[{"role":"assistant","content":"partial"}]`)
	test("empty conversation is rejected", `"messages":[]`)
}

func TestMessages_TranslatesJSONThroughRealWorkerAndCreditsUsage(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	forwarded := make(chan messagesChatRequest, 1)
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		for _, header := range []string{"Authorization", "X-Api-Key", "Cookie", "Anthropic-Beta", "Anthropic-Version"} {
			assert.Empty(t, r.Header.Get(header), header)
		}
		var chat messagesChatRequest
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&chat))
		forwarded <- chat
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Set-Cookie", "private-engine-cookie")
		_, err := io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"checking","tool_calls":[{"id":"call_read","type":"function","function":{"name":"Read","arguments":"{\"path\":\"README.md\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":7}}`)
		assert.NoError(t, err)
	}), engineSettings{kind: engineOpenAI})
	request := openAIRequest(t, fixture, "/v1/messages?beta=true", `{"model":"coder","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}`)
	request.Header.Del("Authorization")
	request.Header.Set("X-Api-Key", testClientToken)
	request.Header.Set("Cookie", "private-cookie")
	request.Header.Set("Anthropic-Version", "2023-06-01")
	response, err := fixture.client.Do(request)
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	assert.Empty(t, response.Header.Get("Set-Cookie"))
	var message messagesResponse
	require.NoError(t, json.NewDecoder(response.Body).Decode(&message))
	assert.Equal(t, "message", message.Type)
	assert.Equal(t, "coder", message.Model)
	assert.Equal(t, messagesUsage{Input: 20, Output: 7}, message.Usage)
	require.NotNil(t, message.StopReason)
	assert.Equal(t, "tool_use", *message.StopReason)
	require.Len(t, message.Content, 2)
	assert.JSONEq(t, `{"path":"README.md"}`, string(message.Content[1].Input))
	assert.Equal(t, "coder", (<-forwarded).Model)
	require.Eventually(t, func() bool {
		view, err := fixture.hub.stats.snapshot("coder")
		return err == nil && len(view.Tokens) == 1 && view.Tokens[0].Tokens == 7
	}, time.Second, 10*time.Millisecond)
}

func readMessagesEvent(t *testing.T, scanner *bufio.Scanner) map[string]json.RawMessage {
	t.Helper()
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "data: ") {
			var event map[string]json.RawMessage
			require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(scanner.Text(), "data: ")), &event))
			return event
		}
	}
	require.NoError(t, scanner.Err())
	t.Fatal("expected another Messages event")
	return nil
}

func TestMessages_StreamsTextBeforeEngineCompletesAndReassemblesParallelTools(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	release := make(chan struct{})
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"First\"}}]}\n\n")
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		for _, frame := range []string{
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"Read","arguments":"{\"path\":"}},{"index":1,"id":"call_b","type":"function","function":{"name":"Read","arguments":"{\"path\":\"b\"}"}}]}}]}`,
			`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":12}}`,
			`[DONE]`,
		} {
			_, err := io.WriteString(w, "data: "+frame+"\n\n")
			assert.NoError(t, err)
		}
	}), engineSettings{kind: engineOpenAI})
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/messages", `{"model":"coder","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	require.Equal(t, http.StatusOK, response.StatusCode)
	scanner := bufio.NewScanner(response.Body)
	assert.JSONEq(t, `"message_start"`, string(readMessagesEvent(t, scanner)["type"]))
	assert.JSONEq(t, `"content_block_start"`, string(readMessagesEvent(t, scanner)["type"]))
	assert.JSONEq(t, `{"type":"text_delta","text":"First"}`, string(readMessagesEvent(t, scanner)["delta"]))
	close(release)
	arguments := make(map[int]string)
	toolIDs := make(map[int]string)
	var kinds []string
	for scanner.Scan() {
		if !strings.HasPrefix(scanner.Text(), "data: ") {
			continue
		}
		var event struct {
			Type  string        `json:"type"`
			Index int           `json:"index"`
			Block messagesBlock `json:"content_block"`
			Delta struct {
				Type    string `json:"type"`
				Partial string `json:"partial_json"`
				Stop    string `json:"stop_reason"`
			} `json:"delta"`
			Usage messagesUsage `json:"usage"`
		}
		require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(scanner.Text(), "data: ")), &event))
		kinds = append(kinds, event.Type)
		if event.Type == "content_block_start" {
			toolIDs[event.Index] = event.Block.ID
		}
		if event.Delta.Type == "input_json_delta" {
			arguments[event.Index] += event.Delta.Partial
		}
		if event.Type == "message_delta" {
			assert.Equal(t, "tool_use", event.Delta.Stop)
			assert.Equal(t, messagesUsage{Input: 50, Output: 12}, event.Usage)
		}
	}
	require.NoError(t, scanner.Err())
	assert.Equal(t, map[int]string{1: "call_a", 2: "call_b"}, toolIDs)
	assert.JSONEq(t, `{"path":"a"}`, arguments[1])
	assert.JSONEq(t, `{"path":"b"}`, arguments[2])
	require.NotEmpty(t, kinds)
	assert.Equal(t, "message_stop", kinds[len(kinds)-1])
}

func TestMessages_RejectsUnsafeAccessAndUnexpectedQueries(t *testing.T) {
	h := newHub(testClientToken, map[string]string{}, 2)
	test := func(name, method, address, bearer, key, origin string, status int) {
		t.Run(name, func(t *testing.T) {
			r := httptest.NewRequest(method, address, strings.NewReader(`{"model":"coder","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`))
			if bearer != "" {
				r.Header.Set("Authorization", "Bearer "+bearer)
			}
			if key != "" {
				r.Header.Set("X-Api-Key", key)
			}
			r.Header.Set("Origin", origin)
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			assert.Equal(t, status, w.Code)
			var result struct {
				Type  string `json:"type"`
				Error struct {
					Type    string `json:"type"`
					Message string `json:"message"`
				} `json:"error"`
			}
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &result))
			assert.Equal(t, "error", result.Type)
			assert.NotEmpty(t, result.Error.Type)
		})
	}
	test("TLS required", http.MethodPost, "http://hub/v1/messages", testClientToken, "", "", http.StatusForbidden)
	test("browser rejected", http.MethodPost, "https://hub/v1/messages", testClientToken, "", "https://hub", http.StatusForbidden)
	test("no credentials", http.MethodPost, "https://hub/v1/messages", "", "", "", http.StatusUnauthorized)
	test("wrong first credential not bypassed", http.MethodPost, "https://hub/v1/messages", "wrong", testClientToken, "", http.StatusUnauthorized)
	test("wrong second credential not bypassed", http.MethodPost, "https://hub/v1/messages", testClientToken, testWorkerToken, "", http.StatusUnauthorized)
	test("x-api-key accepted", http.MethodPost, "https://hub/v1/messages", "", testClientToken, "", http.StatusBadGateway)
	test("beta query accepted", http.MethodPost, "https://hub/v1/messages?beta=true", testClientToken, "", "", http.StatusBadGateway)
	test("duplicate beta rejected", http.MethodPost, "https://hub/v1/messages?beta=true&beta=true", testClientToken, "", "", http.StatusNotFound)
	test("other query rejected", http.MethodPost, "https://hub/v1/messages?x=y", testClientToken, "", "", http.StatusNotFound)
	test("only POST allowed", http.MethodGet, "https://hub/v1/messages", testClientToken, "", "", http.StatusNotFound)
	test("counting not estimated", http.MethodPost, "https://hub/v1/messages/count_tokens", testClientToken, "", "", http.StatusNotFound)
}

func TestMessages_CancellationReleasesWorkerAndSharedCapacity(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 1)
	enableScores(t, fixture.hub)
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"First\"}}]}\n\n")
		assert.NoError(t, err)
		assert.NoError(t, http.NewResponseController(w).Flush())
		<-r.Context().Done()
	}), engineSettings{kind: engineOpenAI})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := openAIRequest(t, fixture, "/v1/messages", `{"model":"coder","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`).WithContext(ctx)
	response, err := fixture.client.Do(r)
	require.NoError(t, err)
	defer func() { assert.NoError(t, response.Body.Close()) }()
	other, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/chat/completions", `{"model":"coder"}`))
	require.NoError(t, err)
	assert.Equal(t, http.StatusServiceUnavailable, other.StatusCode)
	require.NoError(t, other.Body.Close())
	cancel()
	require.Eventually(t, func() bool {
		view, err := fixture.hub.stats.snapshot("coder")
		return err == nil && len(view.Requests) == 1 && view.Requests[0].Canceled == 1 && len(fixture.hub.requests) == 0
	}, time.Second, 10*time.Millisecond)
}

func TestMessagesStream_ReportsTruncatedOrInvalidStreamsWithoutSuccess(t *testing.T) {
	test := func(name, body string) {
		t.Run(name, func(t *testing.T) {
			w := httptest.NewRecorder()
			usage := &usageObserver{failed: true}
			err := streamMessages(strings.NewReader(body), messagesEvents{w}, "coder", "msg_test", usage)
			assert.Error(t, err)
			assert.NotContains(t, w.Body.String(), "event: message_stop")
		})
	}
	test("missing finish reason", "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
	test("missing done event", "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
	test("oversized frame", "data: "+strings.Repeat("x", maxMessagesFrame)+"\n\n")
	test("upstream error", "data: {\"error\":{\"message\":\"private engine error\"}}\n\n")
}

func TestMessages_ContainsEngineFailuresWithoutLeakingBodiesOrAwardingTokens(t *testing.T) {
	test := func(name, kind, body string, upstream, status int, stream bool) {
		t.Run(name, func(t *testing.T) {
			fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
			enableScores(t, fixture.hub)
			var logs logBuffer
			fixture.hub.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
			startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", kind)
				w.Header().Set("Location", "http://unreachable.invalid/PRIVATE_ENGINE_CONTENT")
				w.Header().Set("Set-Cookie", "PRIVATE_ENGINE_CONTENT")
				w.WriteHeader(upstream)
				_, err := io.WriteString(w, body)
				assert.NoError(t, err)
			}), engineSettings{kind: engineOpenAI})
			request := `{"model":"coder","max_tokens":20,"messages":[{"role":"user","content":"PRIVATE_PROMPT"}],"stream":false}`
			if stream {
				request = `{"model":"coder","max_tokens":20,"messages":[{"role":"user","content":"PRIVATE_PROMPT"}],"stream":true}`
			}
			response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/messages", request))
			require.NoError(t, err)
			data, err := io.ReadAll(response.Body)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, status, response.StatusCode)
			assert.NotContains(t, string(data), "PRIVATE_ENGINE_CONTENT")
			assert.Empty(t, response.Header.Get("Location"))
			assert.Empty(t, response.Header.Get("Set-Cookie"))
			if stream {
				assert.Contains(t, string(data), "event: error")
				assert.NotContains(t, string(data), "event: message_stop")
			}
			require.Eventually(t, func() bool {
				view, err := fixture.hub.stats.snapshot("coder")
				return err == nil && len(view.Requests) == 1 && view.Requests[0].Failed == 1 && len(fixture.hub.requests) == 0
			}, time.Second, 10*time.Millisecond)
			view, err := fixture.hub.stats.snapshot("coder")
			require.NoError(t, err)
			assert.Empty(t, view.Tokens)
			// Close joins inference goroutines before inspecting their log buffer.
			require.NoError(t, fixture.hub.Close())
			for _, private := range []string{"PRIVATE_PROMPT", "PRIVATE_ENGINE_CONTENT", testClientToken, testWorkerToken} {
				assert.NotContains(t, logs.text(), private)
			}
		})
	}
	test("upstream error body is private", "application/json", `{"error":{"message":"PRIVATE_ENGINE_CONTENT"}}`, http.StatusInternalServerError, http.StatusInternalServerError, false)
	test("redirect is not followed", "application/json", `PRIVATE_ENGINE_CONTENT`, http.StatusFound, http.StatusBadGateway, false)
	test("malformed JSON is private", "application/json", `PRIVATE_ENGINE_CONTENT`, http.StatusOK, http.StatusBadGateway, false)
	test("wrong content type is rejected", "text/html", `PRIVATE_ENGINE_CONTENT`, http.StatusOK, http.StatusBadGateway, false)
	test("reported usage cannot rescue a truncated stream", "text/event-stream", "data: {\"choices\":[],\"usage\":{\"completion_tokens\":42}}\n\n", http.StatusOK, http.StatusOK, true)
}

func TestMessages_CompletesWithoutInventingMissingUsage(t *testing.T) {
	fixture := startHubFixture(t, map[string]string{testWorkerID: testWorkerToken}, 2)
	enableScores(t, fixture.hub)
	startTestEngineWorker(t, fixture, testWorkerID, testWorkerToken, testOpenAIEngine(t, []string{"coder"}, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, err := io.WriteString(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`)
		assert.NoError(t, err)
	}), engineSettings{kind: engineOpenAI})
	response, err := fixture.client.Do(openAIRequest(t, fixture, "/v1/messages", `{"model":"coder","max_tokens":20,"messages":[{"role":"user","content":"hi"}]}`))
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, response.StatusCode)
	_, err = io.Copy(io.Discard, response.Body)
	require.NoError(t, err)
	require.NoError(t, response.Body.Close())
	require.Eventually(t, func() bool {
		view, err := fixture.hub.stats.snapshot("coder")
		return err == nil && len(view.Requests) == 1 && view.Requests[0].Completed == 1 && len(view.Tokens) == 0
	}, time.Second, 10*time.Millisecond)
}

func TestMessagesRequest_MapsEffortAndAllToolChoices(t *testing.T) {
	test := func(name, choice, expected string, parallel bool) {
		t.Run(name, func(t *testing.T) {
			_, body, err := translateMessagesRequest([]byte(`{"model":"coder","max_tokens":20,"output_config":{"effort":"medium"},"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"Read","input_schema":{"type":"object"}}],"tool_choice":` + choice + `}`))
			require.NoError(t, err)
			var chat messagesChatRequest
			require.NoError(t, json.Unmarshal(body, &chat))
			assert.Equal(t, "medium", chat.ReasoningEffort)
			assert.JSONEq(t, expected, string(chat.ToolChoice))
			if parallel {
				assert.Nil(t, chat.Parallel)
			} else {
				require.NotNil(t, chat.Parallel)
				assert.False(t, *chat.Parallel)
			}
		})
	}
	test("automatic", `{"type":"auto"}`, `"auto"`, true)
	test("required", `{"type":"any","disable_parallel_tool_use":true}`, `"required"`, false)
	test("disabled", `{"type":"none"}`, `"none"`, true)
	test("named", `{"type":"tool","name":"Read"}`, `{"type":"function","function":{"name":"Read"}}`, true)
}
