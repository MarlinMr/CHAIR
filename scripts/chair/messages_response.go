// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"unicode/utf8"
)

const maxMessagesFrame = 1 << 20
const maxMessagesBlocks = 1024

type messagesUsage struct {
	Input  int64 `json:"input_tokens"`
	Output int64 `json:"output_tokens"`
}

type messagesResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Model        string          `json:"model"`
	Content      []messagesBlock `json:"content"`
	StopReason   *string         `json:"stop_reason"`
	StopSequence *string         `json:"stop_sequence"`
	Usage        messagesUsage   `json:"usage"`
}

type messagesChatResponse struct {
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string             `json:"role"`
			Content *string            `json:"content"`
			Tools   []messagesChatTool `json:"tool_calls"`
			Refusal *string            `json:"refusal"`
		} `json:"message"`
		Delta struct {
			Role    string             `json:"role"`
			Content *string            `json:"content"`
			Tools   []messagesChatTool `json:"tool_calls"`
			Refusal *string            `json:"refusal"`
		} `json:"delta"`
		Finish *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		Input  *int64 `json:"prompt_tokens"`
		Output *int64 `json:"completion_tokens"`
	} `json:"usage"`
	Error json.RawMessage `json:"error"`
}

func messagesStop(reason string, tools bool) (string, error) {
	switch reason {
	case "stop":
		if tools {
			return "", fmt.Errorf("tool calls without tool finish reason")
		}
		return "end_turn", nil
	case "length":
		return "max_tokens", nil
	case "tool_calls":
		if !tools {
			return "", fmt.Errorf("tool finish reason without tool calls")
		}
		return "tool_use", nil
	default:
		return "", fmt.Errorf("unsupported engine finish reason")
	}
}

func messagesReportedUsage(frame messagesChatResponse, usage *usageObserver) (messagesUsage, error) {
	var reported messagesUsage
	if frame.Usage == nil {
		return reported, nil
	}
	if frame.Usage.Input != nil {
		reported.Input = *frame.Usage.Input
		if reported.Input < 0 || reported.Input > 1_000_000_000 {
			return reported, fmt.Errorf("invalid engine usage")
		}
	}
	usage.accept(frame.Usage.Output)
	if usage.invalid {
		return reported, fmt.Errorf("invalid or conflicting engine usage")
	}
	if usage.tokens != nil {
		reported.Output = *usage.tokens
	}
	return reported, nil
}

func messagesJSONResponse(data []byte, model, id string, usage *usageObserver) (messagesResponse, error) {
	var frame messagesChatResponse
	out := messagesResponse{ID: id, Type: "message", Role: "assistant", Model: model, Content: []messagesBlock{}}
	if !utf8.Valid(data) || json.Unmarshal(data, &frame) != nil || len(frame.Choices) != 1 || frame.Choices[0].Index != 0 || len(frame.Error) > 0 && string(frame.Error) != "null" {
		return out, fmt.Errorf("invalid engine response")
	}
	choice := frame.Choices[0]
	if choice.Message.Role != "assistant" || choice.Message.Refusal != nil || choice.Finish == nil || len(choice.Message.Tools) > maxMessagesBlocks {
		return out, fmt.Errorf("incomplete or unsupported engine response")
	}
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		out.Content = append(out.Content, messagesBlock{Type: "text", Text: *choice.Message.Content})
	}
	ids := make(map[string]bool)
	for _, tool := range choice.Message.Tools {
		if tool.Type != "function" || !validModel(tool.ID) || ids[tool.ID] || !validModel(tool.Function.Name) || !messagesObject([]byte(tool.Function.Arguments)) {
			return out, fmt.Errorf("invalid engine tool call")
		}
		ids[tool.ID] = true
		out.Content = append(out.Content, messagesBlock{Type: "tool_use", ID: tool.ID, Name: tool.Function.Name, Input: json.RawMessage(tool.Function.Arguments)})
	}
	stop, err := messagesStop(*choice.Finish, len(choice.Message.Tools) > 0)
	if err != nil {
		return out, err
	}
	out.StopReason = &stop
	out.Usage, err = messagesReportedUsage(frame, usage)
	return out, err
}

type messagesEvents struct{ writer http.ResponseWriter }

func (e messagesEvents) send(kind string, fields map[string]interface{}) error {
	fields["type"] = kind
	data, err := json.Marshal(fields)
	if err != nil {
		return err
	}
	if _, err = fmt.Fprintf(e.writer, "event: %s\ndata: %s\n\n", kind, data); err != nil {
		return err
	}
	return http.NewResponseController(e.writer).Flush()
}

type messagesStreamTool struct {
	id, name  string
	arguments strings.Builder
	block     int
}

// Only tool argument objects are retained for validation before a block closes.
// Text deltas are forwarded immediately; frames and aggregate tool inputs are
// bounded independently. No upstream IDs or error bodies are exposed/logged.
func streamMessages(body io.Reader, events messagesEvents, model, id string, usage *usageObserver) error {
	message := messagesResponse{ID: id, Type: "message", Role: "assistant", Model: model, Content: []messagesBlock{}}
	if err := events.send("message_start", map[string]interface{}{"message": message}); err != nil {
		return err
	}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 4096), maxMessagesFrame)
	var event []byte
	blocks, textBlock, argumentBytes := 0, -1, 0
	tools := make(map[int]*messagesStreamTool)
	toolIDs := make(map[string]bool)
	finish := ""
	reported := messagesUsage{}
	startTool := func(tool *messagesStreamTool) error {
		if tool.block >= 0 {
			return nil
		}
		if !validModel(tool.id) || !validModel(tool.name) || toolIDs[tool.id] || blocks >= maxMessagesBlocks {
			return fmt.Errorf("invalid or excessive engine tool calls")
		}
		tool.block = blocks
		toolIDs[tool.id] = true
		blocks++
		return events.send("content_block_start", map[string]interface{}{"index": tool.block, "content_block": map[string]interface{}{"type": "tool_use", "id": tool.id, "name": tool.name, "input": map[string]interface{}{}}})
	}
	consume := func(data []byte) (bool, error) {
		data = bytes.TrimSpace(data)
		if len(data) == 0 {
			return false, nil
		}
		if bytes.Equal(data, []byte("[DONE]")) {
			if finish == "" {
				return false, fmt.Errorf("engine stream lacks a finish reason")
			}
			for index := 0; index < len(tools); index++ {
				tool := tools[index]
				if tool == nil || !messagesObject([]byte(tool.arguments.String())) {
					return false, fmt.Errorf("incomplete engine tool arguments")
				}
				if err := startTool(tool); err != nil {
					return false, err
				}
			}
			stop, err := messagesStop(finish, len(tools) > 0)
			if err != nil {
				return false, err
			}
			for index := 0; index < blocks; index++ {
				if err := events.send("content_block_stop", map[string]interface{}{"index": index}); err != nil {
					return false, err
				}
			}
			if err := events.send("message_delta", map[string]interface{}{"delta": map[string]interface{}{"stop_reason": stop, "stop_sequence": nil}, "usage": reported}); err != nil {
				return false, err
			}
			if err := events.send("message_stop", map[string]interface{}{}); err != nil {
				return false, err
			}
			return true, nil
		}
		var frame messagesChatResponse
		if !utf8.Valid(data) || json.Unmarshal(data, &frame) != nil || len(frame.Error) > 0 && string(frame.Error) != "null" || len(frame.Choices) > 1 {
			return false, fmt.Errorf("invalid engine stream frame")
		}
		if frame.Usage != nil {
			update, err := messagesReportedUsage(frame, usage)
			if err != nil {
				return false, err
			}
			if frame.Usage.Input != nil {
				reported.Input = update.Input
			}
			reported.Output = update.Output
		}
		if len(frame.Choices) == 0 {
			return false, nil
		}
		choice := frame.Choices[0]
		if choice.Index != 0 || finish != "" || choice.Delta.Refusal != nil || choice.Delta.Role != "" && choice.Delta.Role != "assistant" {
			return false, fmt.Errorf("unexpected engine stream choice")
		}
		if choice.Delta.Content != nil && *choice.Delta.Content != "" {
			if textBlock < 0 {
				if blocks >= maxMessagesBlocks {
					return false, fmt.Errorf("too many content blocks")
				}
				textBlock = blocks
				blocks++
				if err := events.send("content_block_start", map[string]interface{}{"index": textBlock, "content_block": map[string]string{"type": "text", "text": ""}}); err != nil {
					return false, err
				}
			}
			if err := events.send("content_block_delta", map[string]interface{}{"index": textBlock, "delta": map[string]string{"type": "text_delta", "text": *choice.Delta.Content}}); err != nil {
				return false, err
			}
		}
		for _, delta := range choice.Delta.Tools {
			if delta.Index == nil || *delta.Index < 0 || *delta.Index >= maxMessagesBlocks || delta.Type != "" && delta.Type != "function" {
				return false, fmt.Errorf("invalid tool index or type")
			}
			tool := tools[*delta.Index]
			if tool == nil {
				tool = &messagesStreamTool{block: -1}
				tools[*delta.Index] = tool
			}
			if tool.block >= 0 && (delta.ID != "" || delta.Function.Name != "") {
				return false, fmt.Errorf("tool identity changed after argument streaming")
			}
			tool.id += delta.ID
			tool.name += delta.Function.Name
			argumentBytes += len(delta.Function.Arguments)
			if len(tool.id) > 512 || len(tool.name) > 512 || argumentBytes > maxRequestBytes {
				return false, fmt.Errorf("engine tool input exceeds adapter limits")
			}
			if delta.Function.Arguments != "" {
				if err := startTool(tool); err != nil {
					return false, err
				}
				tool.arguments.WriteString(delta.Function.Arguments)
				if err := events.send("content_block_delta", map[string]interface{}{"index": tool.block, "delta": map[string]string{"type": "input_json_delta", "partial_json": delta.Function.Arguments}}); err != nil {
					return false, err
				}
			}
		}
		if choice.Finish != nil {
			finish = *choice.Finish
		}
		return false, nil
	}
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			done, err := consume(event)
			event = nil
			if err != nil || done {
				return err
			}
		} else if bytes.HasPrefix(line, []byte("data:")) {
			part := bytes.TrimPrefix(line, []byte("data:"))
			part = bytes.TrimPrefix(part, []byte(" "))
			if len(event)+len(part)+1 > maxMessagesFrame {
				return fmt.Errorf("engine stream event too large")
			}
			event = append(event, part...)
			event = append(event, '\n')
		} else if bytes.HasPrefix(line, []byte(":")) {
			if err := events.send("ping", map[string]interface{}{}); err != nil {
				return err
			}
		}
	}
	if scanner.Err() != nil {
		return fmt.Errorf("engine stream read failed")
	}
	// SSE dispatch requires a blank line. A truncated terminal event is not success.
	return fmt.Errorf("engine stream ended before completion")
}

func (h *hub) forwardMessages(w http.ResponseWriter, r *http.Request, worker *workerConnection, in messagesRequest, translated []byte, id uint64) *usageObserver {
	usage := &usageObserver{failed: true}
	request, err := http.NewRequestWithContext(r.Context(), http.MethodPost, "http://worker/v1/chat/completions", bytes.NewReader(translated))
	if err != nil {
		writeAPIError(w, r, http.StatusBadGateway, "cannot create engine request", "adapter_error")
		return usage
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := worker.transport.RoundTrip(request)
	if err != nil {
		writeAPIError(w, r, http.StatusBadGateway, "inference connection failed", "upstream_connection_failed")
		return usage
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		status := response.StatusCode
		if status < 400 || status > 599 {
			status = http.StatusBadGateway
		}
		// Engine bodies can contain request text or secrets. Translate only status.
		writeAPIError(w, r, status, "engine rejected the translated request", "engine_error")
		return usage
	}
	kind, _, _ := mime.ParseMediaType(response.Header.Get("Content-Type"))
	messageID := fmt.Sprintf("msg_chair_%d", id)
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if in.Stream {
		if kind != "text/event-stream" {
			writeAPIError(w, r, http.StatusBadGateway, "engine did not return an event stream", "invalid_engine_response")
			return usage
		}
		w.Header().Set("Content-Type", "text/event-stream")
		events := messagesEvents{writer: w}
		err = streamMessages(response.Body, events, in.Model, messageID, usage)
		if err != nil {
			if r.Context().Err() == nil {
				_ = events.send("error", map[string]interface{}{"error": map[string]string{"type": "api_error", "message": "engine stream was invalid or interrupted"}})
			}
			return usage
		}
	} else {
		if kind != "application/json" {
			writeAPIError(w, r, http.StatusBadGateway, "engine did not return JSON", "invalid_engine_response")
			return usage
		}
		data, readErr := io.ReadAll(io.LimitReader(response.Body, maxRequestBytes+1))
		if readErr != nil || len(data) > maxRequestBytes {
			writeAPIError(w, r, http.StatusBadGateway, "engine response unreadable or oversized", "invalid_engine_response")
			return usage
		}
		message, convertErr := messagesJSONResponse(data, in.Model, messageID, usage)
		if convertErr != nil {
			writeAPIError(w, r, http.StatusBadGateway, "engine returned an invalid or unsupported response", "invalid_engine_response")
			return usage
		}
		w.Header().Set("Content-Type", "application/json")
		if json.NewEncoder(w).Encode(message) != nil {
			return usage
		}
	}
	usage.eof, usage.terminal, usage.failed = true, true, false
	return usage
}
