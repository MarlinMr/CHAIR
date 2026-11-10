// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// Messages is an explicit API adapter. Ordinary OpenAI/Ollama requests remain
// byte-for-byte forwarding paths; Anthropic-only features are never guessed.
type messagesRequest struct {
	Model     string `json:"model"`
	MaxTokens int64  `json:"max_tokens"`
	Messages  []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	System      json.RawMessage `json:"system"`
	Stream      bool            `json:"stream"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	Stop        []string        `json:"stop_sequences"`
	Tools       []struct {
		Type        string          `json:"type"`
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Schema      json.RawMessage `json:"input_schema"`
		Cache       json.RawMessage `json:"cache_control"`
		Strict      bool            `json:"strict"`
		Deferred    bool            `json:"defer_loading"`
	} `json:"tools"`
	ToolChoice *struct {
		Type            string `json:"type"`
		Name            string `json:"name"`
		DisableParallel bool   `json:"disable_parallel_tool_use"`
	} `json:"tool_choice"`
	Thinking *struct {
		Type string `json:"type"`
	} `json:"thinking"`
	OutputConfig *struct {
		Effort string `json:"effort"`
	} `json:"output_config"`
	Metadata json.RawMessage `json:"metadata"`
	Cache    json.RawMessage `json:"cache_control"`
}

type messagesBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Cache     json.RawMessage `json:"cache_control,omitempty"`
}

type messagesChatTool struct {
	Index    *int   `json:"index,omitempty"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type messagesChatMessage struct {
	Role    string             `json:"role"`
	Content *string            `json:"content"`
	Tools   []messagesChatTool `json:"tool_calls,omitempty"`
	ToolID  string             `json:"tool_call_id,omitempty"`
}

type messagesChatRequest struct {
	Model         string                `json:"model"`
	Messages      []messagesChatMessage `json:"messages"`
	MaxTokens     int64                 `json:"max_tokens"`
	Stream        bool                  `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	Tools       []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	} `json:"tools,omitempty"`
	ToolChoice      json.RawMessage `json:"tool_choice,omitempty"`
	Parallel        *bool           `json:"parallel_tool_calls,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
}

func decodeMessagesJSON(data []byte, target interface{}) error {
	if !utf8.Valid(data) {
		return fmt.Errorf("invalid UTF-8")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("invalid or unsupported Messages fields")
	}
	if decoder.Decode(new(json.RawMessage)) != io.EOF {
		return fmt.Errorf("invalid JSON")
	}
	return nil
}

func messagesBlocks(raw json.RawMessage) ([]messagesBlock, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, fmt.Errorf("content must be text or content blocks")
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return []messagesBlock{{Type: "text", Text: text}}, nil
	}
	var blocks []messagesBlock
	if err := decodeMessagesJSON(raw, &blocks); err != nil || blocks == nil || len(blocks) > maxMessagesBlocks {
		return nil, fmt.Errorf("unsupported content blocks; only text and client tool use/results are supported")
	}
	for _, block := range blocks {
		switch block.Type {
		case "text":
			if block.ID != "" || block.Name != "" || len(block.Input) > 0 || block.ToolUseID != "" || len(block.Content) > 0 || block.IsError {
				return nil, fmt.Errorf("unexpected fields in text block")
			}
		case "tool_use":
			if block.Text != "" || block.ToolUseID != "" || len(block.Content) > 0 || block.IsError {
				return nil, fmt.Errorf("unexpected fields in tool_use block")
			}
		case "tool_result":
			if block.Text != "" || block.ID != "" || block.Name != "" || len(block.Input) > 0 {
				return nil, fmt.Errorf("unexpected fields in tool_result block")
			}
		default:
			return nil, fmt.Errorf("unsupported content block; only text and client tool use/results are supported")
		}
	}
	return blocks, nil
}

func messagesText(raw json.RawMessage) (string, error) {
	blocks, err := messagesBlocks(raw)
	if err != nil {
		return "", err
	}
	var text []string
	for _, block := range blocks {
		if block.Type != "text" {
			return "", fmt.Errorf("only text content is supported here")
		}
		text = append(text, block.Text)
	}
	return strings.Join(text, "\n"), nil
}

func messagesObject(raw json.RawMessage) bool {
	var object map[string]json.RawMessage
	return utf8.Valid(raw) && json.Unmarshal(raw, &object) == nil && object != nil
}

func translateMessagesRequest(body []byte) (messagesRequest, []byte, error) {
	var in messagesRequest
	if err := decodeMessagesJSON(body, &in); err != nil {
		return in, nil, err
	}
	if !validModel(in.Model) || in.MaxTokens <= 0 || len(in.Messages) == 0 || len(in.Messages) > maxMessagesBlocks || len(in.Tools) > maxMessagesBlocks {
		return in, nil, fmt.Errorf("model, positive max_tokens, and nonempty messages are required")
	}
	if in.Thinking != nil && in.Thinking.Type != "disabled" {
		return in, nil, fmt.Errorf("extended/adaptive thinking is unsupported; disable thinking in Claude Code")
	}
	if in.Temperature != nil && (*in.Temperature < 0 || *in.Temperature > 1) || in.TopP != nil && (*in.TopP < 0 || *in.TopP > 1) {
		return in, nil, fmt.Errorf("temperature and top_p must be between zero and one")
	}
	out := messagesChatRequest{Model: in.Model, MaxTokens: in.MaxTokens, Stream: in.Stream, Temperature: in.Temperature, TopP: in.TopP, Stop: in.Stop}
	if in.OutputConfig != nil {
		switch in.OutputConfig.Effort {
		case "", "low", "medium", "high", "xhigh", "max":
			out.ReasoningEffort = in.OutputConfig.Effort
		default:
			return in, nil, fmt.Errorf("unsupported effort level")
		}
	}
	if in.Stream {
		out.StreamOptions = &struct {
			IncludeUsage bool `json:"include_usage"`
		}{true}
	}
	if len(in.System) > 0 {
		text, err := messagesText(in.System)
		if err != nil {
			return in, nil, err
		}
		out.Messages = append(out.Messages, messagesChatMessage{Role: "system", Content: &text})
	}
	toolNames := make(map[string]bool)
	for _, tool := range in.Tools {
		if (tool.Type != "" && tool.Type != "custom") || tool.Strict || tool.Deferred || !validModel(tool.Name) || !messagesObject(tool.Schema) || toolNames[tool.Name] {
			return in, nil, fmt.Errorf("only unique custom tools with input_schema are supported; server tools, strict tools, and tool search are unsupported")
		}
		toolNames[tool.Name] = true
		out.Tools = append(out.Tools, struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		}{Type: "function", Function: struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		}{tool.Name, tool.Description, tool.Schema}})
	}
	if choice := in.ToolChoice; choice != nil {
		switch choice.Type {
		case "auto", "none":
			out.ToolChoice, _ = json.Marshal(choice.Type)
		case "any":
			out.ToolChoice = json.RawMessage(`"required"`)
		case "tool":
			if !toolNames[choice.Name] {
				return in, nil, fmt.Errorf("tool_choice names an undefined tool")
			}
			out.ToolChoice, _ = json.Marshal(struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}{Type: "function", Function: struct {
				Name string `json:"name"`
			}{choice.Name}})
		default:
			return in, nil, fmt.Errorf("unsupported tool_choice")
		}
		if len(in.Tools) == 0 && choice.Type != "none" {
			return in, nil, fmt.Errorf("tool_choice requires tools")
		}
		if choice.DisableParallel {
			parallel := false
			out.Parallel = &parallel
		}
	}
	pending, used := make(map[string]bool), make(map[string]bool)
	for _, message := range in.Messages {
		if message.Role != "user" && message.Role != "assistant" {
			return in, nil, fmt.Errorf("messages roles must be user or assistant")
		}
		blocks, err := messagesBlocks(message.Content)
		if err != nil {
			return in, nil, err
		}
		if len(blocks) == 0 {
			return in, nil, fmt.Errorf("message content must be nonempty")
		}
		chat := messagesChatMessage{Role: message.Role}
		var text []string
		for _, block := range blocks {
			switch block.Type {
			case "text":
				if len(pending) > 0 {
					return in, nil, fmt.Errorf("tool results must precede new conversation content")
				}
				text = append(text, block.Text)
			case "tool_use":
				if message.Role != "assistant" || !validModel(block.ID) || used[block.ID] || !validModel(block.Name) || !messagesObject(block.Input) {
					return in, nil, fmt.Errorf("invalid assistant tool use")
				}
				call := messagesChatTool{ID: block.ID, Type: "function"}
				call.Function.Name = block.Name
				call.Function.Arguments = string(block.Input)
				chat.Tools = append(chat.Tools, call)
				used[block.ID] = true
			case "tool_result":
				if message.Role != "user" || !pending[block.ToolUseID] || len(text) > 0 {
					return in, nil, fmt.Errorf("tool results must match pending calls and precede user text")
				}
				result := ""
				if len(block.Content) > 0 {
					result, err = messagesText(block.Content)
					if err != nil {
						return in, nil, err
					}
				}
				if block.IsError {
					result = "[tool error]\n" + result
				}
				out.Messages = append(out.Messages, messagesChatMessage{Role: "tool", ToolID: block.ToolUseID, Content: &result})
				delete(pending, block.ToolUseID)
			default:
				return in, nil, fmt.Errorf("unsupported content block; images, documents, thinking, and server tools are not implemented")
			}
		}
		if len(pending) > 0 {
			return in, nil, fmt.Errorf("every tool call requires a matching result before the next turn")
		}
		if len(text) > 0 {
			joined := strings.Join(text, "\n")
			chat.Content = &joined
		}
		if chat.Content != nil || len(chat.Tools) > 0 {
			out.Messages = append(out.Messages, chat)
		}
		for _, tool := range chat.Tools {
			pending[tool.ID] = true
		}
	}
	if len(pending) > 0 || in.Messages[len(in.Messages)-1].Role != "user" || len(out.Messages) == 0 || (out.Messages[len(out.Messages)-1].Role != "user" && out.Messages[len(out.Messages)-1].Role != "tool") {
		return in, nil, fmt.Errorf("request must end with user content or complete tool results; assistant prefilling is unsupported")
	}
	translated, err := json.Marshal(out)
	if err != nil || len(translated) > maxRequestBytes {
		return in, nil, fmt.Errorf("translated request is too large")
	}
	return in, translated, nil
}
