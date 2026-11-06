// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime"
	"strings"
)

const maxUsageFrame = 256 << 10

// A bounded, transparent observer, never a response buffer or content log.
// Only numeric usage and completion/error flags survive parsing a frame.
type usageObserver struct {
	path, format                           string
	line, event, body                      []byte
	discardLine, discardEvent, discardBody bool
	eof, terminal, failed, invalid         bool
	tokens                                 *int64
}

func newUsageObserver(path, contentType string) *usageObserver {
	kind, _, _ := mime.ParseMediaType(contentType)
	format := ""
	switch {
	case kind == "text/event-stream":
		format = "sse"
	case kind == "application/x-ndjson" || kind == "application/ndjson":
		format = "ndjson"
	case kind == "application/json" || strings.HasSuffix(kind, "+json"):
		format = "json"
	}
	return &usageObserver{path: path, format: format}
}

func (u *usageObserver) feed(data []byte) {
	if u.format == "json" {
		if len(u.body)+len(data) > maxUsageFrame {
			u.body = nil
			u.discardBody = true
		}
		if !u.discardBody {
			u.body = append(u.body, data...)
		}
		return
	}
	if u.format != "sse" && u.format != "ndjson" {
		return
	}
	for len(data) > 0 {
		index := bytes.IndexByte(data, '\n')
		part := data
		if index >= 0 {
			part = data[:index]
		}
		if len(u.line)+len(part) > maxUsageFrame {
			u.line = nil
			u.discardLine = true
		}
		if !u.discardLine {
			u.line = append(u.line, part...)
		}
		if index < 0 {
			return
		}
		u.consumeLine()
		data = data[index+1:]
	}
}

func (u *usageObserver) consumeLine() {
	if u.discardLine {
		u.discardEvent = true
	} else {
		line := bytes.TrimSuffix(u.line, []byte{'\r'})
		if u.format == "ndjson" {
			u.parse(line)
		} else if len(line) == 0 {
			u.consumeEvent()
		} else if bytes.HasPrefix(line, []byte("data:")) && !u.discardEvent {
			if len(u.event)+len(line) > maxUsageFrame {
				u.event = nil
				u.discardEvent = true
			} else {
				u.event = append(u.event, bytes.TrimSpace(line[5:])...)
				u.event = append(u.event, '\n')
			}
		}
	}
	u.line = nil
	u.discardLine = false
}

func (u *usageObserver) consumeEvent() {
	if !u.discardEvent {
		data := bytes.TrimSpace(u.event)
		if bytes.Equal(data, []byte("[DONE]")) {
			u.terminal = true
		} else {
			u.parse(data)
		}
	}
	u.event = nil
	u.discardEvent = false
}

func (u *usageObserver) finish() {
	if u.eof {
		return
	}
	u.eof = true
	if u.format == "json" && !u.discardBody {
		u.parse(u.body)
	}
	if len(u.line) > 0 || u.discardLine {
		u.consumeLine()
	}
	if u.format == "sse" {
		u.consumeEvent()
	}
	u.body, u.event, u.line = nil, nil, nil
}

func (u *usageObserver) accept(tokens *int64) {
	if tokens == nil {
		return
	}
	if *tokens < 0 || *tokens > 1_000_000_000 {
		u.invalid = true
		return
	}
	if u.tokens != nil && *u.tokens != *tokens {
		u.invalid = true
		return
	}
	value := *tokens
	u.tokens = &value
}

func (u *usageObserver) parse(data []byte) {
	type usage struct {
		Completion *int64 `json:"completion_tokens"`
		Output     *int64 `json:"output_tokens"`
	}
	var frame struct {
		Usage     *usage          `json:"usage"`
		EvalCount *int64          `json:"eval_count"`
		Done      bool            `json:"done"`
		Type      string          `json:"type"`
		Status    string          `json:"status"`
		Error     json.RawMessage `json:"error"`
		Response  *struct {
			Usage  *usage `json:"usage"`
			Status string `json:"status"`
		} `json:"response"`
	}
	if len(data) == 0 {
		return
	}
	if json.Unmarshal(data, &frame) != nil {
		return
	}
	if len(frame.Error) > 0 && !bytes.Equal(bytes.TrimSpace(frame.Error), []byte("null")) {
		u.failed = true
	}
	if frame.Type == "error" {
		u.failed = true
	}
	switch u.path {
	case "/api/chat", "/api/generate":
		if frame.Done {
			u.terminal = true
			u.accept(frame.EvalCount)
		}
	case "/v1/chat/completions":
		if u.format == "json" {
			u.terminal = true
		}
		if frame.Usage != nil {
			u.accept(frame.Usage.Completion)
		}
	case "/v1/responses":
		if frame.Type == "response.failed" || frame.Type == "response.incomplete" || frame.Status == "failed" || frame.Status == "incomplete" {
			u.failed = true
		}
		if frame.Type == "response.completed" && frame.Response != nil && frame.Response.Status == "completed" {
			u.terminal = true
			if frame.Response.Usage != nil {
				u.accept(frame.Response.Usage.Output)
			}
		} else if u.format == "json" && frame.Status == "completed" {
			u.terminal = true
			if frame.Usage != nil {
				u.accept(frame.Usage.Output)
			}
		}
	}
}

type observedBody struct {
	io.ReadCloser
	usage *usageObserver
}

func (b *observedBody) Read(data []byte) (int, error) {
	n, err := b.ReadCloser.Read(data)
	if n > 0 {
		b.usage.feed(data[:n])
	}
	if err == io.EOF {
		b.usage.finish()
	}
	return n, err
}

func (b *observedBody) Close() error {
	b.usage.body, b.usage.line, b.usage.event = nil, nil, nil
	return b.ReadCloser.Close()
}
