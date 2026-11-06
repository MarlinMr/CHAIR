// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
)

// The hub keeps no response ownership or conversation store. Require an explicit
// storage opt-out and self-contained input rather than silently losing affinity.
// Reject duplicate/case-variant control keys so parsers cannot disagree about
// which value governs storage or execution. The original body is forwarded.
func validStatelessResponse(body []byte) bool {
	fields, ok := responseObject(body)
	if !ok {
		return false
	}
	if !bytes.Equal(bytes.TrimSpace(fields["store"]), []byte("false")) {
		return false
	}
	for _, key := range []string{"previous_response_id", "conversation"} {
		if value := fields[key]; len(value) != 0 && !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return false
		}
	}
	if value := fields["background"]; len(value) != 0 && !bytes.Equal(bytes.TrimSpace(value), []byte("false")) && !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
		return false
	}
	input := bytes.TrimSpace(fields["input"])
	if len(input) > 0 && input[0] == '[' {
		var items []json.RawMessage
		if json.Unmarshal(input, &items) != nil {
			return false
		}
		for _, item := range items {
			fields, ok := responseObject(item)
			if !ok {
				return false
			}
			var kind string
			if value := fields["type"]; len(value) != 0 && json.Unmarshal(value, &kind) != nil {
				return false
			}
			if kind == "item_reference" {
				return false
			}
		}
	}
	return true
}

func responseObject(body []byte) (map[string]json.RawMessage, bool) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, false
	}
	fields := make(map[string]json.RawMessage)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, false
		}
		key, ok := token.(string)
		if !ok {
			return nil, false
		}
		canonical := strings.ToLower(key)
		if _, duplicate := fields[canonical]; duplicate {
			return nil, false
		}
		// Only canonical spellings of the controls we interpret are accepted.
		switch canonical {
		case "store", "background", "previous_response_id", "conversation", "input", "type":
			if key != canonical {
				return nil, false
			}
		}
		var value json.RawMessage
		if decoder.Decode(&value) != nil {
			return nil, false
		}
		fields[canonical] = value
	}
	if _, err := decoder.Token(); err != nil {
		return nil, false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, false
	}
	return fields, true
}
