// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type workerCredential struct {
	ID    string `json:"id"`
	Token string `json:"token"`
}

type workerCredentials struct {
	Workers []workerCredential `json:"workers"`
}

func validWorkerID(id string) bool {
	if len(id) == 0 || len(id) > 64 {
		return false
	}
	for _, c := range id {
		if !(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') &&
			!(c >= '0' && c <= '9') && c != '.' && c != '_' && c != '-' {
			return false
		}
	}
	return true
}

func readWorkerTokens(reader io.Reader, clientToken string) (map[string]string, error) {
	body, err := io.ReadAll(io.LimitReader(reader, maxInventoryBytes+1))
	if err != nil || len(body) > maxInventoryBytes {
		return nil, fmt.Errorf("worker credentials file unreadable or oversized")
	}
	var config workerCredentials
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		// JSON errors can contain credentials. Return only a fixed diagnostic.
		return nil, fmt.Errorf("invalid worker credentials JSON")
	}
	if decoder.Decode(&struct{}{}) != io.EOF || config.Workers == nil || len(config.Workers) > 256 {
		return nil, fmt.Errorf("worker credentials must contain one JSON object with 0 to 256 workers")
	}
	tokens := make(map[string]string, len(config.Workers))
	seenTokens := map[string]bool{clientToken: true}
	for _, worker := range config.Workers {
		if !validWorkerID(worker.ID) {
			return nil, fmt.Errorf("invalid worker ID in credentials file")
		}
		token, err := readToken(strings.NewReader(worker.Token))
		if err != nil || token != worker.Token {
			return nil, fmt.Errorf("invalid worker token in credentials file")
		}
		if _, duplicate := tokens[worker.ID]; duplicate || seenTokens[token] {
			return nil, fmt.Errorf("worker IDs and tokens must be distinct; worker tokens must differ from the client token")
		}
		tokens[worker.ID] = token
		seenTokens[token] = true
	}
	return tokens, nil
}
