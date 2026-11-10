// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
)

func commandLogger(value string, output io.Writer) (*slog.Logger, error) {
	var level slog.Level
	switch value {
	case "debug":
		level = slog.LevelDebug
	case "info":
		level = slog.LevelInfo
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		return nil, fmt.Errorf("--log-level must be debug, info, warn, or error")
	}
	return slog.New(slog.NewTextHandler(output, &slog.HandlerOptions{Level: level})), nil
}

// Library errors can contain URLs, headers, certificate details, or response
// contents. Diagnostics use fixed reason codes, never the original error text.
func connectionReason(err error) string {
	var certificate *tls.CertificateVerificationError
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return "connection_timeout"
	case errors.As(err, &certificate):
		return "tls_verification_failed"
	default:
		return "connection_failed"
	}
}

type requestLogWriter struct {
	http.ResponseWriter
	status int
	bytes  int64
}

func (w *requestLogWriter) WriteHeader(status int) {
	if status >= http.StatusOK && w.status == 0 {
		w.status = status
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *requestLogWriter) Write(body []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	n, err := w.ResponseWriter.Write(body)
	w.bytes += int64(n)
	return n, err
}

// Preserve ResponseController's streaming/flush support without logging chunks.
func (w *requestLogWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}
