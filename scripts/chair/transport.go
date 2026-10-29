// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/hashicorp/yamux"
)

const (
	workerPath                   = "/worker"
	protocol                     = "chair-transport-v1"
	defaultMaxConcurrentRequests = 16
	maxRequestBytes              = 8 << 20
	connectionTimeout            = 10 * time.Second
)

func authenticated(r *http.Request, token string) bool {
	expected := sha256.Sum256([]byte("Bearer " + token))
	actual := sha256.Sum256([]byte(r.Header.Get("Authorization")))
	return subtle.ConstantTimeCompare(expected[:], actual[:]) == 1
}

func allowedRequest(r *http.Request) bool {
	if r.URL.RawQuery != "" {
		return false
	}
	switch r.URL.Path {
	case "/api/chat", "/api/generate":
		return r.Method == http.MethodPost
	case "/api/tags":
		return r.Method == http.MethodGet
	default:
		return false
	}
}

// Only inference headers cross the boundary. In particular, neither client nor
// worker credentials, cookies, forwarding headers, or engine probe headers do.
func inferenceHeaders(in http.Header) http.Header {
	out := make(http.Header)
	for _, name := range []string{"Content-Type", "Accept"} {
		if value := in.Get(name); value != "" {
			out.Set(name, value)
		}
	}
	return out
}

func inferenceProxy(target *url.URL, transport http.RoundTripper, logger *slog.Logger) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(target)
			r.Out.Header = inferenceHeaders(r.In.Header)
			r.Out.Host = target.Host
		},
		Transport:     transport,
		FlushInterval: -1,
		ErrorLog:      log.New(io.Discard, "", 0),
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, _ error) {
			if r.Context().Err() == nil {
				logger.Warn("inference forwarding failed", "reason", "upstream_connection_failed")
			}
			http.Error(w, "inference connection failed", http.StatusBadGateway)
		},
	}
}

func sessionConfig() *yamux.Config {
	config := yamux.DefaultConfig()
	config.KeepAliveInterval = 15 * time.Second
	config.ConnectionWriteTimeout = connectionTimeout
	config.StreamOpenTimeout = connectionTimeout
	config.StreamCloseTimeout = connectionTimeout
	config.LogOutput = io.Discard
	return config
}

func engineURL(address string) (*url.URL, error) {
	u, err := url.Parse(address)
	if err != nil {
		return nil, fmt.Errorf("invalid engine URL")
	}
	ip := net.ParseIP(u.Hostname())
	if u.Scheme != "http" || ip == nil || !ip.IsLoopback() || u.User != nil ||
		u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") {
		return nil, fmt.Errorf("engine URL must be HTTP on a literal loopback address, with no credentials, path, query, or fragment")
	}
	u.Path = ""
	return u, nil
}

func hubURL(address string) (*url.URL, error) {
	u, err := url.Parse(address)
	if err != nil {
		return nil, fmt.Errorf("invalid hub URL")
	}
	if u.Scheme != "https" || u.Hostname() == "" || u.User != nil ||
		u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") {
		return nil, fmt.Errorf("hub URL must be HTTPS with no credentials, path, query, or fragment")
	}
	u.Scheme = "wss"
	u.Path = workerPath
	return u, nil
}

func readToken(r io.Reader) (string, error) {
	data, err := io.ReadAll(io.LimitReader(r, 4097))
	if err != nil {
		return "", fmt.Errorf("cannot read token")
	}
	token := strings.TrimSpace(string(data))
	if len(data) > 4096 || len(token) < 32 {
		return "", fmt.Errorf("token must contain 32 to 4096 non-whitespace ASCII characters")
	}
	for _, char := range token {
		if char < 33 || char > 126 {
			return "", fmt.Errorf("token must contain only non-whitespace ASCII characters")
		}
	}
	return token, nil
}

func localTransport() *http.Transport {
	return &http.Transport{
		// Never consult HTTP_PROXY for a worker's private engine.
		DialContext:            (&net.Dialer{Timeout: connectionTimeout}).DialContext,
		DisableKeepAlives:      true,
		ResponseHeaderTimeout:  5 * time.Minute,
		MaxResponseHeaderBytes: 64 << 10,
	}
}

// sessionListener accepts logical connections, never operating-system sockets.
type sessionListener struct {
	*yamux.Session
}

func (s sessionListener) Addr() net.Addr {
	return s.Session.LocalAddr()
}

func serveEngine(ctx context.Context, session *yamux.Session, target *url.URL, logger *slog.Logger) error {
	transport := localTransport()
	defer transport.CloseIdleConnections()
	proxy := inferenceProxy(target, transport, logger)
	server := &http.Server{
		ReadHeaderTimeout: connectionTimeout,
		MaxHeaderBytes:    64 << 10,
		ErrorLog:          log.New(io.Discard, "", 0),
		BaseContext:       func(net.Listener) context.Context { return ctx },
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !allowedRequest(r) {
				http.Error(w, "unsupported inference route", http.StatusNotFound)
				return
			}
			if r.ContentLength > maxRequestBytes {
				http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
			proxy.ServeHTTP(w, r)
		}),
	}
	stop := context.AfterFunc(ctx, func() { _ = server.Close(); _ = session.Close() })
	defer stop()
	defer server.Close()
	return server.Serve(sessionListener{session})
}
