// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if err := run(ctx, os.Args[1:]); err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}
}

func tokenFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("cannot open token file")
	}
	defer f.Close()
	return readToken(f)
}

func run(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: chair hub|worker [flags]")
	}
	switch args[0] {
	case "hub":
		return runHub(ctx, args[1:])
	case "worker":
		return runWorker(ctx, args[1:])
	default:
		return fmt.Errorf("expected hub or worker")
	}
}

func runHub(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("hub", flag.ContinueOnError)
	listen := flags.String("listen", "127.0.0.1:8443", "TLS listen address; explicitly choose an interface for remote access")
	cert := flags.String("tls-cert", "", "hub TLS certificate file")
	key := flags.String("tls-key", "", "hub TLS private key file")
	clientPath := flags.String("client-token-file", "", "inference client bearer token file")
	workersPath := flags.String("workers-file", "", "JSON file of authorized worker IDs and distinct bearer tokens")
	maxConcurrentRequests := flags.Int("max-concurrent-requests", defaultMaxConcurrentRequests, "maximum simultaneous inference requests across all workers (must be positive)")
	logLevel := flags.String("log-level", "info", "logging verbosity: debug, info, warn, or error")
	dashboardPath := flags.String("dashboard-token-file", "", "enable read-only web dashboard using a separate token file")
	dashboardPublic := flags.Bool("dashboard-public", false, "allow dashboard, setup guide, and downloads without login; inference remains authenticated")
	downloadsDir := flags.String("downloads-dir", "", "directory of manually provisioned CHAIR release archives and SHA256SUMS")
	allowTokenReveal := flags.Bool("allow-client-token-reveal", false, "allow dashboard-authenticated users to explicitly reveal the inference client token")
	enrollmentPath := flags.String("enrollment-token-file", "", "enable /join using a separate shared enrollment token; persist generated worker tokens in --workers-file")
	if err := flags.Parse(args); err != nil {
		return err
	}
	logger, err := commandLogger(*logLevel, os.Stderr)
	if err != nil {
		return err
	}
	logger = logger.With("role", "hub")
	if *maxConcurrentRequests < 1 {
		return fmt.Errorf("--max-concurrent-requests must be positive")
	}
	if flags.NArg() != 0 || *cert == "" || *key == "" {
		return fmt.Errorf("hub requires --tls-cert and --tls-key and accepts no positional arguments")
	}
	clientToken, err := tokenFile(*clientPath)
	if err != nil {
		return fmt.Errorf("client token: %w", err)
	}
	workers, err := os.Open(*workersPath)
	if err != nil {
		return fmt.Errorf("cannot open worker credentials file")
	}
	workerTokens, err := readWorkerTokens(workers, clientToken)
	closeErr := workers.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return fmt.Errorf("cannot close worker credentials file")
	}
	h := newHub(clientToken, workerTokens, *maxConcurrentRequests)
	h.logger = logger
	if *dashboardPath != "" {
		token, err := tokenFile(*dashboardPath)
		if err != nil {
			return fmt.Errorf("dashboard token: %w", err)
		}
		if err := h.enableDashboard(token); err != nil {
			return err
		}
		logger.Info("read-only dashboard enabled", "path", "/")
	}
	if err := h.configureDashboard(*dashboardPublic, *allowTokenReveal, *downloadsDir); err != nil {
		return err
	}
	if h.downloadsRoot != nil {
		defer h.downloadsRoot.Close()
	}
	if h.dashboardPublic {
		logger.Warn("public dashboard enabled", "path", "/")
	}
	if h.allowClientTokenReveal {
		logger.Warn("client-token reveal enabled; dashboard credentials grant inference access")
	}
	if *enrollmentPath != "" {
		token, err := tokenFile(*enrollmentPath)
		if err != nil {
			return fmt.Errorf("enrollment token: %w", err)
		}
		if err := h.enableEnrollment(token, *workersPath); err != nil {
			return err
		}
		defer h.enrollment.root.Close()
		logger.Warn("worker enrollment enabled; enrollment credential holders can receive inference requests", "path", "/join")
	}
	defer h.Close()
	server := &http.Server{
		Addr:              *listen,
		Handler:           h,
		TLSConfig:         &tls.Config{MinVersion: tls.VersionTLS12},
		ReadHeaderTimeout: connectionTimeout,
		MaxHeaderBytes:    64 << 10,
		ErrorLog:          log.New(io.Discard, "", 0),
		BaseContext:       func(net.Listener) context.Context { return ctx },
	}
	stop := context.AfterFunc(ctx, func() { _ = h.Close(); _ = server.Close() })
	defer stop()
	logger.Info("starting experimental CHAIR TLS hub", "listen", *listen, "max_concurrent_requests", *maxConcurrentRequests, "authorized_workers", len(workerTokens))
	if err := server.ListenAndServeTLS(*cert, *key); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("cannot serve hub; check listen address and TLS files")
	}
	return nil
}

func runWorker(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("worker", flag.ContinueOnError)
	address := flags.String("hub", "", "hub HTTPS base URL")
	tokenPath := flags.String("worker-token-file", "", "worker bearer token file")
	id := flags.String("worker-id", "", "stable worker ID authorized in the hub's workers file")
	engine := flags.String("engine", "http://127.0.0.1:11434", "local Ollama engine URL (literal loopback address only)")
	ca := flags.String("ca-file", "", "additional trusted PEM CA certificates for the hub")
	logLevel := flags.String("log-level", "info", "logging verbosity: debug, info, warn, or error")
	if err := flags.Parse(args); err != nil {
		return err
	}
	logger, err := commandLogger(*logLevel, os.Stderr)
	if err != nil {
		return err
	}
	logger = logger.With("role", "worker")
	if flags.NArg() != 0 {
		return fmt.Errorf("worker accepts no positional arguments")
	}
	if !validWorkerID(*id) {
		return fmt.Errorf("worker ID must be 1 to 64 letters, digits, dots, underscores, or hyphens")
	}
	u, err := hubURL(*address)
	if err != nil {
		return err
	}
	target, err := engineURL(*engine)
	if err != nil {
		return err
	}
	token, err := tokenFile(*tokenPath)
	if err != nil {
		return err
	}
	config := &tls.Config{MinVersion: tls.VersionTLS12}
	if *ca != "" {
		roots, err := x509.SystemCertPool()
		if err != nil {
			return fmt.Errorf("cannot load system trust store")
		}
		pem, err := os.ReadFile(*ca)
		if err != nil || !roots.AppendCertsFromPEM(pem) {
			return fmt.Errorf("cannot load CA certificates")
		}
		config.RootCAs = roots
	}
	transport := &http.Transport{
		TLSClientConfig:       config,
		TLSHandshakeTimeout:   connectionTimeout,
		DialContext:           (&net.Dialer{Timeout: connectionTimeout}).DialContext,
		ResponseHeaderTimeout: connectionTimeout,
	}
	defer transport.CloseIdleConnections()
	client := &http.Client{
		Transport:     transport,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	delay := time.Second
	hardware := detectHardware(ctx)
	for ctx.Err() == nil {
		started := time.Now()
		logger.Info("connecting experimental CHAIR worker", "worker_id", *id)
		_ = connectToHub(ctx, client, u, *id, token, target, logger, hardware)
		if ctx.Err() != nil {
			break
		}
		if time.Since(started) > time.Minute {
			delay = time.Second
		}
		logger.Warn("worker disconnected; retrying", "worker_id", *id, "delay", delay)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
		case <-timer.C:
		}
		delay = min(2*delay, 30*time.Second)
	}
	return nil
}
