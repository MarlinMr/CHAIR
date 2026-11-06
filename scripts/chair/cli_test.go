// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Re-execute the test binary to exercise main's real output and exit status
// without redirecting process-global streams in the running test suite.
func TestCLI_Process(t *testing.T) {
	if os.Getenv("CHAIR_CLI_TEST_PROCESS") != "1" {
		return
	}
	for index, arg := range os.Args {
		if arg == "--" {
			os.Args = append([]string{"chair"}, os.Args[index+1:]...)
			main()
			os.Exit(0)
		}
	}
	t.Fatal("missing CLI argument separator")
}

func chairCommand(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	executable, err := os.Executable()
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, executable, append([]string{"-test.run=^TestCLI_Process$", "--"}, args...)...)
	command.Env = append(os.Environ(), "CHAIR_CLI_TEST_PROCESS=1")
	var stdout, stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr
	err = command.Run()
	return stdout.String(), stderr.String(), err
}

func TestCLI_VersionPrintsWithoutStartingServices(t *testing.T) {
	test := func(name string, args ...string) {
		t.Run(name, func(t *testing.T) {
			stdout, stderr, err := chairCommand(t, args...)

			require.NoError(t, err)
			assert.Equal(t, "CHAIR v0.1.8\n", stdout)
			assert.Empty(t, stderr)
		})
	}
	test("root version", "--version")
	test("hub version without credentials", "hub", "--version")
	test("worker version without credentials", "worker", "--version")
}

func TestCLI_HelpStartsWithVersionAndExitsSuccessfully(t *testing.T) {
	test := func(name, usage, expectedFlag string, args ...string) {
		t.Run(name, func(t *testing.T) {
			stdout, stderr, err := chairCommand(t, args...)

			require.NoError(t, err)
			assert.Empty(t, stdout)
			assert.True(t, strings.HasPrefix(stderr, "CHAIR v0.1.8 — Central Hub for AI Routing\n"))
			assert.Contains(t, stderr, usage)
			assert.Contains(t, stderr, expectedFlag)
			assert.Contains(t, stderr, "-version")
			assert.NotContains(t, stderr, "ERROR")
		})
	}
	test("root help", "Usage: chair hub|worker [flags]", "Commands:", "--help")
	test("short root help", "Usage: chair hub|worker [flags]", "Commands:", "-h")
	test("hub help", "Usage: chair hub [flags]", "-tls-cert", "hub", "--help")
	test("worker help", "Usage: chair worker [flags]", "-engine-type", "worker", "--help")
}

func TestCLI_InvalidArgumentsStillFail(t *testing.T) {
	test := func(name string, args ...string) {
		t.Run(name, func(t *testing.T) {
			_, stderr, err := chairCommand(t, args...)

			require.Error(t, err)
			assert.Contains(t, stderr, "ERROR")
		})
	}
	test("missing command")
	test("unknown command", "invalid-command")
	test("unknown flag", "--invalid-flag")
	test("missing hub credentials", "hub")
	test("missing worker identity", "worker")
}
