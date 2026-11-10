// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"net"
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHubStartup_ExplainsBindingFailuresWithoutLeakingRawErrors(t *testing.T) {
	test := func(name string, cause error, message string) {
		t.Run(name, func(t *testing.T) {
			err := hubServeError(&net.OpError{Op: "listen", Net: "tcp", Err: cause})
			assert.Contains(t, err.Error(), message)
		})
	}
	test("permission denied", syscall.EACCES, "permission denied")
	test("operation prohibited", syscall.EPERM, "permission denied")
	test("port occupied", syscall.EADDRINUSE, "address already in use")
	test("other network error", errors.New("PRIVATE_RAW_ERROR"), "check that the address is valid")
	assert.NotContains(t, hubServeError(&net.OpError{Op: "listen", Err: errors.New("PRIVATE_RAW_ERROR")}).Error(), "PRIVATE_RAW_ERROR")
}

func TestHubStartup_ReportsTLSFileProblemsWithoutExposingPaths(t *testing.T) {
	err := hubServeError(&os.PathError{Op: "open", Path: "/private/cert.pem", Err: syscall.EACCES})
	assert.EqualError(t, err, "cannot serve hub; check TLS certificate and key files and their permissions")
}
