// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"
)

type dashboardDownload struct {
	Name  string
	Label string
}

// An explicit list, not a file server: no directory indexes or arbitrary files.
var releaseDownloads = []dashboardDownload{
	{Name: "chair-linux-amd64.tar.gz", Label: "Linux · x64"},
	{Name: "chair-linux-arm64.tar.gz", Label: "Linux · ARM64"},
	{Name: "chair-darwin-amd64.tar.gz", Label: "macOS · Intel"},
	{Name: "chair-darwin-arm64.tar.gz", Label: "macOS · Apple silicon"},
	{Name: "chair-windows-amd64.zip", Label: "Windows · x64"},
	{Name: "chair-windows-arm64.zip", Label: "Windows · ARM64"},
	{Name: "SHA256SUMS", Label: "SHA-256 checksums"},
}

func (h *hub) configureDashboard(public, reveal bool, downloadsDir string) error {
	if reveal && h.dashboardToken == "" {
		return fmt.Errorf("--allow-client-token-reveal requires --dashboard-token-file, even with --dashboard-public")
	}
	if downloadsDir != "" {
		if !public && h.dashboardToken == "" {
			return fmt.Errorf("--downloads-dir requires --dashboard-token-file or --dashboard-public")
		}
		root, err := os.OpenRoot(downloadsDir)
		if err != nil {
			return fmt.Errorf("cannot open downloads directory")
		}
		h.downloadsRoot = root
	}
	h.dashboardPublic, h.allowClientTokenReveal = public, reveal
	return nil
}

func (h *hub) dashboardDownloads() []dashboardDownload {
	downloads := []dashboardDownload{}
	if h.downloadsRoot == nil {
		return downloads
	}
	for _, download := range releaseDownloads {
		info, err := h.downloadsRoot.Lstat(download.Name)
		if err == nil && info.Mode().IsRegular() {
			downloads = append(downloads, download)
		}
	}
	return downloads
}

func (h *hub) serveDownload(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/dashboard/downloads/")
	allowed := false
	for _, download := range releaseDownloads {
		if name == download.Name {
			allowed = true
			break
		}
	}
	if !allowed || h.downloadsRoot == nil {
		http.NotFound(w, r)
		return
	}
	before, err := h.downloadsRoot.Lstat(name)
	if err != nil || !before.Mode().IsRegular() {
		http.NotFound(w, r)
		return
	}
	file, err := h.downloadsRoot.Open(name)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	after, err := file.Stat()
	// Root confines resolution. Verify that the opened object is the same
	// regular file observed by Lstat before sending any bytes.
	if err != nil || !after.Mode().IsRegular() || !os.SameFile(before, after) {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Disposition", `attachment; filename="`+name+`"`)
	w.Header().Set("Content-Type", "application/octet-stream")
	http.ServeContent(w, r, name, after.ModTime(), file)
}
