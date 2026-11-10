// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/json"
	"html/template"
	"net/http"
)

var scoreboardTemplate = template.Must(template.New("scoreboard.html").Funcs(template.FuncMap{
	"rank": func(index int) int { return index + 1 },
}).ParseFS(dashboardAssets, "web/scoreboard.html"))

func (h *hub) serveScoreboard(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/dashboard/scoreboard.js" {
		data, err := dashboardAssets.ReadFile("web/scoreboard.js")
		if err != nil {
			http.Error(w, "scoreboard script unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		_, _ = w.Write(data)
		return
	}
	view, err := h.stats.snapshot(r.URL.Query().Get("model"))
	if err != nil {
		http.Error(w, "statistics temporarily unavailable", http.StatusServiceUnavailable)
		return
	}
	if r.URL.Path == "/dashboard/scoreboard/status" {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(view)
		return
	}
	var page bytes.Buffer
	if err := scoreboardTemplate.Execute(&page, view); err != nil {
		http.Error(w, "scoreboard unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(page.Bytes())
}
