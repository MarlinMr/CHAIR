// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

//go:embed web/dashboard.html web/dashboard.css web/dashboard.js web/join.html web/enrollment.js README.md
var dashboardAssets embed.FS

var dashboardTemplate = template.Must(template.ParseFS(dashboardAssets, "web/dashboard.html"))
var joinTemplate = template.Must(template.ParseFS(dashboardAssets, "web/join.html"))

type activeJob struct {
	ID        uint64    `json:"id"`
	WorkerID  string    `json:"worker_id"`
	Model     string    `json:"model"`
	Path      string    `json:"path"`
	StartedAt time.Time `json:"started_at"`
}

type dashboardJob struct {
	activeJob
	Elapsed string `json:"elapsed"`
}

type dashboardWorker struct {
	ID           string   `json:"id"`
	Status       string   `json:"status"`
	StatusClass  string   `json:"-"`
	Models       []string `json:"models"`
	Inflight     int      `json:"inflight"`
	InventoryAge string   `json:"inventory_age"`
	Hardware     string   `json:"hardware"`
}

type dashboardView struct {
	GeneratedAt       time.Time           `json:"generated_at"`
	Authorized        int                 `json:"authorized_workers"`
	Connected         int                 `json:"connected_workers"`
	Ready             int                 `json:"ready_workers"`
	Models            int                 `json:"available_models"`
	Limit             int                 `json:"max_concurrent_requests"`
	Workers           []dashboardWorker   `json:"workers"`
	Jobs              []dashboardJob      `json:"jobs"`
	Live              bool                `json:"-"`
	PagePath          string              `json:"-"`
	Public            bool                `json:"-"`
	RevealEnabled     bool                `json:"-"`
	Authenticated     bool                `json:"-"`
	Downloads         []dashboardDownload `json:"-"`
	DownloadsEnabled  bool                `json:"-"`
	EnrollmentEnabled bool                `json:"-"`
}

func (h *hub) enableDashboard(token string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if token == "" || token == h.clientToken {
		return fmt.Errorf("dashboard token must be distinct from client and worker tokens")
	}
	for _, workerToken := range h.workerTokens {
		if token == workerToken {
			return fmt.Errorf("dashboard token must be distinct from client and worker tokens")
		}
	}
	h.dashboardToken = token
	return nil
}

func sameDashboardOrigin(origin, host string) bool {
	parsed, err := url.Parse(origin)
	if err != nil || parsed.Scheme != "https" || parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return false
	}
	expected := url.URL{Scheme: "https", Host: host}
	port := func(u *url.URL) string {
		if u.Port() == "" {
			return "443"
		}
		return u.Port()
	}
	return strings.EqualFold(parsed.Hostname(), expected.Hostname()) && port(parsed) == port(&expected)
}

func (h *hub) dashboardAuthenticated(r *http.Request) bool {
	name, password, ok := r.BasicAuth()
	expected := sha256.Sum256([]byte(h.dashboardToken))
	actual := sha256.Sum256([]byte(password))
	return h.dashboardToken != "" && ok && name == "chair" && subtle.ConstantTimeCompare(expected[:], actual[:]) == 1
}

func (h *hub) serveDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Vary", "Authorization")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
	if r.TLS == nil {
		http.Error(w, "TLS required", http.StatusForbidden)
		return
	}
	join := r.URL.Path == "/join"
	if (join || r.URL.Path == "/dashboard/enrollment.js") && h.enrollment == nil {
		http.NotFound(w, r)
		return
	}
	joinAsset := h.enrollment != nil && (r.URL.Path == "/dashboard/style.css" || r.URL.Path == "/dashboard/enrollment.js")
	if h.dashboardToken == "" && !h.dashboardPublic && !join && !joinAsset {
		http.NotFound(w, r)
		return
	}
	reveal := r.URL.Path == "/dashboard/client-token"
	access := r.URL.Path == "/dashboard/access"
	if (reveal || access) && !h.allowClientTokenReveal {
		http.NotFound(w, r)
		return
	}
	method := http.MethodGet
	if reveal || (join && r.Method == http.MethodPost) {
		method = http.MethodPost
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		if join {
			w.Header().Set("Allow", "GET, POST")
		}
		http.Error(w, "unsupported dashboard method", http.StatusMethodNotAllowed)
		return
	}
	paused := (r.URL.Path == "/" || access) && r.URL.RawQuery == "refresh=off"
	action := reveal || (join && r.Method == http.MethodPost)
	publicNavigation := (h.dashboardPublic || join) && !action && !access && r.Header.Get("Sec-Fetch-Mode") == "navigate" && r.Header.Get("Sec-Fetch-Dest") == "document"
	crossSite := r.Header.Get("Sec-Fetch-Site") == "cross-site" && !publicNavigation
	if (r.URL.RawQuery != "" && !paused) || (r.Header.Get("Origin") != "" && !sameDashboardOrigin(r.Header.Get("Origin"), r.Host)) || crossSite {
		http.Error(w, "cross-origin requests and unsupported query strings are not accepted", http.StatusForbidden)
		return
	}
	marker := "X-Chair-Token-Reveal"
	if join {
		marker = "X-Chair-Enroll"
	}
	if action && (r.Header.Get("Origin") == "" || r.Header.Get(marker) != "1") {
		http.Error(w, "explicit same-origin action required", http.StatusForbidden)
		return
	}
	authenticated := h.dashboardAuthenticated(r)
	if ((!h.dashboardPublic && !join && !joinAsset) || reveal || access) && !authenticated {
		w.Header().Set("WWW-Authenticate", `Basic realm="CHAIR dashboard", charset="UTF-8"`)
		http.Error(w, "dashboard authentication required", http.StatusUnauthorized)
		return
	}
	switch r.URL.Path {
	case "/join":
		if r.Method == http.MethodPost {
			h.serveEnrollment(w, r)
			return
		}
		var page bytes.Buffer
		if err := joinTemplate.Execute(&page, nil); err != nil {
			http.Error(w, "join page unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(page.Bytes())
	case "/dashboard/enrollment.js":
		script, err := dashboardAssets.ReadFile("web/enrollment.js")
		if err != nil {
			http.Error(w, "enrollment script unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		_, _ = w.Write(script)
	case "/dashboard/client-token":
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			ClientToken string `json:"client_token"`
		}{ClientToken: h.clientToken})
	case "/dashboard/readme":
		guide, err := dashboardAssets.ReadFile("README.md")
		if err != nil {
			http.Error(w, "setup guide unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Header().Set("Content-Disposition", `inline; filename="README.md"`)
		_, _ = w.Write(guide)
	case "/dashboard/dashboard.js":
		script, err := dashboardAssets.ReadFile("web/dashboard.js")
		if err != nil {
			http.Error(w, "script unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		_, _ = w.Write(script)
	case "/dashboard/style.css":
		css, err := dashboardAssets.ReadFile("web/dashboard.css")
		if err != nil {
			http.Error(w, "stylesheet unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
		_, _ = w.Write(css)
	case "/dashboard/status":
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(h.dashboardSnapshot())
	case "/", "/dashboard/access":
		var page bytes.Buffer
		view := h.dashboardSnapshot()
		view.Live = !paused
		view.PagePath = r.URL.Path
		view.Public = h.dashboardPublic
		view.RevealEnabled = h.allowClientTokenReveal
		view.Authenticated = authenticated
		view.Downloads = h.dashboardDownloads()
		view.DownloadsEnabled = h.downloadsRoot != nil
		view.EnrollmentEnabled = h.enrollment != nil
		if err := dashboardTemplate.Execute(&page, view); err != nil {
			http.Error(w, "dashboard unavailable", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(page.Bytes())
	default:
		if strings.HasPrefix(r.URL.Path, "/dashboard/downloads/") {
			h.serveDownload(w, r)
			return
		}
		http.NotFound(w, r)
	}
}

func (h *hub) dashboardSnapshot() dashboardView {
	now := time.Now().UTC()
	h.mu.Lock()
	defer h.mu.Unlock()
	view := dashboardView{GeneratedAt: now, Authorized: len(h.workerTokens), Limit: cap(h.requests), Workers: make([]dashboardWorker, 0, len(h.workerTokens)), Jobs: make([]dashboardJob, 0, len(h.jobs))}
	models := make(map[string]bool)
	for id := range h.workerTokens {
		row := dashboardWorker{ID: id, Status: "Disconnected", StatusClass: "muted", Models: []string{}, InventoryAge: "Never", Hardware: "Hardware unavailable"}
		if worker := h.workers[id]; !h.closed && worker != nil && !worker.session.IsClosed() {
			view.Connected++
			row.Inflight = worker.inflight
			row.Hardware = worker.hardware.summary()
			switch {
			case worker.inventoryFailed:
				row.Status, row.StatusClass = "Inventory unavailable", "warning"
			case worker.updatedAt.IsZero():
				row.Status, row.StatusClass = "Awaiting inventory", "warning"
			case !worker.ready(now):
				row.Status, row.StatusClass = "Inventory stale", "warning"
			default:
				row.Status, row.StatusClass = "Ready", "ready"
				view.Ready++
				for model := range worker.models {
					row.Models = append(row.Models, model)
					models[model] = true
				}
				sort.Strings(row.Models)
			}
			if !worker.updatedAt.IsZero() {
				row.InventoryAge = now.Sub(worker.updatedAt).Round(time.Second).String() + " ago"
			}
		}
		view.Workers = append(view.Workers, row)
	}
	view.Models = len(models)
	for _, job := range h.jobs {
		view.Jobs = append(view.Jobs, dashboardJob{activeJob: job, Elapsed: now.Sub(job.StartedAt).Round(time.Second).String()})
	}
	sort.Slice(view.Workers, func(i, j int) bool { return view.Workers[i].ID < view.Workers[j].ID })
	sort.Slice(view.Jobs, func(i, j int) bool { return view.Jobs[i].ID < view.Jobs[j].ID })
	return view
}
