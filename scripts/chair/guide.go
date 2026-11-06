// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"fmt"
	"html/template"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/parser"
	"github.com/yuin/goldmark/text"
)

var guideTemplate = template.Must(template.ParseFS(dashboardAssets, "web/readme.html"))

type guideSection struct {
	ID    string
	Title string
}

type guideView struct {
	Origin   string
	Version  string
	Sections []guideSection
}

// Accept only authorities safe in shell, JSON, Markdown, and HTML examples.
// Forwarded headers are deliberately ignored: TLS terminates at this hub.
func guideOrigin(authority string) (string, error) {
	u, err := url.Parse("https://" + authority)
	if err != nil || u.Host != authority || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || len(authority) > 260 {
		return "", fmt.Errorf("invalid hub address")
	}
	host := u.Hostname()
	if host == "" {
		return "", fmt.Errorf("invalid hub address")
	}
	if net.ParseIP(host) == nil {
		for _, label := range strings.Split(strings.TrimSuffix(host, "."), ".") {
			if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
				return "", fmt.Errorf("invalid hub address")
			}
			for _, c := range label {
				if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-') {
					return "", fmt.Errorf("invalid hub address")
				}
			}
		}
	}
	expected := host
	if strings.Contains(host, ":") {
		expected = "[" + host + "]"
	}
	if port := u.Port(); port != "" {
		n, err := strconv.Atoi(port)
		if err != nil || n < 1 || n > 65535 {
			return "", fmt.Errorf("invalid hub address")
		}
		expected = net.JoinHostPort(host, port)
	}
	if authority != expected {
		return "", fmt.Errorf("invalid hub address")
	}
	return u.String(), nil
}

func renderGuide(source []byte, origin string) ([]byte, error) {
	// Replace only hub URL placeholders, never worker-local engine URLs,
	// credentials, certificate subject names, or arbitrary example hostnames.
	source = bytes.ReplaceAll(source, []byte("https://hub.example.internal:8443"), []byte(origin))
	source = bytes.ReplaceAll(source, []byte("https://hub.example.internal"), []byte(origin))
	markdown := goldmark.New(goldmark.WithExtensions(extension.Table), goldmark.WithParserOptions(parser.WithAutoHeadingID()))
	document := markdown.Parser().Parse(text.NewReader(source))
	view := guideView{Origin: origin, Version: version}
	base, err := url.Parse("https://github.com/MarlinMr/CHAIR/blob/feat/chair/scripts/chair/")
	if err != nil {
		return nil, err
	}
	if err := ast.Walk(document, func(node ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		if heading, ok := node.(*ast.Heading); ok && heading.Level == 2 {
			if id, ok := heading.AttributeString("id"); ok {
				if value, ok := id.([]byte); ok {
					view.Sections = append(view.Sections, guideSection{ID: string(value), Title: string(heading.Text(source))})
				}
			}
		}
		if link, ok := node.(*ast.Link); ok {
			if target, err := url.Parse(string(link.Destination)); err == nil && target.Scheme == "" && target.Host == "" && target.Path != "" && !strings.HasPrefix(target.Path, "/") {
				link.Destination = []byte(base.ResolveReference(target).String())
			}
		}
		return ast.WalkContinue, nil
	}); err != nil {
		return nil, err
	}
	var page bytes.Buffer
	if err := guideTemplate.ExecuteTemplate(&page, "header", view); err != nil {
		return nil, err
	}
	// Goldmark's default renderer omits raw HTML and unsafe link destinations.
	// Render straight into the buffer, without marking strings as trusted HTML.
	if err := markdown.Renderer().Render(&page, source, document); err != nil {
		return nil, err
	}
	if err := guideTemplate.ExecuteTemplate(&page, "footer", nil); err != nil {
		return nil, err
	}
	return page.Bytes(), nil
}

func (h *hub) serveGuide(w http.ResponseWriter, r *http.Request) {
	origin, err := guideOrigin(r.Host)
	if err != nil {
		http.Error(w, "invalid hub address", http.StatusBadRequest)
		return
	}
	source, err := dashboardAssets.ReadFile("README.md")
	if err != nil {
		http.Error(w, "setup guide unavailable", http.StatusInternalServerError)
		return
	}
	page, err := renderGuide(source, origin)
	if err != nil {
		http.Error(w, "setup guide unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(page)
}
