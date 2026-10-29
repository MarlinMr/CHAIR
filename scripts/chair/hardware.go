// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const hardwareHeader = "X-Chair-Worker-Hardware"

type workerGPU struct {
	Name      string `json:"name"`
	MemoryMiB uint64 `json:"memory_mib"`
}

type workerHardware struct {
	OS          string      `json:"os"`
	Arch        string      `json:"arch"`
	CPU         string      `json:"cpu"`
	LogicalCPUs int         `json:"logical_cpus"`
	MemoryBytes uint64      `json:"memory_bytes"`
	GPUs        []workerGPU `json:"gpus"`
}

func hardwareTextValid(value string) bool {
	return len(value) <= 256 && !strings.ContainsFunc(value, unicode.IsControl)
}

func decodeHardware(value string) (workerHardware, error) {
	var info workerHardware
	// Metadata is optional in transport v2 and has no bearing on routing.
	if value == "" {
		return info, nil
	}
	if len(value) > 16<<10 {
		return info, fmt.Errorf("hardware metadata too large")
	}
	data, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil {
		return info, fmt.Errorf("invalid hardware metadata")
	}
	if err := json.Unmarshal(data, &info); err != nil {
		return workerHardware{}, fmt.Errorf("invalid hardware metadata")
	}
	if !hardwareTextValid(info.OS) || !hardwareTextValid(info.Arch) || !hardwareTextValid(info.CPU) || info.LogicalCPUs < 0 || info.LogicalCPUs > 1<<20 || len(info.GPUs) > 32 {
		return workerHardware{}, fmt.Errorf("invalid hardware metadata")
	}
	for _, gpu := range info.GPUs {
		if gpu.Name == "" || !hardwareTextValid(gpu.Name) {
			return workerHardware{}, fmt.Errorf("invalid hardware metadata")
		}
	}
	return info, nil
}

func (info workerHardware) summary() string {
	parts := []string{}
	if info.OS != "" {
		parts = append(parts, info.OS+"/"+info.Arch)
	}
	if info.CPU != "" {
		parts = append(parts, info.CPU)
	}
	if info.LogicalCPUs > 0 {
		parts = append(parts, fmt.Sprintf("%d logical CPUs", info.LogicalCPUs))
	}
	if info.MemoryBytes > 0 {
		parts = append(parts, fmt.Sprintf("%.1f GiB RAM", float64(info.MemoryBytes)/(1<<30)))
	} else {
		parts = append(parts, "RAM unavailable")
	}
	for _, gpu := range info.GPUs {
		if gpu.MemoryMiB > 0 {
			parts = append(parts, fmt.Sprintf("%s (%.1f GiB VRAM)", gpu.Name, float64(gpu.MemoryMiB)/1024))
		} else {
			parts = append(parts, gpu.Name+" (VRAM unavailable)")
		}
	}
	if len(info.GPUs) == 0 {
		parts = append(parts, "GPU details unavailable")
	}
	return strings.Join(parts, " · ")
}

// Commands are fixed, have a deadline and bounded output, and never include
// credentials or engine contents. Detection failures do not prevent serving.
type hardwareOutput struct{ buffer bytes.Buffer }

func (out *hardwareOutput) Write(data []byte) (int, error) {
	if out.buffer.Len()+len(data) > 64<<10 {
		return 0, fmt.Errorf("hardware output too large")
	}
	return out.buffer.Write(data)
}

func hardwareCommand(ctx context.Context, name string, args ...string) []byte {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, name, args...)
	var output hardwareOutput
	command.Stdout = &output
	command.Stderr = io.Discard
	command.WaitDelay = 100 * time.Millisecond
	if command.Run() != nil {
		return nil
	}
	return output.buffer.Bytes()
}

func readHardwareFile(path string) []byte {
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 64<<10))
	if err != nil {
		return nil
	}
	return data
}

func linuxHardware(cpu, memory []byte) (string, uint64) {
	name := ""
	for _, line := range strings.Split(string(cpu), "\n") {
		key, value, found := strings.Cut(line, ":")
		if found && (strings.TrimSpace(key) == "model name" || strings.TrimSpace(key) == "Hardware") {
			name = strings.TrimSpace(value)
			break
		}
	}
	for _, line := range strings.Split(string(memory), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 3 && fields[0] == "MemTotal:" && fields[2] == "kB" {
			kib, err := strconv.ParseUint(fields[1], 10, 54)
			if err == nil {
				return name, kib * 1024
			}
		}
	}
	return name, 0
}

func nvidiaHardware(data []byte) []workerGPU {
	rows, err := csv.NewReader(bytes.NewReader(data)).ReadAll()
	if err != nil || len(rows) > 32 {
		return nil
	}
	gpus := []workerGPU{}
	for _, row := range rows {
		if len(row) != 2 {
			return nil
		}
		name := strings.TrimSpace(row[0])
		memoryText := strings.TrimSpace(row[1])
		memory, err := strconv.ParseUint(memoryText, 10, 64)
		// Unified-memory GPUs can report a model name but no dedicated VRAM.
		if (err != nil && memoryText != "[N/A]" && memoryText != "N/A") || name == "" || !hardwareTextValid(name) {
			return nil
		}
		gpus = append(gpus, workerGPU{Name: name, MemoryMiB: memory})
	}
	return gpus
}

func detectHardware(ctx context.Context) workerHardware {
	info := workerHardware{OS: runtime.GOOS, Arch: runtime.GOARCH, LogicalCPUs: runtime.NumCPU()}
	switch runtime.GOOS {
	case "linux":
		info.CPU, info.MemoryBytes = linuxHardware(readHardwareFile("/proc/cpuinfo"), readHardwareFile("/proc/meminfo"))
	case "darwin":
		info.CPU = strings.TrimSpace(string(hardwareCommand(ctx, "sysctl", "-n", "machdep.cpu.brand_string")))
		info.MemoryBytes, _ = strconv.ParseUint(strings.TrimSpace(string(hardwareCommand(ctx, "sysctl", "-n", "hw.memsize"))), 10, 64)
	case "windows":
		data := hardwareCommand(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1; $ram = Get-CimInstance Win32_ComputerSystem; @{cpu=$cpu.Name; memory_bytes=$ram.TotalPhysicalMemory} | ConvertTo-Json -Compress")
		var system struct {
			CPU         string `json:"cpu"`
			MemoryBytes uint64 `json:"memory_bytes"`
		}
		if json.Unmarshal(data, &system) == nil {
			info.CPU, info.MemoryBytes = system.CPU, system.MemoryBytes
		}
	}
	if !hardwareTextValid(info.CPU) {
		info.CPU = ""
	}
	info.GPUs = nvidiaHardware(hardwareCommand(ctx, "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"))
	return info
}
