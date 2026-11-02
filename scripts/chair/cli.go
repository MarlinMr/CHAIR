// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"os"
)

// Update when preparing the next CHAIR release. All build paths use this value.
const version = "v0.1.5"

func commandFlags(command string) *flag.FlagSet {
	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	flags.Usage = func() {
		out := flags.Output()
		fmt.Fprintf(out, "CHAIR %s — Central Hub for AI Routing\n\n", version)
		if command == "" {
			fmt.Fprintln(out, "Usage: chair hub|worker [flags]")
			fmt.Fprintln(out, "\nCommands:\n  hub      Run the central routing hub\n  worker   Connect a local engine to the hub")
			fmt.Fprintln(out, "\nUse chair hub --help or chair worker --help for command flags.")
		} else {
			fmt.Fprintf(out, "Usage: chair %s [flags]\n", command)
		}
		fmt.Fprintln(out, "\nFlags:")
		flags.PrintDefaults()
	}
	return flags
}

func parseCommandFlags(flags *flag.FlagSet, args []string) error {
	showVersion := flags.Bool("version", false, "print CHAIR version and exit")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *showVersion {
		if _, err := fmt.Fprintln(os.Stdout, "CHAIR "+version); err != nil {
			return err
		}
		return flag.ErrHelp
	}
	return nil
}
