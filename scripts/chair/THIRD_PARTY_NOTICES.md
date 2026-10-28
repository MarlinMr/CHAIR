<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CHAIR third-party software notices

CHAIR is an independent fork of NVIDIA Personal AI Router. First-party code is
provided under the Apache License 2.0; see the repository's `LICENSE`, which is
also included in preview archives. Existing copyright notices are preserved.

The standalone `chair` binary links the following runtime dependencies:

| Component | Version | License | Source |
| --- | --- | --- | --- |
| `github.com/coder/websocket` | v1.8.14 | MIT | https://github.com/coder/websocket/tree/v1.8.14 |
| `github.com/hashicorp/yamux` | v0.1.2 | MPL-2.0 | https://github.com/hashicorp/yamux/tree/v0.1.2 |
| Go runtime and standard library | Recorded in `BUILD_INFO.txt` | BSD-3-Clause | https://go.dev/dl/ |

The CHAIR workflow includes each dependency's original, unmodified license text
in every archive:

- `licenses/coder-websocket-LICENSE.txt`
- `licenses/hashicorp-yamux-LICENSE.txt`
- `licenses/go-LICENSE.txt`

These components are not modified or vendored by CHAIR. Their source is available
at the version-specific links above; the yamux source remains available under the
Mozilla Public License 2.0. Go checksums are recorded in `go.sum`.

Test-only dependencies are `github.com/stretchr/testify` v1.12.1 (MIT) and
`go.yaml.in/yaml/v3` v3.0.5 (MIT and Apache-2.0); they are not linked into the
distributed `chair` executable. The inherited desktop application's separately
generated notices are in the repository-root `THIRD_PARTY_NOTICES.md` and do not
describe these standalone CHAIR archives.
