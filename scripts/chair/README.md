<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CHAIR — Central Hub for AI Routing

*Finally, a department chair that distributes the workload.*

**Experimental standalone tool, not a production deployment or a replacement for PAIR's
existing runtime.** This independent Go module implements a CPU-only hub routing
inference across multiple GPU workers on routed/firewalled networks.
It is not bundled, supervised by the existing broker, or started by the desktop.
No existing PAIR binary, JSON-RPC contract, discovery behavior, or installer changes.

```text
AI client -- HTTPS + client token --> CPU-only hub
                                         ^
                                         |
                  outbound WSS + per-worker credentials
                          /              |              \
                    GPU worker A   GPU worker B    GPU worker C
                       LAN A          LAN B           LAN C
                          |              |              |
                       local          local          local
                       Ollama         Ollama         Ollama
```

The worker opens the only network connection required between hub and worker.
The hub opens logical HTTP streams inside it, using WebSocket binary messages
and yamux stream multiplexing. It never dials a worker IP, opens a worker TCP
listener, uses mDNS, or communicates with other workers. Each logical stream
carries one HTTP request and its streaming response. Cancellation closes that
stream and cancels the local engine request without closing other streams.

## Current scope

- Multiple active workers, each backed by a separately running local
  Ollama-compatible engine. Provision 1–256 authorized worker IDs on the hub.
- `POST /api/generate`, `POST /api/chat`, and `GET /api/tags` only.
- End-to-end TLS to the hub, with system trust or an additional CA file.
- Operator-provisioned identities and a distinct bearer token for each worker,
  separate from the client token. One live connection per authorized identity;
  another worker cannot use its own token to claim a different identity.
- Configurable concurrent inference requests globally (`--max-concurrent-requests`,
  default **16**), 8 MiB request bodies, 64 KiB response headers,
  and a five-minute wait for engine response headers. Responses stream without
  whole-response buffering; yamux applies per-stream flow control. Request bodies
  are buffered within the limit to read the requested model before dispatch.
- The hub fetches each worker's model inventory through its tunnel on connection
  and every two seconds, with a three-second timeout and 1 MiB response limit.
  Empty inventories serve no inference; invalid or failed reports withdraw all
  capabilities until a successful refresh. Reports older than ten seconds are
  ineligible even if the worker connection is still open.
- Model-aware routing selects the eligible worker with the fewest active hub
  requests. Reservations are made atomically before forwarding, including before
  response headers. Equally busy workers rotate by least-recent assignment,
  then stable ID. There is no GPU-utilization/capacity weighting yet.
- `GET /api/tags` returns a sorted, deduplicated union of fresh inventories,
  retaining model metadata. For duplicate models, the lowest worker ID supplies
  the metadata. Ollama's implicit `:latest` tag is normalized for matching.
- Session keepalives every 15 seconds, with a ten-second write/ping timeout.
- Workers reconnect with exponential backoff capped at 30 seconds. New requests
  can use a reconnected worker once it reports a fresh model inventory.
- A disconnected worker is unavailable. Interrupted requests fail; no request
  is automatically retried or resumed on another worker. Other workers continue
  serving new requests. Requests with no eligible model owner return `502`; no fresh inventory
  for model listing returns `503`, while a valid empty list returns `200`.
- Only literal loopback engine addresses are accepted. No arbitrary remote URL
  forwarding or engine administration is exposed. Client credentials and cookies
  are not sent to the worker's engine. Inference contents are never logged.

**Not implemented:** interactive enrollment, worker certificates/mTLS,
live credential revocation, GPU telemetry, queues, request deadlines,
per-client quotas, OpenAI/Anthropic compatibility, remote engine administration,
desktop integration, service installation, or hub high availability. A stalled
client can occupy a concurrency slot until it disconnects; do not expose this
experiment to untrusted clients. Inventory can change between refresh and
inference; a resulting engine error is returned without replay. Existing PAIR
engine-management and scheduler integration remains separate work: this prototype
uses active-request counts, not PAIR's GPU-pressure scheduler.

## Run

### Download a CI preview

Open the [CHAIR workflow](https://github.com/MarlinMr/CHAIR/actions/workflows/chair.yml),
choose a successful run for your branch/commit, and download the artifact matching
your OS and architecture. GitHub requires you to be signed in to download Actions
artifacts. Extract the artifact ZIP, then extract the enclosed `.tar.gz` or `.zip`.
The executable is `chair` on Linux/macOS or `chair.exe` on Windows; the same binary
provides both the `hub` and `worker` commands.

Artifacts contain a SHA-256 checksum, the setup guide, license notices, and the
unmodified dependency license texts. Check the archive with `sha256sum --check
SHA256SUMS` on Linux or `shasum -a 256 --check SHA256SUMS` on macOS. On Windows,
compare `Get-FileHash .\chair-windows-amd64.zip -Algorithm SHA256` with `SHA256SUMS`
(substitute `arm64` when needed). These are **unsigned experimental previews**,
not signed installers or GitHub Releases. A cross-compiled binary is not evidence
of runtime/GPU testing on that target. Artifacts expire after 14 days.

### Build from source

Requires Go 1.25 or newer. From `scripts/chair/` in the source checkout:

```bash
go test -race ./...
go build -o chair .
```

On Windows, build with `go build -o chair.exe .`. Use `./chair` (or
`.\chair.exe` in PowerShell) for the commands below, from the extracted archive
or your local build directory. CHAIR currently uses server-verified TLS plus
per-worker bearer credentials, not PAIR's PIN pairing or peer certificate mesh.

### Provision credentials and start the hub

Provision a TLS certificate/key for the hub's DNS name. Clients and workers must
trust its issuer and use a name covered by its certificate. For a private CA,
provide the CA certificate to workers with `--ca-file` and clients with their
normal CA configuration. There is deliberately no insecure TLS option.

Generate a client token and **a different random token for every worker** (at
least 32 non-whitespace ASCII characters), store them in owner-only files, and
transfer each worker's token only to that worker. For example, in a private
directory on the hub:

```bash
umask 077
openssl rand -hex 32 > client.token
openssl rand -hex 32 > gpu-a.token
openssl rand -hex 32 > gpu-b.token
```

Tokens are long-lived credentials, not enrollment codes. File permissions and
secure provisioning are the operator's responsibility. Do not commit them.

Create an owner-only `workers.json` on the hub. Replace the placeholders with
the actual generated token contents (not file paths):

```json
{
  "workers": [
    { "id": "gpu-a", "token": "<contents of gpu-a.token>" },
    { "id": "gpu-b", "token": "<contents of gpu-b.token>" }
  ]
}
```

IDs are stable operator-chosen names of 1–64 ASCII letters, digits, dots,
underscores, or hyphens. IDs and tokens must be distinct; no worker token may
equal the client token. Treat `workers.json` as a secret and do not commit it.

Start the hub, explicitly binding the appropriate reachable interface:

```bash
./chair hub \
  --listen 0.0.0.0:8443 \
  --tls-cert hub.crt --tls-key hub.key \
  --client-token-file client.token --workers-file workers.json \
  --max-concurrent-requests 128
```

### Concurrency tuning

`--max-concurrent-requests N` accepts any positive integer and limits simultaneous
inference requests across **all** workers, not per worker. The default of 16 is a
conservative prototype setting, **not** a PAIR, GPU, transport, or measured RAM
limit. Requests above the configured limit receive `503` immediately; they are
not queued. Model listing and inventory refreshes do not consume inference slots.

Choose the value based on your aggregate worker capacity and hub resources.
Request bodies are buffered in the hub's ordinary RAM, not GPU memory. At the
8 MiB per-request maximum, 16 bodies contain 128 MiB of payload, and 128 contain
1 GiB. These figures are payload estimates, not total process-memory bounds:
buffer growth, HTTP/tunnel buffers, inventories, and runtime overhead add memory,
while typical requests are much smaller. Streaming responses are not buffered
in full. Raising the limit requires no code changes; restart the hub with the new
flag. This setting does not enforce per-worker engine concurrency limits.

### Start workers

On each GPU machine, start Ollama separately with a model installed, then run the
worker using the engine's **actual local listening port**, not a PAIR proxy port:

```bash
./chair worker \
  --hub https://hub.example.internal:8443 \
  --worker-id gpu-a --worker-token-file gpu-a.token \
  --ca-file private-ca.crt \
  --engine http://127.0.0.1:11434
```

Omit `--ca-file` when the hub uses an already trusted certificate. The engine
must already be running; this prototype does not start, move, or stop engines.
If PAIR manages Ollama on that machine, its engine port may differ from 11434.
On the second machine, use `--worker-id gpu-b --worker-token-file gpu-b.token`
with its own local engine URL. Repeat with additional authorized IDs as needed.
Install the same model on multiple workers to distribute requests for that model;
workers with different models are used only for their own advertised models.

Send an inference request from an authorized client:

```bash
curl --no-buffer --cacert private-ca.crt \
  --header "Authorization: Bearer $(cat client.token)" \
  --header 'Content-Type: application/json' \
  --data '{"model":"your-installed-model","prompt":"Hello","stream":true}' \
  https://hub.example.internal:8443/api/generate
```

The command shows the header syntax, but shell-expanded tokens can be visible in
process arguments. For sensitive environments, configure the client to read its
authorization header from a protected file instead. Browser clients are not
supported; the hub rejects requests with an `Origin` header.

## Firewall and failure checks

Allow worker-to-hub TCP 8443 (or the configured port), including return traffic,
and client-to-hub access. No inbound worker rule is needed for this transport.
WebSocket upgrades and long-lived TLS connections must be allowed by inspection
devices. The prototype makes direct connections; it does not use `HTTP_PROXY`.
A reverse proxy must pass TLS through or establish TLS to the hub; a forwarded
`https` header cannot substitute for an encrypted hub connection.

To validate on your own networks, block hub-initiated worker connections and all
worker-to-worker traffic, then verify streaming and cancellation. Stop the worker
mid-stream: its response must fail and the engine request must be canceled.
New requests should use surviving workers that advertise the requested model,
or return `502` if none do. Restart the worker and send a **new** request.
Automated tests exercise TLS tunnels to multiple fake local engines, including
simultaneous dispatch, inventory updates, credential isolation, and disconnects;
they do not prove cross-subnet firewall behavior or real GPU execution.

Changing authorized identities or credentials requires restarting the relevant
processes. Stopping the hub disconnects every worker and active request; there is
no job persistence or hub failover. Workers do not need to be reachable from one
another or from new hub-initiated connections.

## Dependencies and integration boundary

CHAIR uses `github.com/coder/websocket` (MIT) and
`github.com/hashicorp/yamux` (MPL-2.0), with Testify (MIT) for tests. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact versions, source links,
and the license texts included in CI archives. Before integrating CHAIR into the
desktop product, update that product's BOM, packaging, and service ownership.
The existing service inventory, generated desktop contracts, release versions,
and application architecture remain unchanged by this unbundled experiment.
