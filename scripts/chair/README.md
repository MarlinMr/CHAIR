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
- Ollama-native `POST /api/generate`, `POST /api/chat`, and `GET /api/tags`.
- OpenAI-compatible `POST /v1/chat/completions` and `GET /v1/models`, including
  chat streaming and transparent tool-call forwarding to Ollama's own API.
- Optional read-only HTTPS dashboard showing worker connections, fresh model
  inventories, and active hub-routed jobs, protected by a separate credential.
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
per-client quotas, other OpenAI endpoints (including Responses/embeddings),
Anthropic compatibility, remote engine administration,
desktop integration, service installation, or hub high availability. A stalled
client can occupy a concurrency slot until it disconnects; do not expose this
experiment to untrusted clients. Inventory can change between refresh and
inference; a resulting engine error is returned without replay. Existing PAIR
engine-management and scheduler integration remains separate work: this prototype
uses active-request counts, not PAIR's GPU-pressure scheduler.

## Run

### Download a release or CI preview

For versioned downloads, visit [CHAIR releases](https://github.com/MarlinMr/CHAIR/releases).
Releases are experimental and unsigned. CI previews below include newer branch
changes that may not yet be in a release; published release archives are not
modified when the branch changes.

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

### Generate a TLS certificate for testing

For a controlled test, generate a self-signed certificate on the hub using
OpenSSL 1.1.1 or newer. The following commands use a POSIX shell (Linux/macOS).
Replace the example hostname and IP with the addresses you will actually use:

```bash
umask 077
openssl req -x509 -newkey rsa:3072 -nodes \
  -keyout hub.key \
  -out hub.crt \
  -days 30 \
  -subj "/CN=hub.example.internal" \
  -addext "subjectAltName=DNS:hub.example.internal,IP:192.168.1.100" \
  -addext "basicConstraints=critical,CA:FALSE"
```

Use `--tls-cert hub.crt --tls-key hub.key` when starting the hub, as shown below.
The certificate is valid for 30 days. Every hostname or IP used in a connection
URL must appear in `subjectAltName`; the common name alone is not sufficient.
A DNS entry does not create name resolution: configure DNS/hosts or connect using
the listed IP.

**Keep `hub.key` private on the hub.** It is unencrypted, so protect it with
owner-only permissions. Copy only the public `hub.crt` to each worker and to
clients that need to trust it. For example, after provisioning worker credentials
and starting the hub on port 443:

```bash
./chair worker \
  --hub https://192.168.1.100:443 \
  --ca-file hub.crt \
  --worker-id gpu-a --worker-token-file gpu-a.token \
  --engine http://127.0.0.1:11434
```

Test the authenticated model-list endpoint from a client:

```bash
curl --cacert hub.crt \
  --header "Authorization: Bearer $(cat client.token)" \
  https://192.168.1.100/v1/models
```

Use `:8443` in both URLs instead if the hub uses the port in the example below.
The dashboard's browser will warn that this self-signed certificate is not
publicly trusted. For this controlled test only, verify that the certificate is
the one you generated and use the browser's certificate exception if available.
For regular use, obtain a certificate from a trusted issuer or configure a private
CA trusted by your browsers and clients. Do not disable TLS verification in
workers or API clients, and never commit certificates' private keys or tokens.

### Provision credentials and start the hub

Provision a TLS certificate/key for the hub's DNS name (or use the test certificate
above). Clients and workers must
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

### Web dashboard on port 443

The dashboard is disabled by default. Generate a **separate** dashboard token;
it must not match the client token or any worker token:

```bash
umask 077
openssl rand -hex 32 > dashboard.token
```

Add `--dashboard-token-file dashboard.token` to the hub command. To serve both
the dashboard and API on standard HTTPS port 443:

```bash
./chair hub \
  --listen 0.0.0.0:443 \
  --tls-cert hub.crt --tls-key hub.key \
  --client-token-file client.token --workers-file workers.json \
  --dashboard-token-file dashboard.token \
  --max-concurrent-requests 128
```

Open **`https://hub.example.internal/`** and use the browser's login prompt:

- **Username:** `chair`
- **Password:** the contents of `dashboard.token`.

Use a certificate trusted by the browser for that hostname. Port 443 may require
an OS-specific low-port binding permission; prefer granting only that permission
to the service rather than running the entire hub as root. You can keep port
8443 instead and visit `https://hub.example.internal:8443/`. `--listen` controls
the single shared listener: update workers' `--hub` and inference-client base
URLs when changing its port. TLS must terminate at CHAIR; plain HTTP behind a
TLS-terminating reverse proxy is not accepted.

The page shows every authorized worker, including disconnected machines, fresh
model inventories, and active requests with their worker, model, API route,
start time, and elapsed duration. It updates data in place every three seconds,
without reloading the page or closing expanded model/hardware lists. **Pause live
refresh** stops polling; **Refresh overview** fetches a snapshot even while paused.
Failed updates leave the last snapshot visible with an update-failed message.
Without JavaScript, the initial snapshot and manual page refresh remain available,
but there is no automatic refresh.
Models from stale or failed inventories are not shown as available. A healthy
empty inventory can be Ready while advertising no models.

Expand **Host details** for worker-reported OS/architecture, logical CPU count,
CPU model, total RAM, and NVIDIA GPU names/total VRAM where detectable. CPU/RAM
detection uses `/proc` on Linux, `sysctl` on macOS, and PowerShell CIM on Windows.
NVIDIA GPU detection requires `nvidia-smi` on the worker's PATH. AMD/Intel/Apple
GPU details are not detected yet. A GPU model is still shown when its dedicated
VRAM is unavailable (for example on unified-memory systems). Missing tools,
permissions, or unsupported
hardware produce unavailable values, not a claim that the machine has no GPU.
Probes run once at worker startup, with bounded command output and a two-second
deadline per command; restart the worker to re-detect hardware.

These are host facts, **not** live utilization, free-memory values, or proof that
Ollama is using any listed device. Containers may report host RAM/CPU rather than
their resource limits; GPU visibility may differ between the worker and Ollama.
The worker sends only the selected summary fields, never hostnames, serial
numbers, device UUIDs, or raw command output. Metadata is bounded and validated
by the hub, and is not used for scheduling. Update the hub for the new UI and
workers for hardware reporting. The optional metadata does not change transport
v2; a worker without a report shows unavailable details.

Jobs are only requests routed through this hub, not engine-local work. They
disappear on completion, failure, or cancellation; there is no job history.
Prompts, tool arguments/results, generated text, and credentials are never
included in job/status snapshots. Individual API-client identities are not tracked: this view lists
GPU workers, not users sharing the client token.

`GET /dashboard/status` provides the same read-only snapshot as JSON. By default,
HTML, CSS, JavaScript, JSON, downloads, and the local setup guide all require the
dashboard credential. The credential cannot directly run inference, administer
engines, or modify cluster membership. Enabling token reveal below explicitly
allows its holders to obtain inference access. A small embedded same-origin script performs status updates;
there are no external assets, frontend build step, or cookie sessions.
Browsers may retain Basic credentials until the browser session is closed; use
a private browser session on shared machines. Token changes require a hub
restart. Keep this experimental dashboard on trusted networks; worker/model
names are operational metadata and should not be exposed publicly.

#### Host downloads and the README on the hub

The **Setup guide** link opens this executable's bundled README at
`/dashboard/readme` as plain text. No GitHub connection or separate documentation
server is needed. The bundled guide matches the hub build; archives you supply
may be from a different build.

To offer binary downloads, create a dedicated directory and copy the release
archives and their matching `SHA256SUMS` into it yourself. For example:

```bash
mkdir -p /srv/chair/downloads
# Copy your downloaded release archives and SHA256SUMS into this directory.
```

Add `--downloads-dir /srv/chair/downloads` to the hub command. The directory must
already exist and be readable by the hub. Use an operator-controlled directory;
do not grant write access to dashboard visitors or place deployment secrets there.
No uploads, automatic GitHub downloads, or release publishing are performed.

Only these exact filenames are offered, when present as regular files:

```text
chair-linux-amd64.tar.gz
chair-linux-arm64.tar.gz
chair-darwin-amd64.tar.gz
chair-darwin-arm64.tar.gz
chair-windows-amd64.zip
chair-windows-arm64.zip
SHA256SUMS
```

Links use `/dashboard/downloads/<filename>` on this hub. Arbitrary paths,
directory indexes, and symlinks are not served. Changes appear on the next page
load without restarting the hub. Publish complete files, preferably by atomic
rename, and keep checksums consistent with the archives. The hub does not verify
or sign supplied archives; clients should verify them against `SHA256SUMS`.

#### Optional public dashboard

Add **`--dashboard-public`** to permit the page, status snapshots, setup guide,
and configured downloads without login. A dashboard token file is optional in
this mode unless token reveal is also enabled. For example:

```bash
./chair hub \
  --listen 0.0.0.0:443 \
  --tls-cert hub.crt --tls-key hub.key \
  --client-token-file client.token --workers-file workers.json \
  --dashboard-public --downloads-dir /srv/chair/downloads
```

**This exposes worker IDs, hardware, model inventories, and active-job metadata
to anyone who can reach the hub.** Limit network access accordingly. HTTPS is
still required, and inference and worker connections still require their own
credentials. Public mode never enables anonymous token reveal.

#### Optional client-token reveal

Client-token reveal is **disabled by default**. To allow trusted dashboard users
to obtain the inference credential, configure both:

```text
--dashboard-token-file dashboard.token --allow-client-token-reveal
```

The dashboard then offers **Reveal client token**, **Copy token**, and **Hide
token** controls. The token is fetched only after an explicit, authenticated,
same-origin POST to `/dashboard/client-token`; it is not included in initial
HTML, normal status snapshots, URLs, or logs. Responses disable caching. The
display clears after one minute, when hidden manually, or when you leave the tab.
JavaScript is required for these controls.

On a public dashboard, use **Sign in for token access** (`/dashboard/access`)
and the `chair` username/dashboard password before revealing. Anonymous visitors
cannot retrieve the token even when the reveal option is enabled.

**Anyone holding the dashboard credential can obtain inference access when this
option is on.** Visual hiding is only a privacy aid: it does not revoke a revealed
token, erase browser developer-tool records, or clear a copied clipboard. Protect
dashboard credentials and use a private browser session on shared machines.
Turning the option off requires a hub restart; rotate the client token and
restart the hub if a revealed credential needs to be invalidated.

### Logging

Both commands accept `--log-level debug|info|warn|error` (default `info`). Logs are
timestamped key/value records written to **stderr**. For example, append
`--log-level debug` to either the hub or worker command. Use `./chair hub --help`
or `./chair worker --help` to list all options.

- **info:** hub startup/configured worker count, authenticated worker connections
  and disconnections, first/changed model-inventory counts, inventory recovery,
  and worker-side hub connection events.
- **warn:** rejected authentication/protocol/duplicate identities, inventory
  failures, missing model owners, concurrency-limit rejections, forwarding
  failures/interrupted streams, and reconnect attempts. Repeated inventory
  failures warn once per episode; subsequent failures are debug until recovery.
- **debug:** unchanged inventory refreshes, repeated inventory failures, and each
  inference's hub-local request ID, model, selected worker, method/path, status,
  duration, response byte count, and completion/cancellation/interruption outcome.
- **error:** fatal command/startup failures. Invalid log-level values are rejected.

Diagnostics use fixed reason codes (for example `tls_verification_failed`,
`engine_unreachable`, `invalid_inventory`, or `inventory_timeout`) and HTTP status
codes, not raw library errors. Prompts, messages, request/response bodies, stream
chunks, headers, cookies, tokens, and key/certificate material are never logged.
Low-level HTTP/proxy/multiplexer logs remain suppressed for that reason; debug
mode enables CHAIR's sanitized metadata, not those raw logs. Model names and
worker IDs are operational metadata and appear in logs, so review logs before
sharing them publicly.

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
supported for inference; the hub rejects inference requests with an `Origin`
header. The separately authenticated dashboard supports same-origin browsers.

## OpenAI-compatible clients and OpenCode

Applications connect to the **hub**, not workers. Configure:

- **Base URL:** `https://hub.example.internal:8443/v1`
- **API key:** the hub's **client** token, not a worker token or an OpenAI key.
- **Model:** an installed ID from authenticated `GET /v1/models`.

The hub returns a sorted, deduplicated OpenAI model list:
`{"object":"list","data":[{"id":"model:latest","object":"model","created":0,"owned_by":"ollama"}]}`.
IDs normalize implicit `:latest` tags; `created` is zero because the aggregate
inventory does not have an authoritative creation timestamp. Empty healthy
inventories return an empty `data` array; unavailable inventories return `503`.

```bash
curl --no-buffer --cacert private-ca.crt \
  --header "Authorization: Bearer $(cat client.token)" \
  --header 'Content-Type: application/json' \
  --data '{"model":"your-installed-model:latest","messages":[{"role":"user","content":"Hello"}],"stream":true}' \
  https://hub.example.internal:8443/v1/chat/completions
```

CHAIR forwards the complete JSON body and the selected worker's response without
translation: messages, tool definitions/results, tool-call argument fragments,
usage fields, finish reasons, and SSE `[DONE]` markers remain intact. The engine
and model must support the requested features. CHAIR does not execute client
tools or emulate unsupported Ollama parameters. Request/response contents remain
excluded from logs. Streaming, cancellation, concurrency limits, and model-owner
selection work exactly as for the native Ollama routes.

Hub-generated errors on `/v1/` use an OpenAI-shaped JSON `error` object; engine
HTTP errors are passed through with their status/body unchanged. There is no
automatic job replay, although a client SDK may independently retry failed calls.
This is the **Chat Completions subset**, not the entire OpenAI API: Responses,
embeddings, individual-model retrieval, audio, files, and assistants are not exposed.
Only the documented methods/paths with no query string are accepted.

### OpenCode V2 configuration

Use a custom **OpenAI-compatible Chat** provider, not the OpenAI Responses package
or Ollama's native discovery (which also needs `/api/show`, not exposed by CHAIR).
For example, add this to your own `opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "chair/coder",
  "providers": {
    "chair": {
      "name": "CHAIR",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "https://hub.example.internal:8443/v1",
        "apiKey": "{env:CHAIR_API_KEY}",
        "encoding": "none"
      },
      "models": {
        "coder": {
          "modelID": "your-installed-model:latest",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "compatibility": { "maxTokensField": "max_tokens" },
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

Replace the model ID, capabilities, and context/output limits with those of your
actual model and worker engine settings; the numbers above are examples, not
detected limits. Choose a tool-capable model for coding-agent workflows. Make the
client token available to the OpenCode server as `CHAIR_API_KEY`, and ensure that
server trusts the hub's certificate issuer. Do not disable TLS verification or
commit the token. See the official [OpenCode V2 provider guide](https://opencode.ai/v2/docs/providers)
and [model guide](https://opencode.ai/v2/docs/models) for current configuration.
Protocol forwarding is tested with fake engines; end-to-end OpenCode/real-model
agent behavior still requires testing on your deployment.

**Upgrade both hub and workers to this build.** The worker protocol is now
`chair-transport-v2`; earlier workers without the OpenAI routes are rejected at
connection time rather than silently accepting jobs they cannot serve.

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

## Dashboard browser regression tests

The Go build and normal Go tests need no Node.js installation. To also run the
headless Chromium tests (Node.js 20+ and OpenSSL required), from `scripts/chair/`:

```bash
go build -o ./browser-tests/chair-browser-test .
npm ci --prefix browser-tests
cd browser-tests
npx playwright install chromium
CHAIR_TEST_BINARY="$PWD/chair-browser-test" npm test
```

These are POSIX-shell commands; on Windows, build an `.exe` and set
`CHAIR_TEST_BINARY` to its absolute path using your shell's environment syntax.
On Linux, missing browser system dependencies can be
installed with `npx playwright install --with-deps chromium`. Tests start a local
TLS hub, worker, and fake engine using temporary test-only credentials; no real
engine or GPU inference is used. They check expansion/focus preservation, safe
model rendering, polling pause/resume, and failed-update behavior. Playwright is
a test-only dependency, not part of the executable or dashboard.

## Dependencies and integration boundary

CHAIR uses `github.com/coder/websocket` (MIT) and
`github.com/hashicorp/yamux` (MPL-2.0), with Testify (MIT) for tests. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact versions, source links,
and the license texts included in CI archives. Before integrating CHAIR into the
desktop product, update that product's BOM, packaging, and service ownership.
The existing service inventory, generated desktop contracts, release versions,
and application architecture remain unchanged by this unbundled experiment.
