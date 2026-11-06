<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Future option: client-to-worker encryption

**Design note only — not implemented, scheduled, or a security guarantee.**

## Current trust boundary

CHAIR terminates client HTTPS and the worker's encrypted tunnel at the hub.
The hub can read prompts, responses, and other inference content in transit.
The selected worker and its engine also see plaintext. CHAIR's no-content-logging
and no-content-persistence policies are application policies, not cryptographic
protection against the hub operator, a compromised hub, or memory inspection.
TLS to the hub is not client-to-engine encryption hidden from the hub.

## Possible goal

Offer a separate mode where the hub schedules and relays ciphertext, but cannot
decrypt inference content. A client-side adapter could preserve a local
OpenAI-compatible interface for existing applications. The standard compatible
CHAIR API would remain a distinct, explicitly trusted-hub mode.

## Questions and requirements for a future design

- Use an established, independently reviewed secure transport/protocol and
  mature cryptographic libraries. Do not invent encryption or framing here.
- Clients must verify worker identity/public keys independently of an untrusted
  hub (for example pinning, trusted certificates, or a verified directory).
  Blindly accepting a key supplied by the hub permits key substitution.
- Bind encrypted sessions to the intended worker, assignment, model, request,
  protocol version, and direction; account for replay, reconnection, rotation,
  revocation, downgrade resistance, and key compromise.
- Define assignment before encryption and authenticated streaming/cancellation
  without buffering whole generations. Do not silently replay onto another
  worker after failure.
- Document remaining metadata exposure: model/routing information, worker IDs,
  timing, traffic size, and availability. Encryption would not hide all traffic
  analysis or protect against a malicious selected worker/engine.
- A test page/adapter delivered by the hub cannot protect against a malicious
  hub changing its JavaScript. A trusted, separately installed client would be
  required for that threat model.
- Contribution scoreboards would need worker-reported/signed usage summaries
  with an explicit trust model, rather than observing plaintext usage frames.
  Signatures prove attribution, not truthful generation, useful work, or a fair
  benchmark; anti-gaming/accounting would require separate design.
- Plan compatibility, performance, abuse controls, threat modelling, security
  review, and deployment/key-recovery procedures before implementation.

For now, trust the hub operator and participating workers, keep deployment on
controlled networks, and do not describe the existing system as hub-blind E2EE.
