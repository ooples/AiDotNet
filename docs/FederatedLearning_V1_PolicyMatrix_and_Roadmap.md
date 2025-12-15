# Federated Learning v1 Policy Matrix and Roadmap

This document captures the **v1 delivery policy matrix** (Free/Pro/Enterprise) and the **implementation roadmap** required to reach 100% completion of the frozen scope defined in:
- `docs/FederatedLearning_V1_FrozenScope.md`

It is intentionally **plan-only** (no code) and is designed to be executable by junior developers through clearly defined epics, user stories, and sprint milestones.

## Guiding Principles

- **Facade-first UX**: Users only build/train via `PredictionModelBuilder` and only run inference via `PredictionModelResult`. All orchestration is internal.
- **Extensible by default**: Everything is strategy-driven (interfaces + base classes + factories) with industry-standard defaults when options are not set.
- **Untrusted client assumption**: Federated training clients are potentially untrusted.
- **Enterprise confidentiality**: Enterprise tier requires attestation gating for both training participation and artifact decryption (Option C).
- **Mobile enterprise path**: Mobile enterprise Option C uses an **attested edge/gateway runtime** (phones do not receive decryptable weights).
- **Compatibility**: Core library targets `net471` and `net8.0`. Serving/coordinator targets `net8.0`.

## Definitions

### Delivery Modes
- **Option A (ServerOnly)**: No client artifact download. Inference is served by `AiDotNet.Serving`.
- **Option B (PlaintextArtifact)**: Clients may receive a plaintext serialized `PredictionModelResult`.
- **Option C (EncryptedArtifact + TEE)**: Clients may receive an encrypted artifact; keys are released only to an attested runtime.

### Trust Levels (capability-based)
- **T1 Integrity Attestation**: Attests device/app identity and basic integrity posture.
- **T2 Confidential Compute Attestation**: Attests a TEE-backed runtime; supports key release only to verified workloads.

## Policy Matrix (Tier Defaults)

| Tier | Default Delivery | Join Gate | Decrypt Gate | Default Privacy | Notes |
|---|---|---|---|---|---|
| Free / OSS | Option A | T1 required | N/A | DP + SecureAgg (conservative defaults) | Server-hosted inference; no artifact download/export |
| Pro | Option B | T1 required (T2 recommended) | N/A | SecureAgg default on; DP optional | Plaintext artifact is a deliverable; export is tier-gated |
| Enterprise | Option C | T2 required | T2 required | DP + SecureAgg default on; HE optional/hybrid | Encrypted artifacts; keys released only to attested runtime |

## Platform Matrix (Tier × Platform)

Legend:
- ✅ allowed
- ⚠ allowed with restrictions/policy limits
- ❌ not allowed

| Platform / Participant | Free (A) | Pro (B) | Enterprise (C) | Notes |
|---|---:|---:|---:|---|
| Windows managed endpoint | ✅ (T1) | ✅ (T1/T2) | ✅ (T2) | First-class enterprise target |
| Windows consumer endpoint | ✅ (T1) | ⚠ (restricted) | ⚠/✅ (T2 if available; else ❌) | Consumer posture is weaker; defaults should be restrictive |
| Mobile device (phone/tablet) | ✅ (T1) | ⚠ (typically no plaintext download by default) | ✅ via attested gateway runtime (T2) | Phone participates via integrity gating; gateway holds keys for Option C |
| Enterprise gateway / edge runtime | ✅ | ✅ | ✅ (T2) | Primary mobile enterprise Option C execution boundary |
| VM/server TEEs | (later) | (later) | ✅ (T2) | Added after device-first path is stable |

## Enforcement Points (Where the policy is enforced)

### Server-side enforcement (primary)
`AiDotNet.Serving` enforces:
- Authentication/authorization and entitlement checks
- Client admission control (attestation verification)
- Key release (short-lived keys bound to attested identity)
- Endpoint gating (download/export/admin endpoints)
- Audit logging and rate limiting

### Client-side enforcement (defense-in-depth)
The client runtime/library enforces:
- Artifact signature verification
- License/entitlement checks (where applicable)
- Refuses to load encrypted artifacts unless the attestation proof is valid and current

## Epics and User Stories

### Epic 0: Program Management Artifacts
- **US0.1** Publish and approve frozen scope: `docs/FederatedLearning_V1_FrozenScope.md`
- **US0.2** Maintain post-v1 backlog: `docs/FederatedLearning_OutOfScope_Backlog.md`
- **US0.3** Maintain this roadmap doc and keep it in sync with scope

### Epic 1: Architecture Compliance Foundation
- **US1.1** Rebase PR on `origin/master` and confirm FL-only diff
- **US1.2** Remove generic constraints from all FL types (no `where T : ...`)
- **US1.3** Replace conversion-based math with `INumericOperations<T>` usage
- **US1.4** Add base classes between interfaces and implementations for clean separation
- **US1.5** Standardize on a canonical parameter-vector update representation compatible with any `IFullModel` and any optimizer

### Epic 2: Issue #398 Core Algorithms + Facade Integration
- **US2.1** Builder-driven federated training mode (options-driven, minimal required inputs)
- **US2.2** Persist federated metadata into `PredictionModelResult` and ensure it survives serialization
- **US2.3** FedAvg implemented as a selectable aggregation strategy
- **US2.4** FedProx implemented (heterogeneity support)
- **US2.5** FedBN implemented (BN handling, with parameter grouping rules documented)

### Epic 3: Privacy and Cryptography (Issue #398)
- **US3.1** Differential privacy as a pipeline stage (local DP + central DP)
- **US3.2** Privacy accounting: basic composition + RDP accountant (reporting in metadata)
- **US3.3** Secure aggregation pipeline stage (dropout-tolerant)
- **US3.4** Homomorphic encryption provider abstraction
- **US3.5** Implement CKKS scheme support
- **US3.6** Implement BFV scheme support
- **US3.7** Hybrid modes (HE for selected parameter groups + DP/SecureAgg elsewhere)
- **US3.8** `net471` compatibility validation for HE dependencies and runtime behavior

### Epic 4: Robustness, Sampling, Async, and Communication Efficiency
- **US4.1** Robust aggregation suite: trimmed mean, median, winsorized mean, RFA, Krum/Multi-Krum, Bulyan, filters
- **US4.2** Client selection suite: uniform/weighted/stratified/clustered/availability-aware/performance-aware
- **US4.3** Async FL suite: FedAsync + FedBuff with staleness controls and deterministic simulation tests
- **US4.4** FedOpt suite: FedAdagrad/FedAdam/FedYogi and FedAvgM
- **US4.5** Heterogeneity correction: SCAFFOLD + FedNova + FedDyn
- **US4.6** Compression suite: quantization, sparsification, error feedback, adaptive compression

### Epic 5: Personalization + Meta-learning (Issue #398)
- **US5.1** Personalization framework (strategy + parameter grouping)
- **US5.2** PFL implementations: FedRep, FedPer, Ditto, pFedMe, clustered personalization
- **US5.3** Local adaptation step configurable per client
- **US5.4** Meta-learning integration: Per-FedAvg + FedMAML (or documented limitations) + optional Reptile-style

### Epic 6: LEAF + Metrics (Issue #398 success criteria)
- **US6.1** LEAF dataset loader(s) with offline tiny fixtures for CI
- **US6.2** Communication efficiency metrics (bytes, compression ratios, timings, dropout, staleness)
- **US6.3** DP guarantee reporting (ε, δ) and budget enforcement policy

### Epic 7: Real Distributed Deployment (Serving) + Tier Enforcement
- **US7.1** `AiDotNet.Serving` as federated coordinator (runs, admission, aggregation)
- **US7.2** AuthN/AuthZ + tier entitlements + audit logging
- **US7.3** Option A: server-only inference + no artifact download endpoints
- **US7.4** Option B: plaintext artifact download endpoints (tier/policy gated)
- **US7.5** Option C: encrypted artifact distribution + key release endpoints (attestation gated)

### Epic 8: Attestation and TEEs (Enterprise)
- **US8.1** Attestation provider abstractions + deterministic mock provider for CI
- **US8.2** Key release service: short-lived keys bound to verified identity
- **US8.3** Confidential gateway runtime for mobile enterprise Option C (TEE boundary)
- **US8.4** Windows managed endpoint attestation provider (T2) and policy enforcement
- **US8.5** Windows consumer endpoint path (T1 supported; Option C only when T2 available)
- **US8.6** Mobile integrity attestation provider (T1 admission control)
- **US8.7** VM/server TEE providers (added after device-first path stabilizes)

## Sprint Plan (High-Level)

This plan assumes 2-week sprints and requires that each sprint delivers tests and documentation updates.

- **Sprint 0**: Rebase + publish frozen scope + policy matrix/roadmap artifacts
- **Sprint 1**: Architecture compliance foundation (no constraints, numeric ops, base classes, canonical updates)
- **Sprint 2**: Core algorithms + builder/result integration (FedAvg/FedProx/FedBN end-to-end)
- **Sprint 3**: DP + accounting + secure aggregation
- **Sprint 4**: Robust aggregation + client selection + communication metrics
- **Sprint 5**: FedOpt + heterogeneity correction + async FL
- **Sprint 6**: Compression + personalization framework and initial methods
- **Sprint 7**: Meta-learning + local adaptation + LEAF loader + offline fixtures
- **Sprint 8**: Serving coordinator + tier enforcement (Option A/B)
- **Sprint 9**: Option C foundation (encrypted artifacts + key release + mock attestation)
- **Sprint 10**: Windows managed T2 provider + gateway runtime
- **Sprint 11**: Mobile T1 admission + mobile enterprise via gateway + Windows consumer policy path
- **Sprint 12**: VM/server TEEs + OS expansion baseline

