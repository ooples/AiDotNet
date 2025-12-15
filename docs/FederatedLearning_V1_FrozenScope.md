# Federated Learning v1 Frozen Scope (Exhaustive Snapshot)

This document defines the frozen, version-1 ("v1") scope for AiDotNet Federated Learning. Once approved, this list is treated as the definition of "100% complete" for the v1 federated learning program. Future pull requests should only add **future breakthroughs** or post-v1 expansions (tracked separately).

Notes:
- This scope intentionally ignores the accidental "Video AI models" content that was posted on Issue #398 comments.
- v1 must support both **CKKS** and **BFV** homomorphic encryption schemes.
- Enterprise-tier delivery must support **attestation-gated participation** and **attestation-gated artifact decryption**.

## 0) Core Platform Capabilities (Required)

- Model-agnostic update representation based on `IFullModel` parameters (vector-based), compatible with all optimizers.
- Deterministic seeding controls (run/round/client) and reproducible test harnesses.
- Pluggable strategy registries/factories for: aggregation, selection, async policies, privacy, accounting, secure aggregation, HE, compression, personalization, transports.
- Metrics and telemetry: per-round timings, bytes up/down, compression ratios, dropout and staleness stats, privacy accounting summaries.
- Failure handling: stragglers, dropouts, partial participation thresholds, retry policies, client quarantine/blacklist.
- Deployment modes:
  - In-memory simulation backend (for tests/dev).
  - Real distributed backend based on `AiDotNet.Serving` (HTTP coordinator).

## A) Federated Optimization (Sync + Server Optimizer FL)

### A1) Baselines
- FedAvg
- FedAvgM (server momentum)
- FedSGD (when gradients are available; optional gradient path but not required)
- FedProx

### A2) FedOpt Family
- FedAdagrad
- FedAdam
- FedYogi

### A3) Heterogeneity Correction
- SCAFFOLD
- FedNova
- FedDyn

## B) Robust Aggregation (Byzantine / Outlier Resilience)

### B1) Robust mean/median family
- Coordinate-wise median
- Trimmed mean (configurable trim fraction)
- Winsorized mean

### B2) Geometric robust estimators
- Geometric median / RFA (Robust Federated Aggregation)

### B3) Byzantine-resilient selection-based
- Krum
- Multi-Krum
- Bulyan

### B4) Anomaly filters / safeguards
- Update norm clipping / bounding
- Cosine-similarity gating vs reference direction
- Loss/metric-based update rejection (optional client-reported validation metrics)

## C) Client Selection / Participation / Scheduling

### C1) Sampling strategies
- Uniform random (fractional participation)
- Weighted by data size (sample-count based)
- Stratified sampling (group-aware)
- Cluster-based sampling (client statistics/embeddings)
- Availability-aware sampling (online probability/dropout model)
- Performance-aware sampling (bandit-style or explore/exploit)

### C2) Straggler/dropout policies
- Deadline cutoff (late updates ignored)
- Partial aggregation threshold (aggregate after K of N)
- Retry/backoff policy
- Quarantine/blacklist policy

## D) Asynchronous and Semi-Synchronous FL

### D1) Async aggregation modes
- FedAsync (staleness-aware weighting)
- FedBuff (buffered aggregation)

### D2) Staleness policies
- Polynomial/exponential staleness decay
- Stale update rejection threshold
- Bounded delay window

## E) Privacy: Differential Privacy + Accounting (Issue #398)

### E1) Differential privacy mechanisms
- Central DP (server adds noise after aggregation)
- Local DP (client adds noise before sending)
- Clip-by-L2 norm (global or per-group)

### E2) Privacy accountants
- Basic composition
- RDP accountant (recommended default when DP enabled)
- Budget enforcement policy (hard stop / warn)

### E3) Reporting
- Per-round accounting
- Final (ε, δ) reporting in federated metadata

## F) Secure Aggregation (Issue #398)

### F1) Protocol requirements
- Dropout-tolerant secure aggregation
- Session management and key exchange abstraction

### F2) Supported modes
- SecureAgg only
- DP + SecureAgg
- Compression + SecureAgg ordering defined and test-covered

## G) Homomorphic Encryption + Hybrid Modes (Issue #398)

### G1) HE provider abstraction
- Key generation/loading
- Encrypt update, aggregate ciphertexts, decrypt aggregate
- Capability flags (additive-only vs broader)

### G2) Required schemes
- CKKS (approximate real-number arithmetic)
- BFV (exact integer arithmetic)

### G3) Deployment modes
- HE-only aggregation
- Hybrid: HE for selected parameter groups + SecureAgg/DP for others
- Key release gated by attestation (enterprise)

## H) Communication Efficiency / Compression

### H1) Quantization
- Uniform k-bit quantization
- Stochastic quantization

### H2) Sparsification
- Top-k sparsification
- Random-k sparsification
- Threshold-based sparsification

### H3) Error feedback
- Residual/error-feedback compatible with quantization and sparsification

### H4) Adaptive policies
- Bandwidth-aware adaptive compression policy

## I) Personalization (Issue #398)

### I1) Representation/head split
- FedRep
- FedPer

### I2) Personalized objective variants
- Ditto
- pFedMe

### I3) Clustered personalization
- Clustered FL (cluster-specific models)

### I4) Local adaptation
- Post-aggregation local fine-tune step

## J) Meta-Learning + Federated Learning (Issue #398)

- Per-FedAvg
- FedMAML (or explicit documented limitations if some model types cannot support it)
- Reptile-style federated meta-learning (optional if needed for breadth)

## K) Operational Presets

- Cross-device preset (high dropout, fractional participation, aggressive compression, async-friendly)
- Cross-silo preset (fewer clients, stronger secure aggregation, heavier evaluation, mostly synchronous)

## L) Real Distributed Deployment + Monetization Tiers (Serving + Attestation)

- `AiDotNet.Serving` as federated coordinator (HTTP)
- Tier policies:
  - Option A: server-only inference (free/OSS)
  - Option B: plaintext artifact download (pro tier)
  - Option C: encrypted artifact + key release gated by TEE attestation (enterprise tier)
- Attestation gating for:
  - training participation ("join")
  - artifact decryption ("decrypt/load")
- Mobile enterprise Option C: attested edge/gateway runtime (phones do not receive decryptable weights)
- Windows managed endpoints: first-class TEE/attestation support; consumer endpoints supported with stricter policies

