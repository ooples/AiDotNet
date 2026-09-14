# US-25: program and trained-model deployment lifecycle

Companion to [Evolution issue #43](https://github.com/ooples/AiDotNet.Evolution/issues/43) and [Tensors PR #1030](https://github.com/ooples/AiDotNet.Tensors/pull/1030). Builds on the US-24 consumer branch and shared foundation #2148. This is an explicit application-owned lifecycle, not automatic execution of an archive winner.

## Acceptance stories

- As an application owner, given an exact program or trained model, when I stage it, then immutable, non-truncated payload bytes and their applicability envelope are retained without activation. Model restoration requires an explicit matching factory and serialization version; hyperparameters alone are not trained weights.
- As an operator, given an explicitly authorized policy and known-valid incumbent, when I request promotion, then fresh correctness-first paired holdout measurements must satisfy the predeclared direction, mean gain, exact one-sided sign-test threshold and P95 latency gate before compare-and-swap activation. Ties, invalid checks, stale comparisons and inconclusive outcomes cannot activate.
- As a runtime owner, given changed runtime, device, compiler, dataset, workload or validation-protocol hashes, when I select at a batch boundary, then I receive an exact-envelope fallback and one coalesced retuning request. The application explicitly drains it through an idle gate; private searches cannot activate directly.
- As an operator, given consecutive raw regression windows attributed to the exact dispatch revision, when the configured threshold is reached, then retained evidence precedes persistent quarantine and restoration of the prior exact artifact. An unavailable prior uses the application fallback. Old windows cannot quarantine a later deployment.

## Integration

Use `EvolutionDeploymentArtifactRegistry` with an application-private directory. Create `EvolutionDeploymentEnvelope` from six reproducible SHA-256 identities. Construct `EvolutionDeploymentPolicy` with explicit `allowBestEffortPersistence` and fixed bounds, then `EvolutionDeploymentLifecycle` with a known-valid per-envelope fallback and an independent evaluator.

`EvolutionDeploymentEvaluators.Program` adapts existing correctness and fitness evaluators, preserving both reported costs and requiring fresh measurements. `Model` restores frozen serialized state through a caller-selected allowlist factory and disposes each evaluation instance. Never reflectively activate an artifact-supplied type name.

`EvolutionDeploymentRetuners.Program` runs the real engine with bounded seeds, attempts/proposals, no retry/cache/checkpoint, and an eight-cell length archive. Factories must supply fresh private variation and fitness providers. Resource-accounted configurations are rejected instead of silently bypassing their ledger. `AutoML` runs a private `MapElitesAutoML` search, overrides inflated trial/proposal allowances, and serializes the trained winner before search disposal. Search data must not contain the deployment holdout. The regression test exercises real CPU linear-regression training, independent held-out prediction, promotion and deserialization.

At each dispatch boundary, call `Select(envelope)` and execute its returned artifact. Preserve the selection revision for `ObserveAsync`. Drain requested work with `RetunePendingAsync(retuner, idleGate)`. Explicit candidates use `PromoteAsync`; staging alone is inert.

## Safety and measurement boundaries

- The registry flushes file contents and atomically replaces its small active pointer, but does **not** promise native directory fsync or power-loss durability on any OS. Promotion requires explicit best-effort authorization. It is not equivalent to Tensors' Linux durability gate.
- Use a private local filesystem and cooperating writers. Digests detect accidental corruption, not malicious replacement by an actor controlling the directory. Direct reparse points, duplicate JSON, noncanonical manifests and oversized payloads are rejected; this is not an untrusted filesystem or model-deserialization sandbox.
- Selection refreshes pointer/quarantine at batch boundaries and caches immutable loaded bytes. It does not revoke in-flight execution. Storage faults latch that controller onto the supplied fallback; an unwritable store cannot promise cross-process quarantine. The application must supply a genuinely known-valid fallback, independently of a quarantined deployment.
- Controller-lifetime retune admissions, engine evaluation/proposal limits, timeout, grace and cooldown are finite. Failed/canceled admissions are not refunded. Uncooperative top-level work retains its slot until it settles and cannot publish late. Nested providers own their cancellation, isolation, resource limits and any monetary ledger; a deadline does not kill arbitrary nested training/processes. No background worker or paid provider is started implicitly.
- Per-decision sign-test validity assumes independently sampled, predeclared pairs; shared pair indices support common random streams. Repeated selection on the same holdout invalidates that assumption. These gates are not family-wise competitor-superiority claims, GPU benchmarks or an API-spending authorization. Unknown failed quality is explicitly marked `HasQuality=false`, not compared as a measured zero. Provider exceptions propagate; absent receipts are not invented as zero cost.
- Consecutive fixed monitoring windows are a configurable rollback heuristic, not a statistical change-point guarantee. Keep objective units and timing scope identical to the recorded validation protocol. The CPU AutoML fixture deliberately relaxes its latency gate to test functional integration, not speed superiority.
- Source-pinned integration CI and ordinary published-package CI remain separate gates. A passing source build does not make unpublished dependency APIs available in NuGet.
- Model serialization/restoration retains normal persistence-license enforcement. The repeated synthetic AutoML fixture alone uses the existing async-local internal persistence scope so CI and developer trial quotas are not consumed; it does not test entitlement. A separate denial-propagation regression ensures artifact creation does not swallow a model's licensing error. No production adapter opens that scope.

## Verification

The maintained `AiDotNet.Evolution.Integration.Tests` project links the deployment regression tests and references the actual consumer library. `.github/workflows/evolution-csharp.yml` includes deployment-source changes and tests against a pinned Evolution source revision. Final run receipts will be recorded with this companion PR; authored tests alone are not a passing result.
