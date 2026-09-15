# Persistent program-fitness reuse

`PersistentProgramFitnessEvaluator` is an opt-in `CustomFitnessEvaluator` decorator for US-23.
It caches fitness below the facade's correctness gate. It does not cache correctness, discover
algorithms, prove runtime speedups, or make repeated observations independent.

## Wiring and evidence contract

Provide an `IProgramFitnessEvaluator` that reports genuine fresh `MeasurementOrigin`, an
`IEvolutionEvaluationStore` (for example, `DirectoryEvolutionEvaluationStore`), and an
`IProgramMeasurementEvidenceStore`. The last interface retains and verifies the **raw observations**
against the exact program, original sample identities and measured values. A digest of the summary
alone does not satisfy this contract. The library deliberately supplies no fabricated timing samples.

Construct the twelve-facet `EvolutionReuseScope` for the inner fitness domain: `TaskId = inner.Id`,
`TaskVersion = inner.VersionHash`, `EvaluatorVersion = inner.VersionHash`, and codec facets from
`ProgramGenomeCodec`. Explicitly declare constraints, data/partition, fidelity, compiler/dependencies,
runtime, hardware/device and correctness policy. The outer `ProgramEvolutionTask.VersionHash` is
not the fitness-domain scope; it also fingerprints this decorator and descriptors. Matching labels
are declarations, not hardware or isolation attestation.

Pass that same scope on the producer's fresh origin. `measurementVersion` identifies the requested
replication/aggregation policy; the evidence provider's verification version also partitions keys.
Use a unique `runId` per independent run and stable evaluation/attempt IDs inside that run. Share
one caller-owned ledger declaring `cache_store_invocations` and `program_evidence_invocations`.
Each dispatched store method consumes one logical unit, including failure and cancellation. These
counters are **not** physical I/O, bytes, elapsed time, model tokens or evaluator `cost_units`.
Bound and account for those separately. `ProgramEvolutionResourceOptions` can meter current
correctness-plus-fitness cost on that same ledger without rebilling original measurement cost.
That sharing is **within one engine run**. Independent facade runs restart evaluation IDs and therefore
need distinct run ledgers; share the persistent stores between them. Campaign-wide limits require explicit
per-run budget allocation and aggregated receipts, not resetting a supposedly shared spending cap.

Assign the decorator to `ProgramEvolutionOptions.CustomFitnessEvaluator`, and explicitly set
`EvolutionOptions.EnableEvaluationCache = false` through `ConfigureEvolution`. The facade refuses
enabled run-local memoization and automatic checkpoint/resume for this directly configured decorator;
otherwise outer memoization could bypass freshness and raw-evidence checks, or resume could reset the
ledger. Custom compositions that hide the decorator must enforce these rules themselves. A directly
invoked evaluator does not supply correctness checks; configure those separately at the facade.

The optional UTC clock is caller-owned. Reuse is checked both before and **after** asynchronous raw
evidence verification. Force-fresh skips reads but may publish the new producer evidence. Disabled
mode performs no cache/evidence calls. Programs whose strict UTF8 codec payload exceeds 64 KiB are
evaluated fresh without persistence; this bound does not silently reduce the program task's own limits.

## User stories and acceptance criteria

### As a program optimizer, I want warm fitness without bypassing current correctness

Given matching scope, exact source/description, unexpired original samples and retrievable raw evidence,
when a second run evaluates that program after current correctness passes,
then fitness is explicitly `PersistentReuse`, original sample identity and uncertainty are preserved,
and current evaluator cost is zero while current correctness and store calls remain charged.

Given current correctness fails,
when the facade considers a warm candidate,
then neither cached nor fresh fitness is requested and no winner is promoted from that candidate.

### As an experiment owner, I want freshness and uncertainty rules to survive reuse

Given force-fresh, expired/future observations, changed applicability or failed raw-evidence verification,
when fitness is requested,
then a new evaluator call is required. Verification latency cannot extend the original sample lifetime.

Given no origin, a reused origin, mismatched scope, rejected/infeasible results, or unavailable raw evidence,
when an evaluator returns,
then the result is not published as fresh reusable evidence. No standard error, interval or raw sample
is invented. `ExistingSamples` requires the core policy's declared sample-count/uncertainty conditions;
`Deterministic` is an explicit caller claim, not a determinism test.

### As a budget owner, I want failed lookups to remain visible and bounded

Given insufficient logical store budget,
when the decorator attempts admission,
then the denied store method is not called. Fitness may still be acquired under its separately enforced
evaluation budget. Expected evidence I/O failures decline reuse; cancellation and identity drift propagate.

Given backend or evidence-policy identity changes after construction,
when a warm lookup is attempted,
then it fails before accessing the store. Metadata hashes are not an authorization or tamper boundary.

## Boundaries still open

This is consumer API integration, not a representative warm/cold runtime campaign. Production raw-evidence
storage, workload-specific stationarity/drift policy, joint ledger/engine checkpointing and a compatible
published core package remain required for their respective deployment stories. Scripted evidence stores
in unit tests validate control flow only. Origin-bearing LLM score blending remains fail-closed until a
combined-score provenance/uncertainty model exists; old uncertainty cannot describe a newly blended scalar.

## Verification (September 11, 2026)

The committed `tests/AiDotNet.Evolution.Integration.Tests` project links the existing focused tests rather than
copying them. The source-pinned workflow runs it independently on .NET 8 and .NET 10 and retains TRX/coverage.
It intentionally reuses the existing `AiDotNetTests` friend identity in an isolated test-host invocation; do not
load both this subset and the full test assembly in the same host. A local source-path reproduction is:

```powershell
dotnet test tests/AiDotNet.Evolution.Integration.Tests/AiDotNet.Evolution.Integration.Tests.csproj -c Release -f net10.0 -p:UseLocalEvolution=true -p:EvolutionProjectPath=C:/path/to/AiDotNet.Evolution/src/AiDotNet.Evolution/AiDotNet.Evolution.csproj
```

- 941 focused consumer tests pass separately on .NET 8 and .NET 10: 33 new persistent-fitness cases and
  seven provenance cases, alongside the prior 901 tests. These include the real facade and shared within-run
  accounting, but use authored fitness and scripted evidence availability, not a production benchmark store.
- Selected .NET 10 evolution namespace coverage: 5,672/6,095 executable lines (93.06%). The new evaluator
  covers 89/90 lines; correctness gate 35/35; program task 91/92; LLM judge 224/240. These are selected-path
  results, not whole-AiDotNet coverage. The added builder guards and accepted configuration paths are exercised.
  Across changed production C# executable lines, 111/112 are covered (including all five added builder lines).
- TRX and coverage are under `TestResults/program-cache-coverage-net8` and `program-cache-coverage-net10`.
  Production/test-copy AiDotNet DLL SHA256 agrees: .NET 10
  `AA716B43BB7D5080908254E60031AC5FE70D28FD99B342BF96307F1B6C2E6F9F`; .NET 8
  `819A28B697271590609E61C0D145F2250A2B26F90C423B3977C1AE97308D846D`.
- Normal .NET 8/.NET 10/.NET Framework 4.7.1 library builds succeed with existing repository warnings; no
  .NET Framework test run is claimed. The final .NET 10 run
  followed a non-incremental rebuild, avoiding an old compilation snapshot after edits during an earlier build.
- Ten coverage-gate checks pass. Downloaded hosted run `34556455936` confirms its two CLI attachments are
  byte-identical (SHA256 `D59A5D094F4C998BF9AB355A8682B3C4266DFF7B2A55354303C2BCA731FD6947`);
  the corrected gate passes their actual 268/270 benchmark lines. That older run passed all 79 compiler and
  42 CLI tests before its former single-path gate failed. This is not current-head hosted approval.
- Local integration and the companion workflow pin core `76d8343939f9ec1a58c4994384a249e6ecc93dc0`.
  The published preview is still too old; no successful clean package-path build or completed roadmap is claimed.
- The checked-in integration test project independently passes the same 941 cases on each modern framework;
  results are under `TestResults/program-cache-committed-net8` and `program-cache-committed-net10`.
- All 79 optional compiler/worker regressions also pass on each modern framework against matching current
  AiDotNet DLLs (`TestResults/program-cache-compiler-net8` and `program-cache-compiler-net10`).
