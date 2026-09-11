# Retained raw scalar observations for persistent program fitness

`DirectoryProgramSampleEvidenceStore` is an opt-in implementation of
`IProgramMeasurementEvidenceStore` for repeated scalar fitness measurements. It stores
individual sample IDs and values, not just a hash of an aggregate. Use it with
`PersistentProgramFitnessEvaluator`; keep the engine's `EnableEvaluationCache = false`
so run-local memoization cannot bypass current freshness/correctness checks.

The caller supplies genuine observations through a bounded, thread-safe capture callback.
`ProgramMeasurementStatistics.Calculate` computes an arithmetic mean and sample standard
error using 2–256 distinct observations. Use its `Version` in `EvolutionMeasurementOrigin`,
with the same ordered original sample IDs, observation time, acquisition cost and standard
error. The provider declines other statistics policies, confidence intervals, single samples,
infeasible outcomes and summaries that do not reproduce from the raw observations.
Different domains can implement the existing evidence-store interface with their own
explicit verification policies.

```csharp
var evidence = new DirectoryProgramSampleEvidenceStore(
    absolutePrivateDirectory,
    "my-evaluator-observation-schema-v1",
    async (program, measured, context, cancellation) =>
        await observations.ReadOriginalSamplesAsync(
            measured.MeasurementOrigin!.SampleIds, cancellation),
    maximumEntries: 4096);
```

`observations` above is caller-owned acquisition storage, not a built-in service. The
callback must return the actual `IReadOnlyList<ProgramMeasurementObservation>` behind
that measurement, or null; do not fabricate repeated values from a mean. The provider
does not execute program text, call models, start services or acquire samples during reuse.

On reuse it reopens the content-addressed file, verifies exact bytes against SHA256,
rejects ambiguous JSON, validates candidate codec bytes and all bound result/origin
metadata, and recomputes mean/standard error. Current evaluation cost and diagnostic
bodies are excluded from the binding: original acquisition cost stays in the origin,
whereas a cache lookup is not a new independent measurement. Scalar observations verify
scalar fitness; descriptors, objectives and metrics are bound metadata, not independent
measurements of those other quantities. Correctness remains outside the fitness cache
and runs on every facade dispatch.

Files are atomic, write-once and at most 2 MiB. Capacity refusal does not evict old evidence.
Cooperating writers use a short-lived exclusive `.writer.lock` handle; contention throws
`IOException`, which the existing decorator treats as unavailable persistence. There is
no unbounded waiting or retry loop. The marker is retained, while the OS releases the
handle on process exit. Equivalent writers must use the same private directory and
capacity policy. `HasTemporaryCleanupFailure` reports a failed cleanup without replacing
the original publication outcome. Protect the directory from hostile writers and aliases;
hashes do not authenticate a producer or attest hardware, stationarity or independence.

The facade meters `program_evidence_invocations` and `cache_store_invocations` as logical
method calls. Actual filesystem bytes/time, sample acquisition and callback costs require
separate caller accounting. A standard error assumes independent samples but cannot
establish that assumption; force fresh measurement for confirmation and never treat cache
hits or deterministic replay as additional statistical power.

Local contracts cover reopened storage, current correctness, force-fresh and expired
measurements, missing/corrupt raw artifacts, fabricated means/uncertainty, changed sample
identity/provider/candidate, capacity, cancellation, contention, strict JSON, byte limits
and finite statistics. Source-linked checks are a fast diagnostic only: the full AiDotNet
build's global imports/dependencies must also compile and pass the integration suite.
