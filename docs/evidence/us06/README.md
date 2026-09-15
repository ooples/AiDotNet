# US-06 consumer implementation and acceptance evidence

Core [Evolution #49](https://github.com/ooples/AiDotNet.Evolution/pull/49), consumer [AiDotNet #2210](https://github.com/ooples/AiDotNet/pull/2210), tracking [issue #24](https://github.com/ooples/AiDotNet.Evolution/issues/24).
The companion builds on AiDotNet #2203 and retains #2148; no Tensors source change is required.

## Verified revisions and checks

Consumer implementation/workload: `38d65dc316febbafaf1fbeeef0b484620eb38e01`.
Evidence-verifier revision: `6573da7746f9fd0236d5729538b484d1e77a9792`.
Core dependency: `e1a791c8d6549adc9ef0f5b40e1668ff7fb12dec`.

- Real consumer Release builds: net10.0 and net8.0 succeeded with 2,849 warnings each; net471 succeeded with
  2,778 warnings. Zero errors on all targets. Logs retain warnings from the existing library/generators/tests;
  these are not described as warning-free builds.
- Consumer evolution/facade/persistence suites: **954 passed on net10.0 and 954 on net8.0**, zero failures/skips,
  including 13 new noise-workflow cases per target. net471 is compilation compatibility, not a claimed consumer test run.
- New production source whitespace verification passed. The benchmark build had zero warnings/errors.
- The fixed 18-run consumer study passed. The evidence verifier accepted it and rejected all six corrupted variants.
- Hosted pinned-source integration on the implementation revision passed both framework jobs in
  [run 34975478285](https://github.com/ooples/AiDotNet/actions/runs/34975478285), including compiler/worker/facade regressions and the workload executable.
  Later documentation/verifier heads have their own hosted checks; this link is not a claim about a future head.

## Given / When / Then proof

**Given** a promising program/configuration, **when** it challenges an incumbent, **then**
`ProgramNoiseEvaluationSession.ChallengeAsync` performs correctness-gated full search and separate hidden
correctness/confirmation. All 12 actual RidgeRegression comparisons confirmed alpha0 over alpha10000; none
confirmed the reverse. Trusted sorting comparisons retained broad intervals and did not claim a timing win.
These outcomes verify this fixed experiment, not general superiority or a guarantee against every lucky draw.

**Given** cheap screening, **when** a frozen candidate set is screened, **then** `ScreenAndAuditAsync` automatically
audits preselected complete rejects at full fidelity, preserving rejected identities, fresh samples and uncertain
or incomplete results. The conservative ridge threshold .5 met the predefined approval rule on all six roots:
its census audit interval for useful-among-rejects was [0,0]. The aggressive .98 threshold failed on every root:
five audit intervals were [.5,.5], one [2/3,2/3]. The zero-ms sorting negative control returned [1,1] on all six roots
and was rejected. **Those failed presets are retained, not retuned or removed.**

**Given** fitness remeasurement, **when** it runs, **then** raw version-pinned backends bypass engine/persistent caches,
and reused/preaggregated origins are refused before correctness metadata can be lost during result combination.
The study executed **5,768 fresh model fits + 1,680 sorting invocations + 6,608 correctness checks = 14,056 charged calls**.
Every model fit owns a new RidgeRegression instance and independent train/test draws. Every sort invocation resets
the input; one warmup plus one measured invocation produces one fitness sample. Tests cover failed checks,
cancellation, restored one-use batches, unknown costs and budget-short audits.

## Reproduce / raw artifacts

See [fixed design, invocation and assumptions](../../../benchmarks/EvolutionNoise/README.md).
[consumer-validation.zip](consumer-validation.zip) includes all raw observations and ledger receipts, assembly hashes,
TRX files, build/format logs, verifier results and package-path CI failure logs.
SHA-256: `df02ef6a2b556353d6ba81cd971fde286bfe09bece05aee24b44c053a5e15131`.
Core's 704/658/658 tests and coverage evidence are linked from Evolution PR #49.

## Remaining dependency gate—not hidden unfinished consumer code

The ordinary consumer jobs resolve the older published Evolution preview. They fail to resolve both inherited
foundation APIs (for example `EvolutionReuseScope`, `EvolutionResourceLedger`) and new noise APIs
(`EvolutionIncumbentChallengeReport`). The pinned-source workflow succeeds with the matching core revision.
This does **not** prove compatibility with the older published package or make those package jobs green.

US-06 consumer/core implementation and the acceptance experiment above are delivered. Release/package integration,
review and merge remain gated by the shared foundation/dependency chain. Issue #24 remains open for that gate;
do not advance to another story or silently bypass package CI. Merging/publishing dependencies requires separate
authority; nothing here merges, publishes or executes arbitrary generated code. Validation applies to the stated
trusted workloads and environment, not all future consumers, external data or untrusted-code isolation.
