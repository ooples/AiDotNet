# PR #2130: audio/visual correspondence execution proof

## Scope and contract

The shared AVC factory previously ignored the requested encoder depth and embedding
width. It always created four Dense/LayerNorm/Tanh blocks plus four fusion layers.
The model then classified fourteen of those layers as attention/FFN triplets, so
real audio and visual entry points threw the modulo-three guard. The existing
feature-input model family did not exercise those paths.

The correction uses an explicit encoder/fusion layout, distinct audio/visual input
adapters, and a pair-input adapter that executes the complete fusion stack. Actual
encoder depth and final width now honor the options. The default remains six
blocks, rather than silently changing the public default to the old ignored four.
This changes checkpoint topology; no automatic old-checkpoint migration is claimed.
Custom architecture layers retain their literal Predict/Train contract without
invented modality roles. Optional layer-owner fields participate in the existing
generator; private throwing guard methods are not duplicate ownership declarations.

Correspondence training recomputes the full paired objective from real inputs on
the shared tape-based training boundary. The same shared helper and auxiliary-layer
lifecycle corrections are coordinated with PR #2136. No generated test leaf,
existing assertion, learning rate, timeout, or public default is relaxed.

Feature-only custom architectures do not acquire invented positional parameters.
Image channel validation runs at the visual-input boundary, not the feature-input
options boundary: the existing `Channels=0` feature-only control remains unchanged.

The shared generator now distinguishes neural layer checkpoints from ModelBase's
flat parameter checkpoints. Raw neural Tensor/Vector/Matrix fields are declared
once in the existing named state envelope; canonical/additional layers remain
owned by the layer serializer. Readonly neural storage uses shape-checked in-place
registry restoration. Non-neural readonly numeric eligibility is unchanged
(including WeightedRegression's per-observation buffer); this is not a claim to
have migrated that independent persistence contract. No checkpoint version changes.
Older checkpoints missing raw values retain constructor state; values never saved
cannot be recovered. The corrected AVC six-block topology is a separate compatibility
limitation noted above.

The generic custom-objective tests preserve PR #2136's thirteen semantic assertions.
Only their observer optimizer adapter differs: PR #2130 exposes public virtual
`Step` and no enclosing `NoGradScope`, so its observer uses `Step`/`context.Reevaluate()`;
PR #2136's newer optimizer uses `StepCore`/`ReevaluateWithGradients`. No unrelated
optimizer implementation was imported and no byte-identical-test claim is made.

## Before evidence

All baseline executions use the actual compiled library, not a source-linked model:

- `AiDotNet.dll` SHA-256: `6798A4DBA4BC7C40C5B54F7B047601A5537B733BF0B5637CE0D8836891455097`.
- Final baseline test assembly SHA-256: `77ABBADCFEF13CA86F46973C4B2E6943C3CC65CAF06F85BAA9EB573FE54D39F9`.
- Existing unchanged AVC family: **29 passed, 0 failed, 1 existing opt-in performance-census skip**, 14 seconds.
- New 28-case execution cohort: **3 passed, 25 failed, 0 skipped**, 3 seconds.

The three passing controls are two default batched feature-input Predict cases and
independent factory re-enumeration. The failures cover topology/options, actual
audio/visual execution, channel geometry, pair fusion, rectangular localization,
nondefault parameter clone/serialization, and fresh paired-objective gradients.
Input-dependence assertions reject constant embeddings; clone controls first alter
the source parameters so deterministic reinitialization cannot masquerade as copying.

Local TRX files are under `artifacts/pr2130-avc/results/`:
`avc-family-baseline.trx` and `avc-execution-bounded-baseline.trx`.
Earlier smaller baseline cohorts are retained separately, not added to these counts.

Intermediate actual-library checks used core SHA-256
`A98360AF0A33125FD793BAC3E3D5ACA749E4BDE64F9C97888F8A66659B981203`:

- AVC execution: **26 passed / 2 failed**, exposing custom-layer extra positional
  storage and positional tensor serialization loss. The latter was isolated to raw
  tensors after canonical layer and auxiliary adapter segments round-tripped exactly.
- Existing native/options cohort: **552 passed / 1 failed**, exposing misplaced
  `Channels` validation on feature-only options; its original assertion is retained.
- Generic shared training controls: **10 passed / 3 failed** before the recursive
  auxiliary-parameter collection and training-mode propagation fix.
- Actual generated mutable raw-state controls: **0 passed / 2 failed before**,
  **2 passed / 0 failed after** the generator-only ownership correction, against the
  same frozen core. Readonly expansion then reproduced **2 readonly failures** while
  the **2 mutable controls stayed green**. Global-namespace/friend-identity harness
  setup errors are not counted as production negative controls.

The final execution tests additionally strengthen ownership/serialization checks and
move the channel negative control to its actual consumer boundary. They are not
represented as byte-identical to the earliest 28-case baseline. Relevant reports:
`avc-execution-intermediate.trx`, `avc-positional-serialization-negative.trx`,
`avc-shared-boundaries-before.trx`, `avc-state-generator-runtime-before.trx`,
`avc-state-generator-runtime-after.trx`, and `avc-readonly-state-before.trx`.

## After evidence

The final actual-library build succeeded with **0 errors / 2,775 warnings** in
5m04s. The focused test assembly then built with **0 errors / 0 warnings**. Two
earlier failed compile attempts exposed the non-neural readonly eligibility
boundary and typed restore/shape expressions; their logs are retained separately,
not represented as successful builds or stale-binary runtime proof.

The complete AVC runner executed in one serial process: **106 passed, 0 failed,
1 existing opt-in performance-census skip** (107 total). No correctness test was
skipped. The measured wall time was **93.32 seconds**, and the sampled peak working
set of the owned test process tree was **932.0 MiB** (400ms sampling). These are
bounded execution/resource observations, not a comparative performance benchmark.
The old failing execution run stopped many tests before meaningful model work, so
its shorter duration is not a useful throughput baseline.

| Cohort | Passed | Failed | Existing skips |
| --- | ---: | ---: | ---: |
| Unchanged AVC model family | 29 | 0 | 1 |
| AVC execution, ownership, training, clone and serialization | 28 | 0 | 0 |
| Shared custom-objective and auxiliary-layer lifecycle | 13 | 0 | 0 |
| Generated mutable/readonly raw ownership and legacy checkpoints | 10 | 0 | 0 |
| Readonly numeric storage, shape/null guards, views and copy-on-write | 20 | 0 | 0 |
| Existing model-state registry controls | 6 | 0 | 0 |

Exact final artifacts:

- Actual core: `artifacts/pr2130-avc/final/bin/AiDotNet/release_net10.0/AiDotNet.dll`,
  SHA-256 `E0865AFB66E8BA9894E9D3A12FBE238C48E053968B15C047F6A724B6CD984859`.
- Runner: `artifacts/pr2130-avc/final/bin/AiDotNet.AudioVisualCorrespondenceReview/release_net10.0/AiDotNetTests.dll`,
  SHA-256 `7C5B96C2A2882BE9A5604F58723DC1F261BD1CED48F0B2B1E0C18C654F0ECB6D`.
- Report: `artifacts/pr2130-avc/results/avc-combined-final.trx`,
  SHA-256 `F2064C99AB51067BA7B81EFDD245ADE290CD14AB7FA3E265C54F173EEC75EFE0`.
- Build/test logs: `artifacts/pr2130-gpt4-width/avc-final-core-typed-shape-build.log`,
  `avc-final-tests-build.log`, and `avc-combined-final.log` in that same log directory.

An independent agent also replayed the actual native/options/generator cohort
(**693/693 passed**) and the original CLIP/VideoCLIP ONNX graph-contract cohort
(**33/33 passed**), both with zero skips against the same `E086...` core. The
original feature-only `Channels=0` assertion therefore passes unchanged. These
results do not cover the separately pending GPT4/Finch ONNX work, new published
packages, all model families, or GPU performance. This AVC/shared runtime replay
is net10.0; the three-target scalar checks below are not represented as full
three-target runtime validation.

## Narrow follow-up review contracts

- Review comments `3993370973` / `3993370978`: the sequence factory oracle now
  semantically compiles every one of its six effective factories against the actual
  library, with twelve typed API-drift negative controls. Exact old-generator
  `671836a347436f4b023ebe3f02a86ea333ad0686`: **27 passed / 4 original structural
  failures**; current: **31 passed / 0 failed**, no skips. Complete commands, pinned
  baseline properties and limits are in `tools/SequenceFixtureReview/README.md`.
- Review comments `3993370899` / `3993370945`: Document's existing copy constructor
  is public; Jamba/Mamba2/XLSTM's migration-added validators are internal. No body or
  validation assertion changed. At master `3185b41f1e1cffb81d76130e319233153c984471`
  these options had neither these public validators nor Document's copy constructor,
  so no established published validator API was removed. Actual A983 baseline:
  **6 failures / 0 passes**. The full source-linked options suite after these edits:
  **540 passed / 0 failed / 0 skipped on each of net10.0, net8.0 and net471**.
  The six actual-library API controls subsequently passed within the independently
  replayed 693-case native cohort against the final `E086...` core.

Reports: `sequence-document-boundary-before.trx` and
`sequence-document-scalar-<tfm>.trx` under `artifacts/pr2130-avc/results/`;
sequence reports under `artifacts/pr2130-sequence-semantic/results/`.

## Reproduction

From the repository root, build the actual library and focused source-linked test
scaffold into a new artifact tree. The runner references the real project; production
AVC implementation files are not compiled into the test project.

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$avcArtifacts = "artifacts/pr2130-avc/replay-$([Guid]::NewGuid().ToString('N'))"
dotnet build tests/AiDotNet.AudioVisualCorrespondenceReview/AiDotNet.AudioVisualCorrespondenceReview.csproj `
    -c Release -f net10.0 --artifacts-path $avcArtifacts -m:1 `
    -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false `
    -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'AVC actual-library build failed' }
$avcRunner = "$avcArtifacts/bin/AiDotNet.AudioVisualCorrespondenceReview/release_net10.0"
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 -RunnerJson "$avcRunner/xunit.runner.json"
if ($LASTEXITCODE -ne 0) { throw 'AVC runner hardening failed' }
dotnet vstest "$avcRunner/AiDotNetTests.dll" `
    '--Logger:trx;LogFileName=avc-combined-final.trx' `
    "--ResultsDirectory:$avcArtifacts/results"
if ($LASTEXITCODE -ne 0) { throw 'AVC combined regression suite failed' }
[xml]$avcReport = Get-Content -LiteralPath "$avcArtifacts/results/avc-combined-final.trx" -ErrorAction Stop
$avcCounters = $avcReport.TestRun.ResultSummary.Counters
if ([int]$avcCounters.total -ne 107 -or [int]$avcCounters.passed -ne 106 -or [int]$avcCounters.failed -ne 0) {
    throw 'Unexpected AVC test census; inspect all results before accepting this replay'
}
$avcSkipped = @($avcReport.TestRun.Results.UnitTestResult | Where-Object { $_.outcome -eq 'NotExecuted' })
if ($avcSkipped.Count -ne 1 -or $avcSkipped[0].testName -ne 'AiDotNet.Tests.ModelFamilyTests.NeuralNetworks.AudioVisualCorrespondenceNetworkTests.ModelPerformanceCensus') {
    throw 'Unexpected AVC skip; correctness tests must not be skipped'
}
```

`CopyLocalRuntimeTargetAssets=false` limits local copied native assets; it does not
change production engine selection or remove GPU support. CPU execution proof is
not GPU performance proof. The existing two-logit-derived task APIs are tested for
execution/shape/finite outputs, not claimed as validated task accuracy. The paired
positive-similarity objective is not a benchmark proving representation quality,
nor does this work claim a trained separation/localization model.
