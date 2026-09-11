# PR #2134 review and regression proof

Validation date: 2026-09-11. This record distinguishes actual execution from source inspection,
and intermediate failure evidence from the final corrected run.

## Reviewed changes

Remote baseline: `1472597173ae67e0ec3c26ebdd8dfd96e4a56fd3` (`fix/2087-heavy-timeout-oom`).
The local follow-up contains these separately reviewable corrections:

- `0d0c264db2`: put MemFlow's specific bounded constructor before the generic default-constructor
  fallback; correct the MelGAN comment to its actual `[1,80,1] -> [1,1,256]` fixture.
- `6c30d59b2f`: honor an explicit `WarmupInitialLearningRate = 0` in `TransformerNERBase`.
- `5e1e8edec7`: make `TransformerNERBase.GetOptions()` return the configuration the model actually
  uses, instead of unrelated `NeuralNetworkOptions` defaults.
- `c2dec2cf0f`: explicitly configure the positive starting rate in generated smoke fixtures;
  support a verified zero-rate preparation step in the shared transformer gradient fixture;
  bound TemplateNER/XLM-R test architectures and remove their unreachable duplicate branch.
- `48ef467ea2`: correct four test-only compile errors (typed model binding and the optimizer's
  `InitialLearningRate` property); the earlier production build had succeeded.
- `ab8be942c5`: explicitly declare ownership of nested ONNX configuration through the existing
  internal `IConfigurationCloneable` contract and central `CloneEngine.CopyConfiguration`;
  add direct/nested mutation-isolation controls and explicit legacy-framework CPU test setup.

No generated scaffold or handwritten model-family leaf was edited. No new skip, tolerance
relaxation, timeout increase, package pin, production architecture default, or GPU-selection
change was made. Public option types and the explicit-zero production contract are unchanged.

Original unresolved review threads:

| Thread | Root correction |
|---|---|
| [MemFlow constructor selection](https://github.com/ooples/AiDotNet/pull/2134#discussion_r3979643316) (`PRRT_kwDOKSXUF86hGCRY`) | Specific rule precedes the generic fallback; emitted-code and actual factory regressions cover both constructors. |
| [MelGAN fixture comment](https://github.com/ooples/AiDotNet/pull/2134#discussion_r3979643302) (`PRRT_kwDOKSXUF86hGCRO`) | Comment matches the existing one-frame fixture; runtime checks retain channels, hop size, residual stacks and sample rate. |

All review-thread/comment pages were read: five total threads, two unresolved at the reviewed
remote head. The other three workflow threads were already resolved. This document does not
itself resolve any GitHub thread or assert that subsequent remote reviews are unchanged.

## Failure-first evidence

The small harnesses link the checked-in regression source and the actual generator/library.
They do not replace the production algorithms with mocks. The generator tests deliberately
use minimal compiler metadata; the full test assembly separately executes the real factories.

| Regression | Before | After targeted correction |
|---|---|---|
| MemFlow-specific rule shadowed by default constructor | 1 failed, 3 controls passed | 4 passed |
| Explicit-zero warmup configuration and actual first training update | 3 failed, 4 controls passed | 7 passed |
| `GetOptions` actual identity, copy and trained clone | 5 failed: expected `TransformerNEROptions`, received generic `NeuralNetworkOptions` | All 5 original contracts passed in the expanded 7-case final class |
| Positive warmup explicitly selected in all 18 relevant generated transformer identities | 18 failed, 4 existing controls passed | 22 passed |
| TemplateNER/XLM-R bounded constructor and matching declared input | 2 failed, 22 controls passed | 24 passed |
| Nested ONNX configuration ownership through the actual shared clone engine | 3 failed, 4 controls passed; direct ONNX copy already independent | All 7 cases passed in the final actual assembly |

The warmup failure was substantive: requesting zero with peak rate `0.01` and four warmup steps
produced a starting rate of `0.0025`, and the first `Train` changed weights. The corrected tests
require every initial weight to remain unchanged on that step, then require a finite, nonzero
weight update at the following positive rate. Positive starts, a copied zero setting, disabled
warmup, and negative-input rejection retain their separate checks.

The wider family run then exposed a distinct test assumption: the shared gradient invariant
asserted a weight update after exactly one `Train`, including a deliberately zero-rate step.
DistilBERT/ELECTRA also fell just short of the unchanged 60-step, 1% memorization threshold
(`2.804178 -> 2.776294`). The generated smoke helper explicitly preserves the previously tested
positive-start trajectory (`LearningRate / WarmupSteps`) without changing production semantics,
configured positive starts, disabled warmup, iteration budgets, or loss thresholds.

The legacy TinyBERT fixture does not go through the generator. Its shared-base preparation is
restricted to actual transformer options plus the actual model-owned optimizer: per-batch
stepping, optimizer and scheduler both at step zero and rate zero, and a finite positive next
linear-warmup rate. It verifies the zero step preserves all parameter-chunk hashes, keeps every
parameter finite, and advances both states/rates exactly once. The original nonzero-change and
full finite assertions then run unchanged. Private optimizer inspection is read-only; no state
is rewritten. Controls include positive starts, custom/no-scheduler optimizers, epoch stepping,
advanced/cloned state, scheduler/optimizer mismatch, broken zero steps, and a broken first
positive update that must still fail the shared invariant.

Actual execution also caught nested configuration sharing that source review had not proved:
the top-level cloned `TransformerNEROptions` and label list were independent, but `OnnxOptions`
was the same mutable object. This type did not implement the clone engine's existing ownership
contract. Its explicit implementation delegates to the central property plan; it adds no
type-name heuristic or handwritten clone field list. The before-control directly copying the
ONNX object already passed, confirming the delegated entry does not recursively call itself.
The nested and trained-clone regressions also mutate a scalar device setting, fallback-provider
list and custom-options dictionary independently; provider/device/path defaults are unchanged.

## Intermediate runs are not final proof

The first actual net10 assembly (after only the MemFlow generator correction) built with zero
errors and ran all four originally affected classes: MemFlow 34 passed, MelGAN 33 passed,
KMaXDeepLab 33 passed, LegalBERTNER 37 passed. Each had its pre-existing opt-in performance-census
skip. Seven new generator/runtime tests also passed: 144 passed in total, zero failed, four skips.

An expanded run against that old positive-start binary was intentionally stopped after 22 of
28 NER fixtures: 809 passed, zero failed, 23 existing skips. TemplateNER was interrupted. This
was not a complete-family pass.

The next actual full assembly included the correct production zero-start behavior. Its 14
focused generator/runtime/warmup cases passed. Before the explicit smoke/guard corrections,
25 NER classes completed: 898 passed, 20 failed, 26 existing skips. The failures were the
zero-step weight-change assertions and the two fixed-budget memorization cases described
above. XLM-R was interrupted; the two manual fixtures had not yet run in that phase.
Only the identified owned obsolete test process tree was stopped; its logs/TRX files were retained.

TemplateNER completed that phase in **8.5156 minutes**. Its isolated testhost's working set was
observed at **32.68 GiB**, exceeding a 16 GiB hosted-runner envelope even without other classes.
That is a measured lower bound, not a claimed exact lifetime peak or an inferred OOM exit cause.
The corrected generated profile uses hidden width 32, four heads, two encoder blocks, FFN 64,
maximum sequence 16 and nine labels, matching declared input `[8,32]`. Actual topology checks
retain two transformer blocks, their dropout layers, and the dense classification head.
The live branch's dropout remains `0.1`, its rate remains `5e-6`, and production options remain
768/12/12/3072/256 with rate `5e-5`. The deleted later XLM-only branch was unreachable; its
different dropout setting was not copied into the live fixture.

The next net10 full-test compile caught four new test-only compile errors; correcting those
produced a successful test-only rebuild in 2:14.48 with 3,967 existing warnings and zero errors.
Its focused run was **55 passed, two failed** out of 57: both failures exposed the ONNX alias
above. That run is retained as a negative control, not presented as the final green suite.

Against that same `48ef467ea2` test binary, the isolated bounded TemplateNER class passed 37
tests with one existing census skip in **36.0212 seconds**, with an observed testhost peak
working set of **0.7478 GiB** (802,963,456 bytes; 49 samples). XLM-R passed 37 with one census
skip in **34.0629 seconds**, with **0.7480 GiB** (803,196,928 bytes; 47 samples). Each used one
testhost. These are sampled Windows peak-working-set values, not a hard memory-capped run.
They establish the architecture improvement; the post-ownership-correction combined family run
below separately checks the final binary and shared-process retained state.

## Final corrected assembly and results

Source under test: `ab8be942c5` and its preceding correction commits. Environment:
Windows build 26220, .NET SDK 10.0.401, Ryzen 9 3950X, approximately 64 GiB RAM. CPU execution,
Workstation GC, serialized xUnit collections. The final NER inventory runs together in one
testhost, after the isolated diagnostic runs, to check shared-process contamination and footprint.

The final net10 full project build completed with **zero errors**, 6,818 reported warnings, in
**7:04.70**. Its focused suite passed **59/59**, with zero failures/skips, in **32.6040 seconds**:
24 generated-code contracts, five actual generated-factory runtime checks, seven production
warmup contracts, seven options/clone contracts, and 16 shared smoke-profile/guard controls.
The parent reviewer independently replayed these same actual-main contracts on all three
frameworks: **59 passed, zero failed/skipped on each of net10.0, net8.0 and net471**. The
separately produced TRX files are under `%TEMP%/pr2134-model-family-results-20260911/`:
`root-final-ner-onnx-contracts.trx`, `root-final-ner-onnx-contracts-net8.0.trx`, and
`root-final-ner-onnx-contracts-net471.trx`. Each file's actual result entries were counted.

An additional **47/47** existing clone/ONNX-option controls passed with zero skips in
**17.5366 seconds**: CloneFidelity (5), shared-engine SerializationShell (1), CopyOnWriteClone
(10), and ONNX option defaults/factories (31). This does not include the unbounded all-planned-
types default-construction census.

Final actual assembly SHA-256 values:

| Framework | Assembly | SHA-256 |
|---|---|---|
| net10.0 | AiDotNetTests.dll | `69935EF842B2441C3148D9DF440810D56D9E439BBDD8667B43374794EF0A02D4` |
| net10.0 | AiDotNet.dll | `EE1023E56D8BF9D2BCF505958B7B6E7800970D9FA15104CBD4E723FA670175E5` |
| net8.0 | AiDotNetTests.dll | `29E3D334E6F65135EEDC7C08849CABF02495131B0DA522813B826EA19FEC2123` |
| net8.0 | AiDotNet.dll | `6819D49DBE398E519AC9A854C3F51551FAB399BAC2CD74CD4694C58234D5A2E2` |
| net471 | AiDotNetTests.dll | `C765920D392317AFB42EF928FA79DF5A65BC0420EBF8072828727E321DD6B923` |
| net471 | AiDotNet.dll | `1CCF6DA1F0284BFCB7D97B2E7F7568D6A06D657223D4251380C8F891129AD760` |

The full test-project compatibility build for **net8.0 and net471** completed with **zero
errors**, 13,378 reported warnings, in **12:46.44**. It did not rebuild or overwrite net10
outputs. This is compile evidence; framework-specific runtime evidence is recorded separately.

The **all-28 shared-process NER run passed**: **1,029 passed, zero failed, 29 existing skips**
(1,058 total results). Required-class checks confirmed every listed fixture contributed results
in exactly one testhost. The 29 skips are 28 opt-in performance-census tests and the pre-existing
Biaffine span-target correctness skip; no skip was added or relabeled.

The run took **19.9154 minutes** (wrapper wall time 1,196.913 seconds). Across 1,560 owned-
process samples, the observed peak working set was **9,475,612,672 bytes / 8.8249 GiB**.
The sampler tracked the launched process's testhost ancestry, excluding independent reviewer
replays in the same worktree. Compatibility compilation and separate reviewer replays occurred
during this run; elapsed time is an observed run duration, not an isolated benchmark claim.
This combined result specifically checks cross-fixture retained state and contamination; it is
not inferred from separate per-class successes.

The final originally affected model/clone-mode cohort also passed: **109 passed, zero failed,
three existing census skips**, in **2.8460 minutes**. This comprises MemFlow (34 passed),
MelGAN (33), KMaXDeepLab (33), and CloneMode (9). The three model classes each retain their
existing census skip; CloneMode has no skips. The actual final-assembly result is
`%TEMP%/pr2134-onnx-fixed-results-20260911/original-models-and-clone-modes.trx`.

The inventory is **all 28 existing concrete NER-derived model-family fixtures** in this compiled
assembly, not every production NER type. BERTNER, for example, has no fixture in that inventory;
no new production-model coverage claim is made for it.

## Reproduction commands

Run these blocks in order, in one PowerShell session from the repository root, after restoring
the unchanged published dependencies. Stop on any build/test failure; do not continue against
stale binaries after an unsuccessful build:

```powershell
$ErrorActionPreference = 'Stop'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$env:AIDOTNET_FORCE_CPU = '1'
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-restore -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'net10 build failed; do not run stale test binaries.' }
& ./.github/scripts/harden-xunit-runner.ps1 -RunnerJson tests/AiDotNet.Tests/bin/Release/net10.0/xunit.runner.json
if (-not $? -or $LASTEXITCODE -ne 0) { throw 'Runner hardening failed; do not start tests with an unhardened runner.' }
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build --no-restore --filter 'FullyQualifiedName~AiDotNet.Tests.Generators.GeneratedHeavyFixture|FullyQualifiedName~AiDotNet.Tests.Generators.TransformerNERSmokeFixtureContractTests|FullyQualifiedName~AiDotNet.Tests.UnitTests.NER.TransformerNERWarmupTests|FullyQualifiedName~AiDotNet.Tests.UnitTests.NER.TransformerNEROptionsContractTests' --logger 'trx;LogFileName=final-focused.trx'
if ($LASTEXITCODE -ne 0) { throw 'Focused regression tests failed.' }
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build --no-restore --filter 'FullyQualifiedName~AiDotNet.Tests.IntegrationTests.Cloning.CloneFidelityTests.|FullyQualifiedName~AiDotNet.Tests.UnitTests.NeuralNetworks.CopyOnWriteCloneTests.|FullyQualifiedName~AiDotNet.Tests.IntegrationTests.Cloning.CloneRoundTripTests.SerializationShell_|FullyQualifiedName~AiDotNet.Tests.IntegrationTests.Onnx.OnnxIntegrationTests.OnnxModelOptions_|FullyQualifiedName~AiDotNet.Tests.IntegrationTests.Onnx.OnnxDeepMathIntegrationTests.OnnxModelOptions_' --logger 'trx;LogFileName=clone-controls.trx'
if ($LASTEXITCODE -ne 0) { throw 'Existing clone/ONNX controls failed.' }
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build --no-restore --filter 'FullyQualifiedName~AiDotNet.Tests.ModelFamilyTests.Generated.MemFlowTests.|FullyQualifiedName~AiDotNet.Tests.ModelFamilyTests.Generated.MelGANTests.|FullyQualifiedName~AiDotNet.Tests.ModelFamilyTests.Generated.KMaXDeepLabTests.|FullyQualifiedName~AiDotNetTests.UnitTests.Serialization.CloneModeTests.' --logger 'trx;LogFileName=original-models-and-clone-modes.trx'
if ($LASTEXITCODE -ne 0) { throw 'Original model-family/clone-mode checks failed.' }
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release --no-restore -m:1 -p:CompatBuildOnly=true -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'Compatibility build failed; do not run stale test binaries.' }
```

The actual reflected inventory, pinned to the reviewed source, is reproduced by this class list:

```powershell
$generated = @(
    'BLINKNER', 'BiaffineNER', 'BioBERTNER', 'CNNBiLSTMCRF', 'ClinicalBERTNER',
    'DeBERTaNER', 'DistilBERTNER', 'ELECTRANER', 'FinBERTNER', 'InstructionNER',
    'LegalBERTNER', 'ONNXNER', 'PURENER', 'PromptNER', 'PubMedBERTNER', 'PyramidNER',
    'RELNER', 'RoBERTaNER', 'SECBertNER', 'SciBERTNER', 'SpERTNER', 'SpanBERTNER',
    'TemplateNER', 'TriaffineNER', 'W2NER', 'XLMRoBERTaNER'
)
$fixtures = @($generated | ForEach-Object { "AiDotNet.Tests.ModelFamilyTests.Generated.$($_)Tests" })
$fixtures += 'AiDotNet.Tests.ModelFamilyTests.NeuralNetworks.BiLSTMCRFTests'
$fixtures += 'AiDotNet.Tests.ModelFamilyTests.NeuralNetworks.TinyBERTNERTests'
if ($fixtures.Count -ne 28) { throw 'NER fixture inventory mismatch.' }
$filter = ($fixtures | ForEach-Object { "FullyQualifiedName~$_." }) -join '|'
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build --no-restore --filter $filter --logger 'trx;LogFileName=NER28Combined.trx'
if ($LASTEXITCODE -ne 0) { throw 'Combined NER fixture run failed.' }
```

The trailing dot in each filter prevents class-name prefix collisions. Count TRX
`UnitTestResult` elements by `Passed`, `Failed`, and `NotExecuted`; do not interpret a missing
`ResultSummary.Counters.notExecuted` attribute as zero skips. The broader NER inventory has
one pre-existing Biaffine span-target correctness skip in addition to the opt-in census skips.
Those two categories must not be conflated.

Preserved local negative-control artifacts (under `%TEMP%`, not remote CI artifacts):

- `pr2134-generator-review-harness-20260911/TestResults/`: `before-generator-fix-confirmed.trx`,
  `after-generator-fix.trx`, `before-ner-smoke-profile.trx`, `after-ner-smoke-profile.trx`,
  `before-two-ner-shape-bounds.trx`, `after-two-ner-shape-bounds.trx`.
- `pr2134-warmup-contract-results-20260911/`: `warmup-before-fix.trx`, `warmup-after-fix.trx`,
  `options-before-fix.trx`, `onnx-ownership-before.trx`. The warmup harness links the checked-in seven-test class;
  `dotnet test <harness>/warmup/WarmupReviewTests.csproj -c Release --no-restore -m:1 -p:BuildProjectReferences=false -p:UseSharedCompilation=false --filter FullyQualifiedName~TransformerNERWarmupTests` replays it against the already-built library.
- `pr2134-model-family-results-20260911/`: initial four-class and seven-regression proof.
- `pr2134-ner-family-results-20260911/`: the explicitly partial 22-class old-binary run.
- `pr2134-final-ner-results-20260911/`: the explicitly partial 25-class zero-start run;
  despite the historical directory name, this is **not** the final corrected family proof.
- `pr2134-verified-results-20260911/final-focused.trx`: the intermediate 55/2 ONNX alias
  negative control, before `ab8be942c5`; despite its historical filename, it is not green proof.
- `pr2134-verified-ner-results-20260911/`: the isolated TemplateNER/XLM-R shape measurements
  at `48ef467ea2`, before the final ONNX ownership correction.
- `pr2134-onnx-fixed-results-20260911/`: `onnx-fixed-focused.trx` (59/59) and
  `onnx-fixed-clone-controls.trx` (47/47) against the final `ab8be942c5` assembly.
- `pr2134-onnx-fixed-ner-results-20260911/NER28Combined.trx`: the complete final 28-class,
  single-process run; its normal-verbosity stdout/stderr logs are in the same directory.
- `pr2134-onnx-fixed-ner28-measurement-20260911.log`: final process count, wall duration,
  sample count and observed peak. `pr2134-measure-combined-20260911.ps1` is the local sampler;
  it checks result presence for each required fixture and rejects a multi-testhost proof.
- `pr2134-onnx-fixed-net10-build-20260911.log` and
  `pr2134-onnx-fixed-compat-build-20260911.log`: final build logs with unsuppressed warnings.

The earlier draft `before-generator-fix.trx` had an incomplete mock vocoder hierarchy and is
not used as the negative control; the `-confirmed` artifact is authoritative.

## Limits

This includes local net10 CPU family evidence and 59 focused runtime contracts on each of the
three supported frameworks. It is not a hosted full-nightly run, a 16 GiB memory-capped test,
full-family runtime coverage on every framework, or a GPU performance measurement. Production
GPU routes are unchanged. Repository/analyzer warnings are reported rather than hidden.

The nightly workflow's process chunking limits inter-process retained state; it does not make
a catastrophic runner/OOM termination recoverable, nor does it impose a hard per-chunk memory
ceiling. A shutdown signal alone does not establish OOM causality. The unchanged hosted job
budget and the final remote CI/review status still need their own checks after the reviewed
commits are pushed.
