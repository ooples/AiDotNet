# PR #2130: bounded native-options review proof

This is a native-construction/geometry batch, not approval of every phase-3 API.
The PR must remain draft while the ONNX/configuration and other open items below
are unresolved. No full-shard or GPU execution claim is made by these CPU results.
These are local Windows runs, not hosted CI or Linux execution evidence.

## Source checkpoints

- Original reviewed PR: `597ae5849ca5f89e132de2ac674cbefeb594e824`.
- First native fixes and original 92-case harness: `2a5eb037312fe46674a7b6ab5864f465f557d743`.
- Integrated reviewed #2128: `78d4c176b871d12ca4773a5c5d756999f6e96b09`.
- Frozen merged implementation: `0b1932c294acf4e0a8f0eda0b6d60729917f51fd`.

The #2128 merge was clean. Phase-3 callers were migrated to the shared
`VisionDim` / `VisionLayers` names; unrelated vision-language option hierarchies
and persisted metadata keys were not renamed. `VisionScanPattern` moved into its
own source file without changing its namespace, members, or values, allowing the
real enum and options to be compiled in the scalar-only runner.

## Reproduced defects and retained controls

The original 92-case harness compiled the untouched original production project,
then executed 92 actual tests: **69 failed, 23 passed, zero skipped**. These are
not 69 independent defects: many boundary cases encounter the same bad common
validation gate before reaching the particular property under test.

| Control | Original behavior | Fixed/retained behavior |
| --- | --- | --- |
| Small VisionMamba native constructor | Rejected unused `VisionMambaOptions.VocabSize = 0` | Constructs; real float/double prediction, training, parameters and scan-pattern tests pass |
| Small audio-visual correspondence constructor | Rejected unused `MaxSequenceLength = 0` | Constructs and executes its existing native graph; ignored width/depth configuration is **not** claimed fixed |
| Small audio-visual event-localization constructor | Rejected unused `MaxSequenceLength = 0` | Constructs its real dual-stream graph; zero-depth remains valid; depth 0/1/2 yields 18/20/22 layers and 8/10/12 attention layers |
| Small unified multimodal constructor | Rejected unused `ImageSize = 0` | Constructs and predicts; physical layer widths and depths change with configuration |
| Flamingo, image 32 and configured patch 8 | Produced 4 vision tokens | Produces 16 tokens, plus 4 resampled tokens |
| Flamingo, image 31 and configured patch 8 | Produced 4 vision tokens | Produces 9 tokens: intentional floor cropping is preserved |
| Generated Flamingo factory | No patch initializer; only one LM layer; real factory produced `[4,64]` vision features | Generator emits patch 8 and four LM layers; real compiled factory produces `[16,64]` and contains separate resampler/LM gates |
| Public Flamingo generation with different images and four LM layers | Passed the actual image-context sensitivity control | Still passes: real language-gate context changes by more than `1e-6`; generated text need not be nonempty because immediate EOS is valid |
| BLIP image 31, patch 8 | Existing source explicitly supports floor cropping | Actual native path produces `[9,8]` patch features and a finite 8-wide image embedding; no blanket divisibility rejection was added |

The generator regression compiles the **emitted factory body** against the actual
AiDotNet library and invokes it. Minimal symbols are used only to isolate one
model during generator discovery, never as executable model dependencies.
The corrected original-generator control compiled and executed successfully and
then failed its assertions in `pr2130-generator-baseline-confirmed.trx` (2 failed,
0 passed). Earlier harness-only missing-import/accessor mistakes are excluded
from production defect evidence.

The Flamingo minimum of four language layers is an explicit tightening:
previously, sub-four stacks constructed without any multimodal language gate.
Two handwritten integration fixtures and the generator were migrated to four;
no generated model test file was edited. Native GPU dispatch is unchanged.

Validation uses typed capability flags. Only consumed dimensions are required;
patch-based models validate a positive patch before division and require at least
one full patch. Exact tiling remains required only by the existing BLIP-2 and
VisionMamba contracts. Audio-visual event localization preserves its legitimate
zero optional-encoder depth and fixed eight-head geometry.

## Executed checks

| Check | Result |
| --- | --- |
| Original native actual-library harness, net10 | 69 failed / 23 passed / 0 skipped |
| Corrected original-generator actual-factory control, net10 | 2 failed / 0 passed / 0 skipped |
| Final actual-library focused suite, net10 | **525 passed / 0 failed / 0 skipped**, 11 seconds |
| Independent parent replay of the same frozen binaries, net10 | **525 passed / 0 failed / 0 skipped**, 10 seconds |
| Full main test-project build, net10 | **0 errors**, 6,819 warnings, 7m01s |
| Actual main assembly, net10, retained recovery replay | **525 passed / 0 failed / 0 skipped**, 8 seconds |
| Full main compatibility rebuild, net8 + net471 | **0 errors**, 13,445 warnings, 11m45s |
| Actual main assembly, net8, retained recovery replay | **525 passed / 0 failed / 0 skipped**, 10 seconds |
| Actual main assembly, net471, retained recovery replay | **525 passed / 0 failed / 0 skipped**, 21 seconds |
| Real-source scalar suite, net10 | **350 passed / 0 failed / 0 skipped** |
| Real-source scalar suite, net8 | **350 passed / 0 failed / 0 skipped** |
| Real-source scalar suite, net471 | **350 passed / 0 failed / 0 skipped** |

The 525-case actual-library suite contains 104 native option cases, 13 actual
native construction/execution cases, 2 generator cases, 16 existing VisionMamba
cases, 46 ratchet/runtime cases, 342 sequence option cases, and 2 shared
documentation cases. VisionMamba width and depth are changed **independently**;
the width-only comparison retains the layer count while increasing actual
materialized parameters. The image model retains the supplied options object.

The main-assembly matrix is **1,575 passing executions of the same 525 checks**,
not 1,575 distinct tests. The scalar/focused runs overlap this matrix and are not
added to it as distinct coverage. All three recovery TRXs are nonempty XML with
525 total/passed cases and zero failed or skipped cases.

The #2128 exhaustive guards initially reported the new eighteenth subtype. Both
now use an explicit union: **17 language consumers + 1 image state-space
consumer**. No subtype is excluded, and all 17 language copy/topology controls
remain. The measured ratchet remains **861 gaps across 285 types, scanning 1005
concrete models**; its baseline was not raised or relaxed.

Final focused actual-library build: 0 errors, 2775 warnings. The untouched baseline
build had 0 errors and 2842 warnings. Compilation and executed tests are separate
claims. An initial no-reference-build attempt on this fresh worktree
reported missing auxiliary reference DLLs; it is build setup, not a source
regression, and is superseded by the normal dependency-building command.

## Commands and local evidence

From this worktree, with a serialized runner and Workstation GC:

```powershell
$ErrorActionPreference = 'Stop'
dotnet build tests/AiDotNet.VisionLanguageOptionsReview/AiDotNet.VisionLanguageOptionsReview.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false
if ($LASTEXITCODE -ne 0) { throw 'Build failed; do not execute an older test binary.' }
pwsh -NoProfile -File ./.github/scripts/harden-xunit-runner.ps1 -RunnerJson tests/AiDotNet.VisionLanguageOptionsReview/bin/Release/net10.0/xunit.runner.json
if ($LASTEXITCODE -ne 0) { throw 'Runner hardening failed; do not run tests.' }
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
dotnet vstest tests/AiDotNet.VisionLanguageOptionsReview/bin/Release/net10.0/AiDotNetTests.dll '--Logger:trx;LogFileName=pr2130-final-net10.trx' '--ResultsDirectory:artifacts/pr2130-review/final-net10'
if ($LASTEXITCODE -ne 0) { throw 'Actual-library regression tests failed.' }
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false '--logger:trx;LogFilePrefix=pr2130-scalar-final' --results-directory artifacts/pr2130-review/scalar-final
if ($LASTEXITCODE -ne 0) { throw 'Scalar contract tests failed.' }
```

The actual main test assembly uses the same seven fixture classes. These commands
build project dependencies and reject build/hardening/test failures before using
an older assembly. The compatibility command deliberately rebuilds: interrupted
disk-full outputs are not accepted as an up-to-date validation result.

```powershell
$ErrorActionPreference = 'Stop'
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'Main net10 build failed; do not test stale output.' }
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -t:Rebuild -p:CompatBuildOnly=true -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'Compatibility rebuild failed; do not test stale output.' }

$reviewClasses = @(
    'AiDotNet.Tests.Generators.GeneratedVisionLanguageFixtureContractTests',
    'AiDotNet.Tests.IntegrationTests.Configuration.OptionsSurfaceRatchetTests',
    'AiDotNet.Tests.UnitTests.Models.Options.SequenceModelOptionsContractTests',
    'AiDotNet.Tests.UnitTests.Models.Options.SharedOptionsDocumentationContractTests',
    'AiDotNet.Tests.UnitTests.Models.Options.VisionLanguageNativeOptionsTests',
    'AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM.VisionMambaModelTests',
    'AiDotNet.Tests.UnitTests.NeuralNetworks.VisionLanguageNativeConstructionTests'
)
$reviewFilter = ($reviewClasses | ForEach-Object { 'FullyQualifiedName~{0}.' -f $_ }) -join '|'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
foreach ($reviewTarget in 'net10.0', 'net8.0', 'net471') {
    $reviewBin = "tests/AiDotNet.Tests/bin/Release/$reviewTarget"
    $reviewResults = "artifacts/pr2130-review/main-$reviewTarget-reproduction"
    if (Test-Path -LiteralPath $reviewResults) { throw 'Choose a fresh results directory; preserve earlier proof.' }
    pwsh -NoProfile -File ./.github/scripts/harden-xunit-runner.ps1 -RunnerJson "$reviewBin/xunit.runner.json"
    if ($LASTEXITCODE -ne 0) { throw "Runner hardening failed for $reviewTarget." }
    dotnet vstest "$reviewBin/AiDotNetTests.dll" "--TestCaseFilter:$reviewFilter" '--Logger:trx;LogFileName=pr2130-main-reproduction.trx' "--ResultsDirectory:$reviewResults"
    if ($LASTEXITCODE -ne 0) { throw "Focused tests failed for $reviewTarget." }
    $reviewTrxPath = "$reviewResults/pr2130-main-reproduction.trx"
    if ((Get-Item -LiteralPath $reviewTrxPath).Length -eq 0) { throw 'Empty TRX; result not retained.' }
    $reviewTrx = [xml](Get-Content -LiteralPath $reviewTrxPath -Raw)
    if ([int]$reviewTrx.TestRun.ResultSummary.Counters.passed -ne 525 -or
        [int]$reviewTrx.TestRun.ResultSummary.Counters.total -ne 525) {
        throw "Expected all 525 reviewed checks to pass on $reviewTarget."
    }
    Get-FileHash -Algorithm SHA256 -LiteralPath "$reviewBin/AiDotNet.dll", "$reviewBin/AiDotNetTests.dll", "$reviewBin/AiDotNet.Generators.dll"
}
```

Use fresh results directories for a new proof run. Hardening is invoked through a
separate `pwsh` process so its exit code is defined; the test-process CPU and GC
variables are set explicitly in the parent shell. Direct script invocation had
correctly hardened the runner but left an unset native exit status in a fresh
shell, causing one recovery command to stop before testing. This was a command
guard error, not a model failure.

The original 92-case harness is retained at `2a5eb03731`. Its project supports
`-p:ReviewProductionProject=<untouched-597ae5849c-worktree>/src/AiDotNet.csproj`,
which was used for the failure-first build, without replacing production
dependencies. The current generator-only control also supports
`-p:ReviewGeneratorOnly=true` and
`-p:ReviewGeneratorProject=<untouched-597ae5849c-worktree>/src/AiDotNet.Generators/AiDotNet.Generators.csproj`.
Both overrides must reference the same original checkout for that control;
mixing the old generator with renamed new option APIs is not valid defect proof.

Local evidence is preserved under `artifacts/pr2130-review/`:

- `baseline/pr2130-native-baseline.trx` and `baseline-build.log`.
- `baseline-generator/pr2130-generator-baseline-confirmed.trx`.
- `final-net10/pr2130-final-net10.trx` and `final-net10-build.log`.
- `root-final-net10/pr2130-root-independent-525.trx`.
- `scalar-final/pr2130-scalar-final_<TFM>_*.trx`.
- `main-net10-build-with-dependencies.log`.
- `main-net10-recovered/pr2130-main-net10-recovered.trx` and
  `main-net10-recovered-test.log`.
- `main-compat-recovered-build.log`.
- `main-net8.0-recovered/pr2130-main-net8.0-recovered.trx` and
  `main-net8.0-recovered-test.log`.
- `main-net471-recovered/pr2130-main-net471-recovered.trx` and
  `main-net471-recovered-test.log`.

The initial main net10 replay printed 525 passing tests but left a **zero-byte TRX**
when the local disk filled. The first compatibility build was then canceled; its
outputs are **not** counted as completed validation. After disk recovery, net10
was replayed into the nonempty recovery TRX above and the compatibility targets
were rebuilt from scratch. The empty/canceled records are retained, not relabeled
as passing evidence. No native dependency directory was manually copied for this
recovery. Test durations in the tables are VSTest durations, not build or complete
discovery wall times.

SHA-256 of the frozen net10 binaries used by both final 525-case replays:

| Binary | SHA-256 |
| --- | --- |
| AiDotNet.dll | `8153AB88BBE30CF9473FCCED2A4CEB871EC14981F61CBDF33336A690A509B1A8` |
| Focused AiDotNetTests.dll | `DC678563AF3DAF3B7A61764D8FB1C166907C8FD4A81B661E425D70365D5B3EE8` |
| AiDotNet.Generators.dll | `2774A57356BF2D0A1FD5A9E25D12C1001544DC689208CBBDBA0876398BE79717` |
| AiDotNet.Tensors.dll | `92D30417AB66F523894833A8C4C18936A00BC8ADC4B26DB793D5FB2245341286` |

The normal main-project dependency build generated new assembly metadata from
the committed checkpoint, so its hashes differ from the earlier focused-runner
binaries. Production/test source remained frozen at `0b1932c294`; do not mix the
two sets when reproducing a run.

SHA-256 of the actual main-assembly recovery binaries:

| Target | Binary | SHA-256 |
| --- | --- | --- |
| net10.0 | AiDotNet.dll | `66115508954B2301BE04075CEE249585A70CB32F5C989DA676B28F879E28084F` |
| net10.0 | AiDotNetTests.dll | `6EF51AF348C4D323EE75D451E4F8F58402547CB89E56BDC0597B2726F77F8FBA` |
| net10.0 | AiDotNet.Tensors.dll | `92D30417AB66F523894833A8C4C18936A00BC8ADC4B26DB793D5FB2245341286` |
| net8.0 | AiDotNet.dll | `B5ADAE98AAF3772E31A25E8C69CBADE91C9D0EB88AB8745D645648C7C43E85DD` |
| net8.0 | AiDotNetTests.dll | `A5B1BE85D0C1E67D1CB857833BCF72099BD55F697D0191A70D27281E8E4A7161` |
| net8.0 | AiDotNet.Tensors.dll | `A7DA6EC6FC246020CDDDF7FAAEC9323521A798557CAD85DA2C3C148CD72B8A93` |
| net471 | AiDotNet.dll | `04FC1B84903694DA3566879FAA92F4826D869866AE25DD68CC4E65EAF803D976` |
| net471 | AiDotNetTests.dll | `7232BBB05D29B818A4E872EB7D5302F40DC46B6BCF6C29AC9A229E28ECB11D4B` |
| net471 | AiDotNet.Tensors.dll | `B34365B7CE9AF877D43A2CB9B2CA24D53464B0C6CBC6E8997D009320D9419BE2` |

The main-project `AiDotNet.Generators.dll` is identical on all three targets:
`D5376674468CD7E6BACFAC570B700C6638BDDA61290316F26A33443D8C655A00`.

The untouched baseline AiDotNet.dll hash is
`03D23E34BCB1546324B6392C73E90C3E2F20EEDE2D2EDA226967023B88FBCD10`;
its original 92-case test DLL is
`C7B6C0D2ACC14D95EA494AB9101463F9F6F9A169095F4DC8D4CCE02CED6D7930`.
The corrected generator-only baseline test DLL is
`C55D595FC3724FE4E11B4A27CEDA7D9A339C3DD6DE57DCB625B7D3BC9ED103B6`;
the original generator DLL is
`2E365E489B969EF3D919E7FD379CEF25827D00C8C6ED19D35A322EC0D4C4CEBF`.

## Review disposition and remaining scope

The exhaustive initial inventory contained 42 threads, 40 unresolved. This
document does not automatically resolve any thread.

The eight native findings below are fully addressed by the bounded implementation
and retained controls (including corrected review premises where indicated):

| Comment | Thread | Disposition |
| --- | --- | --- |
| `3964560807` | `PRRT_kwDOKSXUF86ggA20` | Shared actionable parameter names, with isolated invalid-property cases |
| `3964560821` | `PRRT_kwDOKSXUF86ggA3C` | BLIP floor cropping is valid; actual nondivisible geometry is covered |
| `3964560831` | `PRRT_kwDOKSXUF86ggA3I` | Native patch configuration and minimum multimodal gate count are enforced |
| `3964560937` | `PRRT_kwDOKSXUF86ggA4g` | Typed consumed-dimension/patch validation; four native construction defects covered |
| `3964560945` | `PRRT_kwDOKSXUF86ggA4k` | Finite positive Flamingo rate; integrated #2128 covers XLSTM |
| `3968055419` | `PRRT_kwDOKSXUF86go0tu` | Repeated constructor default assignments removed |
| `3968055433` | `PRRT_kwDOKSXUF86go0t6` | VisionMamba image/SSM contract replaces irrelevant text requirements |
| `3968055473` | `PRRT_kwDOKSXUF86go0uY` | Class-count rejection is isolated from unrelated option validation |

- Native fixes address comments `3964560807`, `3964560937`, `3968055433`,
  `3968055473`, the Flamingo portion of `3964560945`, and duplicate defaults in
  `3968055419`. #2128 supplies the XLSTM portion of `3964560945`.
- `3964560821` and `3964560831`: the claimed universal exact-tiling requirement
  was corrected against the actual patch-layer contract; floor-crop controls
  remain. The real zero-language-gate and ignored native patch issues are fixed.
- `3964560922` is only partially addressed: native configurable patches are
  proved, but ONNX constructor configuration remains part of the deferred work.
- The shared #2128 fixes are integrated, including its copy/sequence validation,
  generator cleanup, diagnostics example, documentation and physical ratchet
  evidence. Its proof is in [PR2128_REVIEW_PROOF.md](PR2128_REVIEW_PROOF.md).
  Imported completed findings are `3964555367`, `3964555371`, `3964555379`,
  `3964555383`, `3964555388`, `3964555414`, `3964555418`, `3964555427`,
  `3964555438`, `3964555448`, `3964555451`, `3964555457`, `3964555476`,
  `3964555483`, `3964555504` and `3964560954`. These are not sixteen new native
  fixes in this batch.
- `3964555490` and `3964555505`: the skipped placeholder is gone, the real
  configuration guards cover the explicit 18-type cohort, and concrete-options
  discovery/report controls pass. The broader phase-3/ONNX behavior guarantee
  is still incomplete; those portions must not be resolved on ratchet counts.
- `3964555435`: the requested public Eagle rename was rejected by the #2128
  review because both names predate the migration; see its merge-base evidence.
  No public type was removed merely to silence that compatibility finding.
- Still open: ONNX configuration/API semantics (`3964555485`, `3964560801`,
  `3964560839`, `3964560900`); inert inherited CLIP options (`3964560909`);
  GPT-4 vision-width/name semantics (`3964560930`); ignored correspondence
  width/depth; phase-3 copy/configuration ownership; remaining public-property
  and constructor XML documentation (`3964555465`, `3964560814`, `3964560846`);
  and the previously recorded compatibility/gradient-norm policy questions.
- In particular, `3964560779` still needs the gradient-limit canonicalization
  decision even though inherited copying is now covered; `3964560916` still
  needs an explicit public-validation API disposition. `3964555442` concerns
  the separate GAN validation surface and is not addressed by native VLM tests.
- The public `Validate()` methods are also called by ONNX constructors, so the
  new geometry/rate/depth checks apply there too. These tests do not establish
  that exported graph geometry matches the options. That boundary must be
  resolved before claiming the ONNX paths ready.

The source-focused token-optimization workflow helped keep the review on the
actual consumer contracts; it did not replace runtime proof. Independent review
also tightened the separate width/depth oracle and explicit Framework test
initialization. Added C# lines contain no null-forgiving operators; closed
validation and fixture policies use typed flags/enums.
