# PR #2130: canonical GPT-4-style native vision width

Baseline: `a9577751bd0f2cb2502891716eb3f7c0ce19205f`.
Review: [3964560930](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560930),
thread `PRRT_kwDOKSXUF86ggA4a`.

## Compatibility and implementation

`Gpt4VisionOptions` already inherits `VisionDim`, the common native vision-tower
width. The migration added a duplicate `VisionEmbeddingDim`, initialized only
that duplicate, and made both constructors ignore `VisionDim`. The native
factory passed the duplicate value directly to `PatchEmbeddingLayer`.

The correction makes `VisionDim` the sole public option for that width, retains
the pre-migration constructor default of 1024, and uses the shared positive-value
guard before constructing layers. Source at
`671836a347436f4b023ebe3f02a86ea333ad0686` shows `visionEmbeddingDim = 1024` on
both former scalar constructors. `EmbeddingDimension`, `HiddenDim`,
`VisionLayers`, and `NumLanguageLayers` retain their defaults and names.

Callers of this migration-only API should replace
`new Gpt4VisionOptions { VisionEmbeddingDim = 64 }` with
`new Gpt4VisionOptions { VisionDim = 64 }`. The two existing handwritten
integration callers receive exactly that identifier migration. No generated
leaf test is changed. There is no GPT4-specific generator initializer using
the old property; unrelated CLIP/SigLIP option types that legitimately expose
`VisionEmbeddingDim` are untouched.

Both GPT4 constructors read the same canonical option, but this does **not**
establish that ONNX graph dimensions match it. The effective loaded-graph
configuration review remains open; this change neither resizes a graph nor
adds a metadata-only substitute for graph validation. GPU dispatch, layer
algorithms, and session creation are unchanged.

## Failure-first evidence

The seven scalar controls at the old implementation produced **5 failures,
2 passes, 0 skips**: the common default was zero, the duplicate API was present,
and all three invalid common widths were accepted. The two positive common
property controls already passed.

The actual-native harness was compiled with reference rebuilding disabled,
against the frozen real library SHA-256
`66115508954B2301BE04075CEE249585A70CB32F5C989DA676B28F879E28084F`.
Both new runtime controls failed for the intended reason:

| Requested `VisionDim` | Actual old patch tensor | Required patch tensor |
| ---: | --- | --- |
| 8 | `[4, 1024]` | `[4, 8]` |
| 16 | `[4, 1024]` | `[4, 16]` |

The tests construct the real native model with small language-side dimensions
and execute its actual `PatchEmbeddingLayer`. The final oracle also checks the
materialized projection weight/bias count, finite tensor contents, and options
identity. It is not an assertion about an echoed metadata value.

Native baseline harness compilation: **0 warnings, 0 errors**. Its library hash
matches both the preserved main test output and the preserved source output.
Baseline runtime used CPU initialization, Workstation GC, and the serialized
xUnit runner. No old main/native proof binary was overwritten.

Baseline TRXs under `artifacts/pr2130-gpt4-width/results/`:

- `gpt4-width-scalar-baseline.trx`: 2 passed / 5 failed / 0 skipped.
- `gpt4-width-native-baseline.trx`: 0 passed / 2 failed / 0 skipped.

## Final scalar evidence

All **402** scalar tests pass with zero failures/skips on each of `net10.0`,
`net8.0`, and `net471` (the preceding 395 plus seven new cases).
Files: `gpt4-width-scalar-final-<tfm>.trx` in the same results directory.

Final scalar DLLs are under
`artifacts/pr2130-gpt4-width/scalar-build/bin/AiDotNet.OptionsContractTests/release_<tfm>/`:

| TFM | `AiDotNet.OptionsContractTests.dll` SHA-256 |
| --- | --- |
| net10.0 | `404D93C819848EBC0D3AC48FBBCCCC1E0D84C002542CFD30B07F2B12A0C75F42` |
| net8.0 | `306615047A3811AD11DA9C774C1B9C7B29DF0F96867E5439E7CC52A55CF42B84` |
| net471 | `E014926199EB94E1C93F10ED3D74A238477928D08CC349C26C11281871879A45` |

## Final actual-native evidence

The corrected real library, generator, and focused harness build completed
with **0 errors / 2775 warnings** in 4m22s. The complete focused harness then
passed **534 / 534**, with **0 failures / 0 skips**, in 13s: the preceding 525
native/options/generator cases plus seven scalar and two native width controls.
Both requested widths now produce the required actual patch tensor and
projection parameter count. TRX: `gpt4-width-native-final.trx` in the same
results directory. Build/runtime logs are
`artifacts/pr2130-gpt4-width/native-final-build.log` and
`artifacts/pr2130-gpt4-width/native-final-tests.log`.

Executed DLLs under
`artifacts/pr2130-gpt4-width/native-final/bin/AiDotNet.VisionLanguageOptionsReview/release_net10.0/`:

| DLL | SHA-256 |
| --- | --- |
| AiDotNet.dll | `6798A4DBA4BC7C40C5B54F7B047601A5537B733BF0B5637CE0D8836891455097` |
| AiDotNetTests.dll | `D7953CBFB190EA2E3554D2AC225D095FD4F6192C8D21A4BB9F3D8ADD301E8E8D` |
| AiDotNet.Generators.dll | `7C01DC9ADEED0C4EA5FDB5216A1C55B453D1657419F349BA5D5A683DF71CEC1C` |
| AiDotNet.Tensors.dll | `92D30417AB66F523894833A8C4C18936A00BC8ADC4B26DB793D5FB2245341286` |

The preserved original main test/library DLLs were rehashed and are unchanged.
An earlier build attempt was deliberately interrupted to yield the shared
compiler slot; only the completed build and executed TRX above are evidence.
The SDK's `CopyLocalRuntimeTargetAssets=false` command-line setting avoids
copying every RID's native runtime assets into this CPU harness (262.0 MiB
versus the baseline's 1429.2 MiB) without changing production package policy.
The passing CPU cohort establishes the closure it needs; it does not establish
the native dependency closure for GPU or ONNX-session execution.

Limits: this batch executes the actual native library on net10.0 and the scalar
contracts on all three TFMs, not the whole main test assembly. The two
handwritten integration callers receive only the source identifier migration;
their full integration classes were not executed. The ONNX effective-graph
configuration review remains separate.

## Reproduction

The small source-linked scalar project is independent of native model builds:

```powershell
foreach ($gpt4ScalarTfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj `
        -c Release -f $gpt4ScalarTfm `
        --artifacts-path artifacts/pr2130-gpt4-width/scalar-build `
        -m:1 -p:UseSharedCompilation=false `
        --logger "trx;LogFileName=gpt4-width-scalar-final-$gpt4ScalarTfm.trx" `
        --results-directory artifacts/pr2130-gpt4-width/results -v:quiet
    if ($LASTEXITCODE -ne 0) { throw "GPT4 scalar suite failed on $gpt4ScalarTfm." }
}
```

The baseline test-only build used `--artifacts-path
artifacts/pr2130-gpt4-width/native-baseline`, `-p:BuildProjectReferences=false`,
and `-p:ReviewReferencePropertiesToRemove=ArtifactsPath`. That explicit test-only
setting redirects the harness without redirecting the existing reference
outputs. It must not be passed to the final build: the final build must compile
the actual corrected dependency graph into its own redirected artifacts.

All native executions must set `AIDOTNET_FORCE_CPU=1`, `DOTNET_gcServer=0`, and
`COMPlus_gcServer=0`, serialize the copied xUnit runner with
`.github/scripts/harden-xunit-runner.ps1`, reject nonzero build/hardening exits,
and verify the executed TRX counters before claiming a result.

```powershell
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
dotnet build tests/AiDotNet.VisionLanguageOptionsReview/AiDotNet.VisionLanguageOptionsReview.csproj `
    -c Release -f net10.0 --artifacts-path artifacts/pr2130-gpt4-width/native-final `
    -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false `
    -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'GPT4 actual-library build failed.' }
$gpt4NativeBin = 'artifacts/pr2130-gpt4-width/native-final/bin/AiDotNet.VisionLanguageOptionsReview/release_net10.0'
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 `
    -RunnerJson "$gpt4NativeBin/xunit.runner.json"
if ($LASTEXITCODE -ne 0) { throw 'GPT4 runner hardening failed.' }
dotnet vstest "$gpt4NativeBin/AiDotNetTests.dll" `
    '--Logger:trx;LogFileName=gpt4-width-native-final.trx' `
    '--ResultsDirectory:artifacts/pr2130-gpt4-width/results'
if ($LASTEXITCODE -ne 0) { throw 'GPT4 actual-native review suite failed.' }
[xml]$gpt4NativeTrx = Get-Content `
    artifacts/pr2130-gpt4-width/results/gpt4-width-native-final.trx -Raw
$gpt4NativeCounters = $gpt4NativeTrx.TestRun.ResultSummary.Counters
if ($gpt4NativeCounters.total -ne '534' -or $gpt4NativeCounters.executed -ne '534' -or
    $gpt4NativeCounters.passed -ne '534' -or $gpt4NativeCounters.failed -ne '0' -or
    $gpt4NativeCounters.notExecuted -ne '0') {
    throw 'Unexpected GPT4 review inventory or counters.'
}
```
