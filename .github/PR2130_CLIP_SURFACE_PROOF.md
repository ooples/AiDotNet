# PR #2130: inference-only input options and canonical language-depth name

Baseline: `688641a3fb95928891b6596c50833d816d884470`.
Reviews: [CLIP no-op native setters](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560909)
and the second request in [GPT4 naming](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560930).

## What changes, and what does not

`ClipNeuralNetwork` is ONNX-only and consumes three dimensions:
`EmbeddingDimension`, `MaxSequenceLength`, and `ImageSize`. Its image path
accepts RGB input. Inheriting native tower setters exposed eight silent no-ops,
including `Channels`, even though changing them did not change execution.

The shared `VisionLanguageInputOptions` base now owns those three input/output
dimensions and their validation. `ClipOptions` uses that base.
`VisionLanguageModelOptions` inherits it and retains **all** existing native
tower properties, channel validation, patch cropping/tiling checks, and the
typed validation-requirement flags. Other model options keep their public
properties, values, validation order, and `NeuralNetworkOptions` ancestry.
No layer implementation, ONNX session code, GPU policy, or generated leaf test
changes in this batch.

The same review requested one family name for language depth. GPT4 now uses
`NumLmLayers`, matching Flamingo and LLaVA, with its unchanged default of 32.
The native constructor and all four existing test references use that name;
there is no second alias or ignored setter. The earlier width fix remains:
`VisionDim` is the single native vision width, with default 1024.

Compatibility was checked against current `origin/master`
`3185b41f1e1cffb81d76130e319233153c984471`: both `ClipOptions` and
`Gpt4VisionOptions` were empty subclasses of `NeuralNetworkOptions` there.
The removed native CLIP setters, its native-family base relationship, and the
GPT4 names being corrected were introduced by this unmerged migration.
Callers of that unmerged API should use only the three supported CLIP dimensions
and replace GPT4 `NumLanguageLayers` with `NumLmLayers`.

## Failure-first and regression evidence

| Actual executed scope | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| New CLIP surface/default/invalid-value cases before the production edit, net10.0 | 9 | 9 | 0 |
| New GPT4 family-depth naming case before its rename, net10.0 | 0 | 1 | 0 |
| Entire scalar suite after both edits, net10.0 | 421 | 0 | 0 |
| Entire scalar suite after both edits, net8.0 | 421 | 0 | 0 |
| Entire scalar suite after both edits, net471 | 421 | 0 | 0 |
| Fresh actual-library native/options/generator cohort, net10.0 | 553 | 0 | 0 |

The scalar total is the preceding 402 plus 18 CLIP cases and one naming case.
The actual-library total is the preceding 534 plus the same 19 cases. Existing
native constructor, generated-fixture, shared validation, documentation, and
GPT4 real patch-tensor/parameter-count controls remain green.

The CLIP before-run used a temporary source-linked project containing the real
options and complete base chain, without replacement production types:
`%TEMP%/pr2130-clip-review-20260911/results/clip-surface-before.trx`.
It failed because the eight native setters were present and the public input
surface was not restricted to the three consumed dimensions. The nine default,
positive-input and invalid-input controls already passed.
The added naming test failed because `NumLmLayers` did not exist.

Final TRXs are in `artifacts/pr2130-clip-surface/results/`:

- `gpt4-depth-name-before.trx`
- `clip-depth-final-net10.0.trx`, `clip-depth-final-net8.0.trx`, `clip-depth-final-net471.trx`
- `clip-depth-native-final.trx`

The fresh actual-library build completed in 5m36s with **0 errors and 2842
warnings**; test execution took 15 seconds. Build log:
`artifacts/pr2130-clip-surface/native-final-build.log`.
The previous frozen GPT4 actual-library output was also independently rerun:
534 passed, zero failures/skips, and its library hash remained unchanged.

### Executed binary identities

Scalar DLLs under `artifacts/pr2130-clip-surface/scalar-build/bin/AiDotNet.OptionsContractTests/release_<tfm>/`:

| TFM | SHA-256 |
| --- | --- |
| net10.0 | `68466F75F37B0F15F21C3AA73A54DC38FC1AE0BE174AFCEBE303263C33B83AF1` |
| net8.0 | `7156B831BB524B7E80ECD65E6DB99DA83AA768DE165229F771AC1C13BD950F18` |
| net471 | `89699D62A17A808721BBE24D4CEE19EE266F64B40B3AA908C3CF1F632A11B2AE` |

Actual-library cohort under `artifacts/pr2130-clip-surface/native-final/bin/AiDotNet.VisionLanguageOptionsReview/release_net10.0/`:

| DLL | SHA-256 |
| --- | --- |
| AiDotNet.dll | `055589B5572E2F30B1750F79942B9670BCFB98E2229ECF4B4F4C9696A8F445D2` |
| AiDotNetTests.dll | `653B6A3E64DD572C3394ABAB3147FB6F1867653A3DFD33B71F334759715EC6A8` |
| AiDotNet.Generators.dll | `48CFD3109EA8AB159C860786D79CD06EBBB3E9BC441D6DB0E894A4A22626AD9F` |

## Reproduce

```powershell
foreach ($clipTfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj `
        -c Release -f $clipTfm --artifacts-path artifacts/pr2130-clip-surface/scalar-build `
        -m:1 -p:UseSharedCompilation=false `
        --logger "trx;LogFileName=clip-depth-final-$clipTfm.trx" `
        --results-directory artifacts/pr2130-clip-surface/results -v:quiet
    if ($LASTEXITCODE -ne 0) { throw "CLIP/depth scalar suite failed on $clipTfm" }
}
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
dotnet build tests/AiDotNet.VisionLanguageOptionsReview/AiDotNet.VisionLanguageOptionsReview.csproj `
    -c Release -f net10.0 --artifacts-path artifacts/pr2130-clip-surface/native-final `
    -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false `
    -p:GeneratePackageOnBuild=false -p:CopyLocalRuntimeTargetAssets=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'CLIP actual-library build failed.' }
$clipNativeBin = 'artifacts/pr2130-clip-surface/native-final/bin/AiDotNet.VisionLanguageOptionsReview/release_net10.0'
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 -RunnerJson "$clipNativeBin/xunit.runner.json"
if ($LASTEXITCODE -ne 0) { throw 'CLIP runner hardening failed.' }
dotnet vstest "$clipNativeBin/AiDotNetTests.dll" `
    '--Logger:trx;LogFileName=clip-depth-native-final.trx' `
    '--ResultsDirectory:artifacts/pr2130-clip-surface/results'
if ($LASTEXITCODE -ne 0) { throw 'CLIP native cohort failed.' }
```

## Adversarial review and limits

Independent review checked preserved native validation ordering/flags,
inheritance compatibility, the removed no-op channel setter, and that the
tests assert actual public API shape rather than source-string patterns.
The frozen previous cohort was executed again, not relabeled as a new build.

This proves the options/API correction and the executed native regression
cohort. It does **not** prove that arbitrary ONNX graphs match requested input
or output dimensions, and it does not resize loaded graphs. The separate
effective-ONNX configuration threads remain open. No full repository matrix,
real GPU performance, ONNX runtime execution, or detector/audio-model fix is
claimed by this batch. The CLI-only runtime-asset copy setting does not change
the shipped package or users' GPU support.
