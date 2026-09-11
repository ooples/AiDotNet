# PR #2130: BLIP ONNX options and tensor contracts

The ONNX constructor now rejects eight nondefault native-only options instead of
silently ignoring them. The host image/context/embedding settings are checked
against real graph signatures, separately from native patch geometry. Effective
metadata contains immutable graph signatures rather than guessed layer counts or
vocabulary size. Failed construction releases every opened session.

Runtime checks reject image batches that would previously lose all but the first
example, incompatible host image geometry, and symbolic output widths that do not
match the configured embedding. Existing normalized first-token pooling is retained.

## Actual before and after

The identical expanded 27-case source was tested with real, data-dependent local
ONNX graphs against the actual production library:

| net10 CPU runtime | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Before | 1 | 26 | 0 |
| After | 27 | 0 | 0 |

Baseline core SHA-256:
`D558A12AA80DCF019EED941EA25F9077EA5981EC4879932D71742137A0843C89`.
Baseline test assembly:
`A5442E0BB758E649DD63091BE4C3FD21BF2BC8351CF660A880C403835B582B9B`.
Corrected core:
`BC40C3928D37C803726ED4D1297069D062E6013C0B40408BFB435C4C46C26BEC`.
Corrected test assembly:
`19B4CBE7B2DFC190367322913B97405A4C53AE4B5662AED1A23CADCFE13BEF0B`.

The four positive encoder cases already passed their numerical assertions before;
they then failed the newly required configuration assertion. The single-image batch
control already passed before. These are not claims that the existing pooling
arithmetic was broken. The channel mismatch previously reached ONNX Runtime; the
new host check rejects it explicitly. Symbolic height/width mismatches were accepted.

Actual corrected core build: zero errors, 2,845 warnings, 5m19s. The 27 checks passed
in four seconds. Against the same corrected core, existing CLIP/VideoCLIP checks
passed **33/33**, GPT4Vision checks **24/24**, and native/options/generator checks
**693/693**, all without skips. The first native replay omitted the generated XML
documentation artifact and failed five documentation checks; providing the actual
corrected XML artifact yielded the final 693/693 without any source adjustment.

Reports in `artifacts/pr2130-onnx-contracts/results/`:
`blip-onnx-expanded-before.trx`, `blip-onnx-expanded-after.trx`,
`blip-regression-AiDotNet.OnnxOptionsReview.trx`,
`blip-regression-AiDotNet.Gpt4OnnxReview.trx`, and `blip-native-options-final.trx`.
The original 21-case baseline is retained separately, not added to these counts.

## Reproduce

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$proofRoot = 'artifacts/blip-replay-' + [Guid]::NewGuid().ToString('N')
dotnet build tests/AiDotNet.CompositeOnnxReview/AiDotNet.CompositeOnnxReview.csproj `
    -c Release -f net10.0 --artifacts-path $proofRoot -m:1 `
    -p:UseSharedCompilation=false -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'Actual library/runner build failed' }
$runner = "$proofRoot/bin/AiDotNet.CompositeOnnxReview/release_net10.0"
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 -RunnerJson "$runner/xunit.runner.json"
if ($LASTEXITCODE -ne 0) { throw 'Runner hardening failed' }
dotnet vstest "$runner/AiDotNetTests.dll" '--Logger:trx;LogFileName=blip-after.trx' `
    "--ResultsDirectory:$proofRoot/results"
if ($LASTEXITCODE -ne 0) { throw 'BLIP contracts failed' }
```

The recorded constrained-disk local run disabled `CopyLocalRuntimeTargetAssets`
and reused verified existing Windows CPU native dependencies. Production provider
selection is unchanged; CPU results are not GPU performance evidence.

The decoder's signature is exposed because its session is loaded. The existing
caption path does not execute that decoder; these checks do not establish pretrained
captioning, VQA, or decoder quality. The grouped BLIP/BLIP2/Flamingo review finding
remains open until the other two constructors are addressed.
