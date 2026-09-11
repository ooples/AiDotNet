# PR #2130: GPT4Vision ONNX contract proof

## Correction

GPT4Vision uses the shared immutable graph signatures and runtime tensor reader.
The image encoder produces `[1, tokens, VisionDim]`; the text encoder produces
`[1, tokens, EmbeddingDimension]`. Those widths need not be equal for the separate
embedding APIs. The reader retains every token for the existing mean pooling,
instead of treating token features as a single pooled vector.

Construction rejects incompatible fixed input/output shapes, wrong element types,
unsupported required inputs, extra output batches and empty token sequences.
Symbolic dimensions are allowed and checked against the actual executed output.
Image inputs are no longer silently cropped/padded. Text inputs must fit the
configured context and any concrete graph context; short fixed-context inputs are
rejected, not padded with invented token/mask semantics.

Nondefault native-only topology options now fail explicitly. ONNX input geometry
is validated separately from native patch geometry. Session ownership is committed
only after both graphs and all contracts validate; session options and partially
created sessions are disposed on failure. Effective graph metadata replaces guessed
native tower depth/vocabulary claims. Production provider selection is unchanged.

## Actual before and after

The identical expanded **24-case** test source was compiled against the frozen old
actual library and then the corrected actual library:

| Actual net10 CPU runtime | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Before | 0 | 24 | 0 |
| After | 24 | 0 | 0 |

All fixtures are local, data-dependent ONNX graphs executed by ONNX Runtime, not
session mocks or downloaded pretrained models. The two existing valid-graph cases
already produced correct embedding numbers before; they failed the newly required
effective-configuration assertion. They are not evidence that numerical pooling was
previously broken. Similarly, the short static-context case already failed inside
ONNX Runtime; it now fails at the wrapper's explicit validated input boundary.

The other negative controls reproduced ignored options, missing construction checks,
silent input alteration, unchecked dynamic widths/batches and native-only patch
restrictions leaking into ONNX geometry. Output checks examine all feature values
and preserve separate image/text widths. Mutating requested options after creation
does not change the effective graph snapshot or the already-configured input path.

Baseline core SHA-256:
`A98360AF0A33125FD793BAC3E3D5ACA749E4BDE64F9C97888F8A66659B981203`.
Baseline test assembly:
`BC8DCCD0169CD365595247F48E3FA780768DF0B6355EAC0D41DEA186755344D1`.
Corrected core:
`D558A12AA80DCF019EED941EA25F9077EA5981EC4879932D71742137A0843C89`.
Corrected test assembly:
`1743ECED7D355D36AAADC3E0D35D590B01E57921FAD71856FA5661C9E1DFDFF9`.
Actual production plus focused runner build: zero errors, 2,842 warnings, 5m09s.
The final 24 tests completed in two seconds.

The existing CLIP/VideoCLIP ONNX cohort then passed **33/33**, and the complete
native/options/generator cohort passed **693/693**, against the same corrected
core, both with zero failures/skips. Their test-only builds each had zero warnings
and errors. Reports: `gpt4-regression-AiDotNet.OnnxOptionsReview.trx` and
`gpt4-regression-AiDotNet.VisionLanguageOptionsReview.trx` in the same results folder.
These regressions are reproduced with the commands in
[the earlier options proof](PR2130_ONNX_OPTIONS_REVIEW_PROOF.md).

Reports under `artifacts/pr2130-onnx-contracts/results/`:
`gpt4-onnx-expanded-before.trx` and `gpt4-onnx-expanded-after.trx`.
The earlier 17-case baseline is retained separately and not added to these counts.

## Reproduce the final check

Build the actual project and focused runner into a fresh directory. Native asset
copying below is normal SDK behavior. The recorded local build instead disabled
the large transitive native closure and copied only verified existing ONNX Runtime
CPU binaries, then used the existing verified runner directory on process PATH.

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$proofRoot = 'artifacts/pr2130-gpt4-replay-' + [Guid]::NewGuid().ToString('N')
dotnet build tests/AiDotNet.Gpt4OnnxReview/AiDotNet.Gpt4OnnxReview.csproj `
    -c Release -f net10.0 --artifacts-path $proofRoot -m:1 `
    -p:UseSharedCompilation=false -p:GeneratePackageOnBuild=false -v:quiet
if ($LASTEXITCODE -ne 0) { throw 'Actual library/runner build failed' }
$runner = "$proofRoot/bin/AiDotNet.Gpt4OnnxReview/release_net10.0"
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 -RunnerJson "$runner/xunit.runner.json"
if ($LASTEXITCODE -ne 0) { throw 'Runner hardening failed' }
dotnet vstest "$runner/AiDotNetTests.dll" '--Logger:trx;LogFileName=gpt4-after.trx' `
    "--ResultsDirectory:$proofRoot/results"
if ($LASTEXITCODE -ne 0) { throw 'GPT4 ONNX contracts failed' }
[xml]$report = Get-Content "$proofRoot/results/gpt4-after.trx" -Raw
$counts = $report.TestRun.ResultSummary.Counters
if ([int]$counts.passed -ne 24 -or [int]$counts.failed -ne 0 -or [int]$counts.total -ne 24) {
    throw 'Unexpected test census'
}
```

## Limits

This proves the two ONNX feature-encoder boundaries, not learned language generation
or a learned cross-modal projector from this encoder-only API. It does not establish
GPU throughput, physical GPU execution, other ONNX wrappers or a passing full CI
matrix. Remaining BLIP/BLIP2/Flamingo/LLaVA/ImageBind and Finch findings stay open.
