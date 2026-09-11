# PR #2130: ONNX boundaries and options review proof

## What this batch establishes

CLIP and VideoCLIP now validate the graph inputs and consumed outputs against the
wrapper's actual tensor contract. Fixed incompatible geometry and element types
fail at construction; symbolic axes remain supported, with actual output width,
rank and batch checked at execution. Outputs are not silently truncated or padded.
Unused non-tensor outputs remain legal. Graph metadata is an immutable snapshot,
separate from mutable requested options. Session initialization transfers ownership
only after both sessions and their contracts have been accepted.

VideoCLIP rejects nondefault native-only options its two ONNX encoder graphs cannot
honor. Operative image channels, image size, frame count and text context still
control the wrapper inputs. Its metadata no longer invents native tower geometry.
This is not a claim that changing options can reconfigure a pretrained graph.

The shared options copy chain preserves inherited and declared properties, rejects
null sources, documents defaults and beginner effects, and keeps migration-added
validation plumbing assembly-only. Native validators reject consumed invalid
dimensions, rates and identifiers before large model allocations. The original
feature-only AVC `Channels=0` assertion remains unchanged: channel validation belongs
at the actual visual-input boundary. The shared AVC/state correction has its own
evidence in [PR2130_AVC_REVIEW_PROOF.md](PR2130_AVC_REVIEW_PROOF.md).

## Executed before and after

All actual-library results below use real project references, not source-linked
production model implementations. Small ONNX graphs are generated locally by the
test fixture and executed with ONNX Runtime; no model download or mocked session
is used. Successful cases assert data-dependent numeric outputs from both encoders.

| Cohort | Before | Final after |
| --- | --- | --- |
| Initial CLIP/VideoCLIP ONNX contracts | 2 passed, 10 failed | Expanded 33 passed, 0 failed, 0 skipped |
| Copy/documentation/visibility contracts | 0 passed, 14 failed on actual A983 core | All 14 included in actual 693-pass cohort |
| Native scalar validation contracts | 4 passed, 109 failed, source-linked options | All 113 included in actual 693-pass cohort |
| Document copy / sequence validator API boundaries | 0 passed, 6 failed on actual A983 core | All 6 included in actual 693-pass cohort |
| Original native/options/generator cohort | Intermediate 552 passed, 1 failed | Expanded 693 passed, 0 failed, 0 skipped |

The final 693 consist of the original 553 plus 14 copy/docs/visibility cases, 113
scalar validation cases, seven actual constructor rejection cases, and six API
boundary cases. The seven new constructor cases are after controls, not claimed as
executed before. Expanded ONNX coverage is likewise not represented as an unchanged
33-case baseline. The sequence generator's separate 27-pass/4-fail before and
31-pass/0-fail after comparison, real factory semantic compilation and exact
commands are in [SequenceFixtureReview](../tools/SequenceFixtureReview/README.md).

The complete source-linked scalar/options suite also passed **540 tests on each of
net10.0, net8.0 and net471**, with no failures or skips. Those runs do not establish
actual ONNX runtime compatibility on those other frameworks. The actual combined
library and ONNX executions reported here are **net10.0, CPU**, not GPU performance
measurements or a passing full CI matrix.

Final actual core SHA-256:
`E0865AFB66E8BA9894E9D3A12FBE238C48E053968B15C047F6A724B6CD984859`.
Its build completed with zero errors and 2,775 warnings.
Both focused test-only builds completed with zero errors and zero warnings.

Final actual test assembly SHA-256 values:

- Native/options: `ED011F6F408E3E0C4AF4CF7EF0627007BE707D6A0E896C851A258B9E2B7B0D72`.
- ONNX: `FBFB333060A686ADF8652BE79D42E192C2E9EA86449BCEF99B99435FE953557E`.

Local reports in `artifacts/pr2130-onnx-contracts/results/`:

- `onnx-contract-before.trx`: original 2/10 baseline.
- `vlm-copy-doc-visibility-before.trx`: original 0/14 actual baseline.
- `onnx-native-regression-first-pass.trx`: intermediate 552/1; the failing assertion was not relaxed.
- `combined-final-AiDotNet.VisionLanguageOptionsReview.trx`: final 693/0, 17 seconds.
- `combined-final-AiDotNet.OnnxOptionsReview.trx`: final 33/0, 3 seconds.

The A983 intermediate core used by the options baselines has SHA-256
`A98360AF0A33125FD793BAC3E3D5ACA749E4BDE64F9C97888F8A66659B981203`.
Scalar validation and sequence reports are retained separately under
`artifacts/pr2130-native-validation`, `artifacts/pr2130-avc/results`, and
`artifacts/pr2130-sequence-semantic/results`; they are not added again to the 693.

## Reproduce the final runtime cohorts

From the repository root, use a fresh artifact directory. This builds the actual
production project before executing each focused runner. Standard native runtime
asset copying is retained here for portability; the recorded space-bounded local
run disabled copying the full transitive native closure and supplied the verified
ONNX Runtime CPU binaries from its already-built runner instead.

```powershell
$ErrorActionPreference = 'Stop'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
$proofRoot = 'artifacts/pr2130-review-replay-' + [Guid]::NewGuid().ToString('N')
$cohorts = @{
    'AiDotNet.VisionLanguageOptionsReview' = 693
    'AiDotNet.OnnxOptionsReview' = 33
}
foreach ($project in $cohorts.Keys) {
    dotnet build "tests/$project/$project.csproj" -c Release -f net10.0 `
        --artifacts-path $proofRoot -m:1 -p:UseSharedCompilation=false `
        -p:GeneratePackageOnBuild=false -v:quiet
    if ($LASTEXITCODE -ne 0) { throw "$project build failed" }
    $runner = "$proofRoot/bin/$project/release_net10.0"
    pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 `
        -RunnerJson "$runner/xunit.runner.json"
    if ($LASTEXITCODE -ne 0) { throw "$project runner hardening failed" }
    dotnet vstest "$runner/AiDotNetTests.dll" `
        "--Logger:trx;LogFileName=$project.trx" "--ResultsDirectory:$proofRoot/results"
    if ($LASTEXITCODE -ne 0) { throw "$project tests failed" }
    [xml]$report = Get-Content "$proofRoot/results/$project.trx"
    $counts = $report.TestRun.ResultSummary.Counters
    if ([int]$counts.passed -ne $cohorts[$project] -or [int]$counts.failed -ne 0 -or
        [int]$counts.total -ne $cohorts[$project]) { throw "$project unexpected test counts" }
}
```

## Deliberate limits

This batch does not fix the remaining GPT4Vision/BLIP/BLIP2/Flamingo/LLaVA/ImageBind
ONNX findings or Finch optimizer/schedule wiring. Those review threads remain open.
It does not prove learned caption generation from encoder-only ONNX graphs,
representation quality, trained model accuracy, physical GPU execution, or GPU
speedups. The PR remains draft until its remaining review contracts are addressed.
