# PR #2130: constructor-owned validation surface

Baseline: `f5d4aec0377f740026fc825c6d1bb501b54d4258`.
Review: [3964560916](https://github.com/ooples/AiDotNet/pull/2130#discussion_r3964560916),
thread `PRRT_kwDOKSXUF86ggA4Q`.

## Scope

Only the parameterless `Validate()` methods on `FinchOptions`, `GLAOptions`,
`GatedDeltaNetOptions`, `GriffinOptions`, `HawkOptions`, and
`RecurrentGemmaOptions` change from public to internal. Their bodies, public
options types, constructors, properties, defaults, and all model calls stay
unchanged. These six validation methods were added by the options migration;
they did not exist on those types at merge-base
`2a53ff3d4e27845773b4c9ad0a52b81f089613f7`.

The shared scalar test's reflection lookup now explicitly includes both public
and non-public instance methods. This keeps every existing invalid-value,
shipped-default, copy, and diagnostic assertion running after the visibility
change. It does not change the 17-language-model plus one-image-model census.
The actual main assembly already grants `InternalsVisibleTo` to `AiDotNetTests`.

Six new shared API-boundary cases each establish that the options type/default
constructor remain public, the declared validator is exactly assembly-visible
(not private/protected/public), default validation succeeds, and a zero
vocabulary still fails with the exact option name and `options` parameter.

## Failure-first and final evidence

The source-linked scalar project compiles the real options and their complete
base chain. It does not substitute production classes or require a native
model allocation.

| Run | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| New six cases, before the production change, net10.0 | 0 | 6 | 0 |
| Entire scalar suite, after the change, net10.0 | 390 | 0 | 0 |
| Entire scalar suite, after the change, net8.0 | 390 | 0 | 0 |
| Entire scalar suite, after the change, net471 | 390 | 0 | 0 |

Each baseline failure names the intended cause: the corresponding `Validate`
method is not internal. The final total is the previous 384 tests plus six
new boundary tests. No test is skipped or weakened to accommodate accessibility.

TRX files are under `artifacts/pr2130-validation-surface/results/`:

- `validation-surface-baseline.trx`
- `validation-surface-final-net10.0.trx`
- `validation-surface-final-net8.0.trx`
- `validation-surface-final-net471.trx`

Final scalar test DLLs are under
`artifacts/pr2130-validation-surface/build/bin/AiDotNet.OptionsContractTests/release_<tfm>/`.

| TFM | `AiDotNet.OptionsContractTests.dll` SHA-256 |
| --- | --- |
| net10.0 | `5DEEAA8EA33C6E775BB669DC857BA2BABABE0C345CA4D023D14BEA37E4DCE879` |
| net8.0 | `78F6F07E8FF0905CB038E9A1E0C39E1DB6B1D14331F10D132B3E673D8D0FF178` |
| net471 | `31E314F0AE4752C06642C53D7B6C892F0245FE1ED8EB421CBE78379B5D914A27` |

## Reproduction

From the repository root, use redirected outputs to preserve the earlier
native-model proof binaries:

```powershell
foreach ($validationTfm in @('net10.0', 'net8.0', 'net471')) {
    dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj `
        -c Release -f $validationTfm `
        --artifacts-path artifacts/pr2130-validation-surface/build `
        -m:1 -p:UseSharedCompilation=false `
        --logger "trx;LogFileName=validation-surface-final-$validationTfm.trx" `
        --results-directory artifacts/pr2130-validation-surface/results -v:quiet
    if ($LASTEXITCODE -ne 0) { throw "Validation surface tests failed for $validationTfm." }
}
```

To reproduce the negative control, in a separate worktree at the baseline SHA
add only the new `SequenceValidationSurfaceTests.cs` and its source link from
this change. Do not apply the six visibility edits. Run:

```powershell
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj `
    -c Release -f net10.0 `
    --artifacts-path artifacts/pr2130-validation-surface/build `
    -m:1 -p:UseSharedCompilation=false `
    --filter FullyQualifiedName~SequenceValidationSurfaceTests `
    --logger 'trx;LogFileName=validation-surface-baseline.trx' `
    --results-directory artifacts/pr2130-validation-surface/results -v:quiet
if ($LASTEXITCODE -eq 0) { throw 'The negative control unexpectedly passed.' }
$baselineReport = [xml](Get-Content -LiteralPath `
    artifacts/pr2130-validation-surface/results/validation-surface-baseline.trx -Raw)
$baselineCounts = $baselineReport.TestRun.ResultSummary.Counters
if ($baselineCounts.total -ne 6 -or $baselineCounts.failed -ne 6 -or $baselineCounts.passed -ne 0) {
    throw 'The negative control did not execute and fail exactly the six intended cases.'
}
```

The recorded final executions used `--no-restore` after the baseline restore.
Reproduction above includes restore so a clean worktree does not rely on prior
assets. The baseline TRX must also show the six intended internal-visibility
failures, not build or discovery failures.

## Limits

This is new scalar/API-boundary proof, not a rerun of all native model tests.
No main-library/model-test output, earlier native proof inventory, ONNX graph
behavior, GPU selection, training algorithm, default, or allocation changes
are included. The remaining ONNX/effective-configuration, duplicate gradient
setting, and GPT-4 naming questions remain separate review items. This change
does not establish that the whole PR or current remote CI is green.
