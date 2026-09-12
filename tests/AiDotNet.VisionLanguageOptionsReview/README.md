# Native vision-language options review runner

This small runner compiles the actual AiDotNet project and source-links the
native-options, generator, existing VisionMamba, shared-options and ratchet
fixtures. It does not stub model/layer dependencies or run every model shard.
The generator check compiles and invokes the emitted Flamingo factory against
the real library.

```powershell
$ErrorActionPreference = 'Stop'
$reviewExpectedCount = 525
$reviewResults = 'artifacts/vision-language-review'
if (Test-Path -LiteralPath $reviewResults) { throw 'Choose a fresh results directory; preserve earlier proof.' }
dotnet build tests/AiDotNet.VisionLanguageOptionsReview/AiDotNet.VisionLanguageOptionsReview.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false
if ($LASTEXITCODE -ne 0) { throw 'Build failed; do not execute an older test binary.' }
pwsh -NoProfile -File ./.github/scripts/harden-xunit-runner.ps1 -RunnerJson tests/AiDotNet.VisionLanguageOptionsReview/bin/Release/net10.0/xunit.runner.json
if ($LASTEXITCODE -ne 0) { throw 'Runner hardening failed; do not run tests.' }
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
dotnet vstest tests/AiDotNet.VisionLanguageOptionsReview/bin/Release/net10.0/AiDotNetTests.dll '--Logger:trx;LogFileName=vision-language-review.trx' "--ResultsDirectory:$reviewResults"
if ($LASTEXITCODE -ne 0) { throw 'Actual-library regression tests failed.' }
$reviewTrxPath = "$reviewResults/vision-language-review.trx"
if ((Get-Item -LiteralPath $reviewTrxPath).Length -eq 0) { throw 'Empty TRX; result not retained.' }
$reviewTrx = [xml](Get-Content -LiteralPath $reviewTrxPath -Raw)
if ([int]$reviewTrx.TestRun.ResultSummary.Counters.passed -ne $reviewExpectedCount -or
    [int]$reviewTrx.TestRun.ResultSummary.Counters.total -ne $reviewExpectedCount) {
    throw "Expected all $reviewExpectedCount checks from this reviewed snapshot to pass."
}
```

Replace `net10.0` in the build target and binary/runner paths with `net8.0` or
`net471` for the other supported test runtimes.
Runtime fixtures explicitly initialize the test engine/license support because
.NET Framework does not invoke the assembly module initializer automatically.
CPU forcing is a test-process setting, not a production restriction.
The separate `pwsh` invocation gives hardening a defined process exit status;
checking an unset status after direct script invocation can falsely reject a
successful hardening step. Use a fresh results directory for each proof run.

For the three new fixture classes only, set `$reviewExpectedCount = 119` and add
this VSTest filter (104 option cases, 13 real construction/execution cases, and
2 actual generated-factory cases):

```text
FullyQualifiedName~VisionLanguageNativeOptionsTests|FullyQualifiedName~VisionLanguageNativeConstructionTests|FullyQualifiedName~GeneratedVisionLanguageFixtureContractTests
```

The production and generator project paths can be overridden with
`ReviewProductionProject` and `ReviewGeneratorProject`. `ReviewGeneratorOnly=true`
excludes native/manual scalar fixtures for the original-generator control.
That mode contains two cases; the untouched-original generator is expected to
fail them, whereas the current generator passes both.
Use compatible project checkpoints: old and renamed options APIs cannot be
mixed and their compilation failure is not a runtime defect reproduction.

See [the bounded proof and remaining scope](../../.github/PR2130_REVIEW_PROOF.md).
