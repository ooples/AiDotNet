# CI shard evidence for the sequence-options migration

Snapshot checked on 2026-09-11. This document is an evidence record, not a claim
that all shards are green or that an old failure classification remains current.

## Current recorded CI evidence

| Field | Recorded value |
| --- | --- |
| PR | #2128 |
| Tested head requested by the run | `671836a347436f4b023ebe3f02a86ea333ad0686` |
| Dependency at that head | AiDotNet.Tensors 0.130.2, from `Directory.Packages.props` |
| Workflow / attempt | Build & SonarCloud / attempt 1 |
| Run | [34303995195](https://github.com/ooples/AiDotNet/actions/runs/34303995195) |
| Created | 2026-09-09 02:38:07 UTC |
| Conclusion | Cancelled |
| Jobs returned by the paginated jobs API | 1 |
| Only job | Resolve certified PR validation, job 102316758308, cancelled |
| Observed test shards | 0 |
| Current passing-shard list | None established by this run |
| Current failing-test classification | Unknown: no shard results were produced |
| Current work assignments derived from these results | None |

The source and dependency above describe that exact run, not a later review-fix
commit. Subsequent fixes need their own local test evidence and/or completed CI
results. Cancellation is neither a passing matrix nor evidence of a model defect.

Recheck the run and every jobs page using:

```powershell
gh api repos/ooples/AiDotNet/actions/runs/34303995195
gh api --paginate 'repos/ooples/AiDotNet/actions/runs/34303995195/jobs?per_page=100'
```

The previous June inventory (run 28023414937, head 01a4ddb4, Tensors 0.102.12) is
[retained in Git history](https://github.com/ooples/AiDotNet/blob/671836a347436f4b023ebe3f02a86ea333ad0686/CI_SHARD_INVENTORY.md).
Its reported 44/19/1 counts, pass list, root-cause claims, and owner assignments
must not be used as the status of this branch.

## Arena comparison: diagnostic evidence, not a verdict

Disabling the inference arena changes allocation, recycling, and timing.
A pass with the arena disabled suggests an interaction; it does not prove tensor
aliasing, pool corruption, or OOM. A failure with it disabled does not exclude an
arena-related defect. A killed host without a captured exception is unclassified.

Compare the same built binary and test filter in separate processes, with all
other settings unchanged. The helper below requires a real filter, rejects an
empty/placeholder filter, and refuses to accept a run with zero executed tests.
Build the Release/net10.0 test project first. Keep both TRXs, exit codes, the
test/library/tensor assembly hashes, and any crash/resource evidence before
assigning a root cause. Do not rebuild or change dependencies during comparison.

```powershell
function Invoke-ArenaComparison {
    param(
        [Parameter(Mandatory)]
        [ValidateScript({ -not [string]::IsNullOrWhiteSpace($_) -and -not $_.Contains('<YourModelTests>') })]
        [string] $TestFilter
    )
    $testProject = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
    $binaryRoot = 'tests/AiDotNet.Tests/bin/Release/net10.0'
    $binaryHashes = [ordered]@{}
    foreach ($name in 'AiDotNetTests.dll', 'AiDotNet.dll', 'AiDotNet.Tensors.dll') {
        $path = Join-Path $binaryRoot $name
        $binaryHashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash
    }
    $resultsRoot = Join-Path ([IO.Path]::GetTempPath()) ('aidotnet-arena-comparison-' + [guid]::NewGuid().ToString('N'))
    $previousArena = [Environment]::GetEnvironmentVariable('AIDOTNET_INFERENCE_ARENA')
    try {
        foreach ($arenaMode in 1, 0) {
            [Environment]::SetEnvironmentVariable('AIDOTNET_INFERENCE_ARENA', [string]$arenaMode)
            $reportName = "arena-$arenaMode.trx"
            dotnet test $testProject -c Release -f net10.0 --no-build --no-restore --filter $TestFilter --logger "trx;LogFileName=$reportName" --results-directory $resultsRoot
            $testExit = $LASTEXITCODE
            $reportPath = Join-Path $resultsRoot $reportName
            if (-not (Test-Path -LiteralPath $reportPath)) { throw "No test report was produced: $reportPath (exit $testExit)" }
            [xml]$report = Get-Content -LiteralPath $reportPath -Raw -ErrorAction Stop
            $executed = @($report.TestRun.Results.UnitTestResult | Where-Object { $_ -and $_.outcome -ne 'NotExecuted' })
            if ($executed.Count -eq 0) { throw "No tests executed for filter '$TestFilter'." }
            foreach ($name in $binaryHashes.Keys) {
                $path = Join-Path $binaryRoot $name
                if ((Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash -ne $binaryHashes[$name]) {
                    throw "An assembly changed during comparison: $name"
                }
            }
            [pscustomobject]@{ Arena = $arenaMode; ExitCode = $testExit; Executed = $executed.Count; AssemblySha256 = $binaryHashes; Report = $reportPath }
        }
    }
    finally {
        if ($null -eq $previousArena) {
            Remove-Item Env:AIDOTNET_INFERENCE_ARENA -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable('AIDOTNET_INFERENCE_ARENA', $previousArena)
        }
    }
}
```

Invoke the helper with `-TestFilter` set to the exact failing class or test from
the current run. Do not replace production defaults, weaken assertions, or label
a failure an arena bug solely because one side of this comparison passes.
