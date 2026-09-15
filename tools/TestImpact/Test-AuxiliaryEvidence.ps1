[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/ReviewFixtureCleanup.ps1"
$root = Join-Path ([IO.Path]::GetTempPath()) "aidotnet-auxiliary-evidence-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $root | Out-Null
$previousOutput = $env:GITHUB_OUTPUT
$env:GITHUB_OUTPUT = Join-Path $root 'outputs.txt'
$workload = @{ name = 'Parameter sweep - Count 0/8'; workload = 'ParameterSweep' } | ConvertTo-Json -Compress
try {
    foreach ($case in @(
        @{ Name = 'complete'; Outcome = 'success'; Total = 1; Executed = 1; Passed = 1; Coverage = $true; Trx = $true; Digest = $true },
        @{ Name = 'failed-findings'; Outcome = 'success'; Total = 1; Executed = 1; Passed = 0; Coverage = $true; Trx = $true; Digest = $false },
        @{ Name = 'partial'; Outcome = 'success'; Total = 2; Executed = 1; Passed = 1; Coverage = $true; Trx = $true; Digest = $false },
        @{ Name = 'empty'; Outcome = 'success'; Total = 0; Executed = 0; Passed = 0; Coverage = $true; Trx = $true; Digest = $false },
        @{ Name = 'crashed'; Outcome = 'failure'; Total = 1; Executed = 1; Passed = 1; Coverage = $true; Trx = $true; Digest = $false },
        @{ Name = 'no-trx'; Outcome = 'success'; Total = 1; Executed = 1; Passed = 1; Coverage = $true; Trx = $false; Digest = $false },
        @{ Name = 'no-coverage'; Outcome = 'success'; Total = 1; Executed = 1; Passed = 1; Coverage = $false; Trx = $true; Digest = $false }
    )) {
        $results = Join-Path $root $case.Name
        $digests = Join-Path $results 'digests'
        New-Item -ItemType Directory -Path $results | Out-Null
        if ($case.Trx) {
            "<TestRun><ResultSummary><Counters total='$($case.Total)' executed='$($case.Executed)' passed='$($case.Passed)' /></ResultSummary></TestRun>" |
                Set-Content (Join-Path $results 'test.trx')
        }
        if ($case.Coverage) {
            '<CoverageSession><File uid="1" fullPath="/fixture/src/Feature.cs" /><SequencePoint fileid="1" sl="10" vc="1" /></CoverageSession>' |
                Set-Content (Join-Path $results 'coverage.opencover.xml')
        }
        & "$PSScriptRoot/Write-AuxiliaryEvidence.ps1" -WorkloadJson $workload -TestStepOutcome $case.Outcome `
            -ResultsDirectory $results -DigestDirectory $digests
        if ($LASTEXITCODE -ne 0) { throw "$($case.Name) evidence writer failed." }
        $metadata = Get-Content (Join-Path $results 'shard-metadata.json') -Raw | ConvertFrom-Json
        if ($metadata.shard -cne 'Parameter sweep - Count 0/8' -or $metadata.testStepOutcome -cne $case.Outcome) {
            throw "$($case.Name) lost workload identity or test outcome."
        }
        $actual = Test-Path (Join-Path $digests 'Parameter_sweep_Count_0_8.digest.json')
        if ($actual -ne $case.Digest) { throw "$($case.Name) authorized an incorrect coverage disposition." }
    }
}
finally {
    $env:GITHUB_OUTPUT = $previousOutput
    Remove-ReviewFixtureDirectory -LiteralPath $root -ExpectedLeafPrefix 'aidotnet-auxiliary-evidence-' -ThrowOnUnsafePath
}
if (Test-Path -LiteralPath $root) { throw 'Auxiliary evidence fixture was not cleaned up.' }
Write-Host 'Auxiliary evidence passed: complete coverage plus six incomplete/failed controls.'
exit 0
