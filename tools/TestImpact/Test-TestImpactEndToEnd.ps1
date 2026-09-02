<#
.SYNOPSIS
    Proves a certified map selects a strict subset in a real git diff and fails closed for infra.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$selector = Join-Path $PSScriptRoot 'Select-Shards.ps1'
$certificateValidator = Join-Path $PSScriptRoot 'Test-CertifiedShardMap.ps1'
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ("aidotnet-impact-e2e-" + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
$fixtureFailure = $null

function Assert-True {
    param([bool] $Condition, [string] $What)
    if (-not $Condition) { [void] $failures.Add($What) }
}

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments)] [string[]] $Arguments)

    $output = & git @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $details = ($output | Out-String).Trim()
        throw "git $($Arguments -join ' ') failed with exit code $exitCode$(if ($details) { ": $details" })"
    }
    return $output
}

function Assert-CertificateRejected {
    param([string] $CertificateFile, [string] $ExpectedPattern, [string] $What)

    $message = $null
    try {
        & $certificateValidator -MapFile shard-map.json -CertificateFile $CertificateFile `
            -CertificationRunId 102
    }
    catch { $message = [string] $_.Exception.Message }

    if ($null -eq $message) {
        [void] $failures.Add($What)
    }
    elseif ($message -notmatch $ExpectedPattern) {
        [void] $failures.Add("$What (rejected for the wrong reason: $message)")
    }
}

try {
    New-Item -ItemType Directory -Path (Join-Path $fixture 'src') -Force | Out-Null
    Push-Location $fixture
    try {
        Invoke-Git init --quiet
        Invoke-Git config user.email 'ci-impact-fixture@example.invalid'
        Invoke-Git config user.name 'CI impact fixture'
        Invoke-Git config commit.gpgSign false
        $disabledHooks = Join-Path $fixture 'disabled-hooks'
        New-Item -ItemType Directory -Path $disabledHooks -Force | Out-Null
        Invoke-Git config core.hooksPath $disabledHooks
        @('one', 'alpha before', 'three', 'beta unchanged', 'five') |
            Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        Invoke-Git add src/Feature.cs
        Invoke-Git commit --quiet -m baseline
        $baseSha = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()
        if ($baseSha -cnotmatch '^[0-9a-f]{40}$') {
            throw "fixture HEAD is not a full lowercase commit ID: '$baseSha'"
        }

        $map = [ordered]@{
            schemaVersion = 1
            sha = $baseSha
            knownShards = @('Alpha', 'Beta')
            alwaysRun = @('Always')
            files = [ordered]@{
                'src/Feature.cs' = @(
                    [ordered]@{ s = 0; r = @(2, 2) },
                    [ordered]@{ s = 1; r = @(4, 4) }
                )
            }
        }
        $map | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath shard-map.json -Encoding utf8
        [ordered]@{
            schemaVersion = 1
            candidateMapRunId = 100
            candidateMapSha = $baseSha
            auditSourceRunId = 101
            auditSourceSha = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            certificationRunId = 102
            escalated = $false
            wouldRun = 2
            wouldSkip = 1
            failedShards = 1
            missCount = 0
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath certification.json -Encoding utf8

        $global:LASTEXITCODE = 97
        try {
            & $certificateValidator -MapFile shard-map.json -CertificateFile certification.json `
                -CertificationRunId 102
            if ($LASTEXITCODE -ne 0) {
                throw "validator returned exit code $LASTEXITCODE"
            }
        }
        catch {
            [void] $failures.Add("the valid end-to-end certificate was rejected: $($_.Exception.Message)")
        }

        @('one', 'alpha after', 'three', 'beta unchanged', 'five') |
            Set-Content -LiteralPath src/Feature.cs -Encoding utf8
        Invoke-Git add src/Feature.cs
        Invoke-Git commit --quiet -m narrow-change
        & $selector -MapFile shard-map.json -ExpectedShards @('Alpha', 'Beta', 'Always') `
            -OutFile selection.json
        $selection = Get-Content selection.json -Raw | ConvertFrom-Json
        Assert-True (-not [bool] $selection.escalate) 'a covered one-line edit escalated'
        Assert-True (@($selection.shards).Count -eq 2) 'the covered edit did not select a strict 2/3 subset'
        Assert-True ($selection.shards -contains 'Alpha') 'the covering shard was omitted'
        Assert-True ($selection.shards -contains 'Always') 'the always-run shard was omitted'
        Assert-True (-not ($selection.shards -contains 'Beta')) 'an unaffected shard was selected'

        New-Item -ItemType Directory -Path .github/workflows -Force | Out-Null
        'name: changed-ci' | Set-Content -LiteralPath .github/workflows/fixture.yml -Encoding utf8
        Invoke-Git add .github/workflows/fixture.yml
        Invoke-Git commit --quiet -m infrastructure-change
        & $selector -MapFile shard-map.json -ExpectedShards @('Alpha', 'Beta', 'Always') `
            -OutFile infrastructure-selection.json
        $infrastructure = Get-Content infrastructure-selection.json -Raw | ConvertFrom-Json
        Assert-True ([bool] $infrastructure.escalate) 'a workflow edit did not fail closed to the full matrix'

        $badCertificate = Get-Content certification.json -Raw | ConvertFrom-Json
        $badCertificate.missCount = 1
        $badCertificate | ConvertTo-Json -Depth 5 | Set-Content bad-certification.json -Encoding utf8
        Assert-CertificateRejected -CertificateFile bad-certification.json `
            -ExpectedPattern 'selection miss' `
            -What 'a certificate recording a selection miss was accepted'

        $badCertificate = Get-Content certification.json -Raw | ConvertFrom-Json
        $badCertificate.failedShards = 0
        $badCertificate | ConvertTo-Json -Depth 5 | Set-Content no-failure-certification.json -Encoding utf8
        Assert-CertificateRejected -CertificateFile no-failure-certification.json `
            -ExpectedPattern 'failedShards must be at least 1' `
            -What 'a certificate without a failure opportunity was accepted'
    }
    finally {
        Pop-Location
    }
}
catch {
    $fixtureFailure = $_
    throw
}
finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    if ($resolvedFixture.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedFixture).StartsWith('aidotnet-impact-e2e-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        $message = "refusing to remove unexpected fixture path '$resolvedFixture'"
        if ($null -ne $fixtureFailure) { Write-Warning "$message; preserving the original failure" }
        else { throw $message }
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'Test-impact end-to-end proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'Test-impact end-to-end proof passed: certified covered edit selected 2/3; CI edit escalated.'
exit 0
