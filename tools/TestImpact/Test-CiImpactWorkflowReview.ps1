<#
.SYNOPSIS
    Proves workflow contract checks reject unsafe wiring, not just the checked-in happy path.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$workflowPath = Join-Path $repositoryRoot '.github/workflows/sonarcloud.yml'
$contractPath = Join-Path $PSScriptRoot 'Test-CiImpactWorkflow.ps1'
$workflow = Get-Content -LiteralPath $workflowPath -Raw
$mapHeadLine = '(?m)^[ \t]+-PullRequestHeadSha \$env:PR_HEAD_SHA `[ \t]*\r?$'
if ([regex]::Matches($workflow, $mapHeadLine).Count -ne 1) {
    throw 'The fixture must identify exactly one map-backed PR-head argument, excluding the classifier.'
}
$cases = @(
    [pscustomobject]@{
        Name = 'map-head-removed'
        Content = [regex]::Replace($workflow, $mapHeadLine, '')
    },
    [pscustomobject]@{
        Name = 'map-head-wrong-variable'
        Content = [regex]::Replace($workflow, $mapHeadLine, '                    -PullRequestHeadSha $env:WRONG_HEAD_SHA `')
    },
    [pscustomobject]@{
        Name = 'map-head-only-in-comment'
        Content = [regex]::Replace($workflow, $mapHeadLine, '                    # -PullRequestHeadSha $env:PR_HEAD_SHA `')
    }
)

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ('aidotnet-ci-contract-review-' + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
try {
    New-Item -ItemType Directory -Path $fixture | Out-Null
    Push-Location $repositoryRoot
    try {
        $baseline = @(& pwsh -NoProfile -File $contractPath 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "The unmodified workflow must pass first: $($baseline -join [Environment]::NewLine)" }
        foreach ($case in $cases) {
            if (-not $case.Content.Contains('-PullRequestHeadSha $env:PR_HEAD_SHA -OutFile path-classification.json')) {
                throw 'A map-scoping negative control accidentally changed the classifier.'
            }
            $path = Join-Path $fixture ($case.Name + '.yml')
            Set-Content -LiteralPath $path -Value $case.Content -Encoding utf8
            $output = @(& pwsh -NoProfile -File $contractPath -ValidationWorkflow $path 2>&1)
            if ($LASTEXITCODE -eq 0) {
                [void] $failures.Add("$($case.Name): unsafe map-backed PR scoping passed the workflow contract")
            }
            elseif (($output -join [Environment]::NewLine) -notmatch 'map-backed selector.*pull.request') {
                [void] $failures.Add("$($case.Name): rejected for an unrelated reason: $($output -join [Environment]::NewLine)")
            }
            else {
                Write-Host "Rejected unsafe workflow: $($case.Name)"
            }
        }
    }
    finally { Pop-Location }
}
finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    $expectedPrefix = $tempRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if ($resolvedFixture.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase) -and
        [IO.Path]::GetFileName($resolvedFixture).StartsWith('aidotnet-ci-contract-review-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Host $failure }
    exit 1
}
Write-Host "Workflow review controls passed: baseline accepted, $($cases.Count) unsafe mutations rejected."
exit 0
