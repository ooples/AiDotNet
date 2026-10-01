<#
.SYNOPSIS
    Proves Select-AffectedTests passes through (and the step succeeds) for a change with no C# edits.

.DESCRIPTION
    A workflow-only pull request broke the Build job twice over: the line-level diff was piped to Set-Content, which
    never runs on an empty pipeline, so no diff file existed and TypeImpact crashed opening it; and the fallback that
    catches such failures left the native exit code in $LASTEXITCODE, which the Actions pwsh wrapper returns as the
    step's exit code. This runs the real script the way the workflow does - in its own pwsh, followed by the
    wrapper's `exit $LASTEXITCODE` - in a throwaway repository whose only change is a YAML file, with a self-test
    that fails, and asserts the step exits 0, passes through, and wrote both change files.
#>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script = (Resolve-Path (Join-Path $PSScriptRoot 'Select-AffectedTests.ps1')).Path
$root = Join-Path ([IO.Path]::GetTempPath()) ("select-affected-" + [guid]::NewGuid().ToString('N'))
$repo = Join-Path $root 'repo'
$tmp = Join-Path $root 'tmp'
New-Item -ItemType Directory -Path $repo, $tmp -Force | Out-Null
# The workflow runs self-tests in ONE pwsh process: restore every variable this sets, or the step's later
# commands would write their outputs into this fixture.
$saved = @{}
$names = 'GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY', 'GITHUB_EVENT_NAME', 'REQUIRES_SHARDS', 'DELTA_MODE',
    'COLLECT_COVERAGE_EVERYWHERE', 'SELECTED_MATRIX', 'SELECTED_LEDGER_MATRIX', 'SELECTED_SKIPPED', 'TMP', 'TEMP', 'TMPDIR'
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }
try {
    Push-Location $repo
    try {
        & git init -q
        & git config user.email 'fixture@example.invalid'
        & git config user.name 'fixture'
        New-Item -ItemType Directory -Path '.github', 'tools/TestImpact/TypeImpact' -Force | Out-Null
        Set-Content -LiteralPath '.github/test-shards.yml' -Value 'shard: []'
        # The self-test fails: a native exit code the selector must tolerate.
        Set-Content -LiteralPath 'tools/TestImpact/TypeImpact/Test-TypeImpact.ps1' -Value 'exit 3'
        Set-Content -LiteralPath 'ci.yml' -Value 'a: 1'
        & git add -A; & git commit -q -m base
        Set-Content -LiteralPath 'ci.yml' -Value 'a: 2'   # the only change: no C# file
        & git commit -q -am change
        if ($LASTEXITCODE -ne 0) { throw 'fixture repository setup failed' }
    }
    finally { Pop-Location }

    $output = Join-Path $root 'github-output.txt'
    $summary = Join-Path $root 'summary.md'
    $env:GITHUB_OUTPUT = $output
    $env:GITHUB_STEP_SUMMARY = $summary
    $env:GITHUB_EVENT_NAME = 'push'
    $env:REQUIRES_SHARDS = 'true'
    $env:DELTA_MODE = ''
    $env:COLLECT_COVERAGE_EVERYWHERE = ''
    $env:SELECTED_MATRIX = '{"include":[]}'
    $env:SELECTED_LEDGER_MATRIX = '{"include":[]}'
    $env:SELECTED_SKIPPED = '[]'
    $env:TMP = $tmp; $env:TEMP = $tmp; $env:TMPDIR = $tmp

    # The Actions pwsh wrapper: run the script, then `exit $LASTEXITCODE`. yq is stubbed so the run does not depend
    # on the runner having it; it returns one shard, so the selector reaches the failing self-test.
    $command = "function yq { '[{""name"":""x""}]' }; Set-Location '$repo'; & '$script' -Repository '$repo'; " +
        'if ((Test-Path -LiteralPath variable:\LASTEXITCODE)) { exit $LASTEXITCODE }'
    & pwsh -NoProfile -Command $command
    $stepExit = $LASTEXITCODE

    if ($stepExit -ne 0) { throw "the step failed (exit $stepExit) after the selector fell back to passing through" }
    $recorded = Get-Content -LiteralPath $output -Raw
    if ($recorded -notmatch 'impact_mode=passthrough') { throw "the selector did not pass through: $recorded" }
    foreach ($file in 'type-impact-changes.txt', 'type-impact-changes.diff') {
        if (-not (Test-Path -LiteralPath (Join-Path $tmp $file))) { throw "$file was not written for a change without C# edits" }
    }
    Write-Host 'Select-AffectedTests passes through and succeeds for a change without C# edits.'
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}
