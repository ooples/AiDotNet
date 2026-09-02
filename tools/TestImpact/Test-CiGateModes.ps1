<#
.SYNOPSIS
    Executes the checked-in CI Gate Bash body in full-validation and exact-reuse modes.
#>
[CmdletBinding()]
param([string] $Workflow = '.github/workflows/sonarcloud.yml')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$lines = Get-Content -LiteralPath $Workflow
$nameLine = ($lines | Select-String -SimpleMatch '- name: Evaluate required jobs' |
    Select-Object -First 1).LineNumber
if (-not $nameLine) { throw 'Evaluate required jobs step is absent.' }

$runLine = 0
$indent = 0
$stepIndent = $lines[$nameLine - 1].Length - $lines[$nameLine - 1].TrimStart().Length
for ($i = $nameLine; $i -le $lines.Count; $i++) {
    $current = $lines[$i - 1]
    $currentIndent = $current.Length - $current.TrimStart().Length
    if ($i -gt $nameLine -and $current.Trim() -and $currentIndent -le $stepIndent) {
        break
    }
    if ($current -match '^(\s*)run:\s*\|') {
        $runLine = $i
        $indent = $Matches[1].Length
        break
    }
}
if (-not $runLine) { throw 'CI Gate run block is absent.' }

$body = [System.Collections.Generic.List[string]]::new()
for ($i = $runLine + 1; $i -le $lines.Count; $i++) {
    $line = $lines[$i - 1]
    $lineIndent = $line.Length - $line.TrimStart().Length
    if ($line.Trim() -and $lineIndent -le $indent) { break }
    [void] $body.Add($(if ($line.Length -ge $indent + 2) { $line.Substring($indent + 2) } else { '' }))
}

$bash = if ($IsWindows) {
    $candidate = Join-Path $env:ProgramFiles 'Git\bin\bash.exe'
    if (-not (Test-Path -LiteralPath $candidate)) { throw 'Git Bash is required for this proof on Windows.' }
    $candidate
}
else {
    (Get-Command bash -ErrorAction Stop).Source
}

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ("aidotnet-ci-gate-" + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
$fixtureFailure = $null

try {
    New-Item -ItemType Directory -Path $fixture -Force | Out-Null
    $script = Join-Path $fixture 'gate.sh'
    [IO.File]::WriteAllText($script, ($body -join "`n"), [Text.UTF8Encoding]::new($false))

    function Invoke-GateCase {
        param(
            [string] $Name,
            [string] $Source = 'success',
            [string] $Reuse,
            [string] $Promotion,
            [string] $CodeQL,
            [string] $Tests,
            [string] $Verdict,
            [int] $ExpectedExit
        )

        $environmentNames = @(
            'SOURCE_RESULT', 'BUILD_RESULT', 'BUILD_COMPAT_RESULT', 'CODEQL_RESULT',
            'SELECT_RESULT', 'TESTS_RESULT', 'PARAMETER_SWEEP_RESULT', 'MODEL_SHAPE_RESULT',
            'REGRESSION_ANALYSIS_RESULT', 'VERDICT_ENFORCED', 'AGGREGATE_ANALYSIS_RESULT',
            'SIZE_CHECK_RESULT', 'PROMOTION_RESULT', 'SONAR_RESULT', 'REUSED_VALIDATION',
            'GITHUB_STEP_SUMMARY'
        )
        $savedEnvironment = @{}
        foreach ($variableName in $environmentNames) {
            $savedEnvironment[$variableName] = [Environment]::GetEnvironmentVariable(
                $variableName,
                [EnvironmentVariableTarget]::Process)
        }

        try {
            $env:SOURCE_RESULT = $Source
            $env:BUILD_RESULT = 'success'
            $env:BUILD_COMPAT_RESULT = 'success'
            $env:CODEQL_RESULT = $CodeQL
            $env:SELECT_RESULT = 'success'
            $env:TESTS_RESULT = $Tests
            $env:PARAMETER_SWEEP_RESULT = 'success'
            $env:MODEL_SHAPE_RESULT = 'success'
            $env:REGRESSION_ANALYSIS_RESULT = 'success'
            $env:VERDICT_ENFORCED = $Verdict
            $env:AGGREGATE_ANALYSIS_RESULT = 'success'
            $env:SIZE_CHECK_RESULT = 'success'
            $env:PROMOTION_RESULT = $Promotion
            $env:SONAR_RESULT = 'success'
            $env:REUSED_VALIDATION = $Reuse
            $env:GITHUB_STEP_SUMMARY = Join-Path $fixture "$Name-summary.md"

            & $bash $script 2>&1 | Out-Null
            $actualExit = $LASTEXITCODE
            if ($actualExit -ne $ExpectedExit) {
                [void] $failures.Add("$Name expected exit $ExpectedExit, got $actualExit")
            }
        }
        finally {
            foreach ($variableName in $environmentNames) {
                [Environment]::SetEnvironmentVariable(
                    $variableName,
                    $savedEnvironment[$variableName],
                    [EnvironmentVariableTarget]::Process)
            }
        }
    }

    Invoke-GateCase -Name reuse_success -Reuse true -Promotion success -CodeQL skipped `
        -Tests skipped -Verdict false -ExpectedExit 0
    Invoke-GateCase -Name reuse_missing_promotion -Reuse true -Promotion skipped -CodeQL skipped `
        -Tests skipped -Verdict false -ExpectedExit 1
    Invoke-GateCase -Name full_success -Reuse false -Promotion skipped -CodeQL success `
        -Tests success -Verdict true -ExpectedExit 0
    Invoke-GateCase -Name full_codeql_failure -Reuse false -Promotion skipped -CodeQL failure `
        -Tests success -Verdict true -ExpectedExit 1
    Invoke-GateCase -Name full_known_test_failure -Reuse false -Promotion skipped -CodeQL success `
        -Tests failure -Verdict true -ExpectedExit 0
    Invoke-GateCase -Name full_unenforced_test_failure -Reuse false -Promotion skipped -CodeQL success `
        -Tests failure -Verdict false -ExpectedExit 1
    Invoke-GateCase -Name source_failure_blocks_reuse -Source failure -Reuse true `
        -Promotion success -CodeQL skipped -Tests skipped -Verdict false -ExpectedExit 1
    Invoke-GateCase -Name source_failure_blocks_full -Source failure -Reuse false `
        -Promotion skipped -CodeQL success -Tests success -Verdict true -ExpectedExit 1
    Invoke-GateCase -Name empty_reuse_takes_full_path -Reuse '' -Promotion skipped -CodeQL failure `
        -Tests success -Verdict true -ExpectedExit 1
}
catch {
    $fixtureFailure = $_
    throw
}
finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    if ($resolvedFixture.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedFixture).StartsWith('aidotnet-ci-gate-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        $message = "refusing to remove unexpected fixture path '$resolvedFixture'"
        if ($null -ne $fixtureFailure) { Write-Warning "$message; preserving the original failure" }
        else { throw $message }
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'CI Gate mode proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'CI Gate mode proof passed (reuse/full/default success and failure controls).'
exit 0
