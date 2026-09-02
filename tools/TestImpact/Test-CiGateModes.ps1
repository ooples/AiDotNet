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
for ($i = $nameLine; $i -le $lines.Count; $i++) {
    if ($lines[$i - 1] -match '^(\s*)run:\s*\|') {
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

try {
    New-Item -ItemType Directory -Path $fixture -Force | Out-Null
    $script = Join-Path $fixture 'gate.sh'
    [IO.File]::WriteAllText($script, ($body -join "`n"), [Text.UTF8Encoding]::new($false))

    function Invoke-GateCase {
        param(
            [string] $Name,
            [string] $Reuse,
            [string] $Promotion,
            [string] $CodeQL,
            [string] $Tests,
            [string] $Verdict,
            [int] $ExpectedExit
        )

        $env:SOURCE_RESULT = 'success'
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
        if ($LASTEXITCODE -ne $ExpectedExit) {
            [void] $failures.Add("$Name expected exit $ExpectedExit, got $LASTEXITCODE")
        }
    }

    Invoke-GateCase reuse_success true success skipped skipped false 0
    Invoke-GateCase reuse_missing_promotion true skipped skipped skipped false 1
    Invoke-GateCase full_success false skipped success success true 0
    Invoke-GateCase full_codeql_failure false skipped failure success true 1
    Invoke-GateCase full_known_test_failure false skipped success failure true 0
    Invoke-GateCase full_unenforced_test_failure false skipped success failure false 1
}
finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    if ($resolvedFixture.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedFixture).StartsWith('aidotnet-ci-gate-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        throw "refusing to remove unexpected fixture path '$resolvedFixture'"
    }
}

if ($failures.Count -gt 0) {
    Write-Host 'CI Gate mode proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'CI Gate mode proof passed (reuse/full success and failure controls).'
exit 0
