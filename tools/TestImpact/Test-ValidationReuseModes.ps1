<#
.SYNOPSIS
    Executes the checked-in exact-tree resolver against full and non-runtime certificate fixtures.
#>
[CmdletBinding()]
param([string] $Workflow = '.github/workflows/sonarcloud.yml')

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$lines = Get-Content -LiteralPath $Workflow
$nameLine = ($lines | Select-String -SimpleMatch '- name: Resolve exact-tree PR run' |
    Select-Object -First 1).LineNumber
if (-not $nameLine) { throw 'Resolve exact-tree PR run step is absent.' }

$runLine = 0
$indent = 0
$stepIndent = $lines[$nameLine - 1].Length - $lines[$nameLine - 1].TrimStart().Length
for ($i = $nameLine; $i -le $lines.Count; $i++) {
    $current = $lines[$i - 1]
    $currentIndent = $current.Length - $current.TrimStart().Length
    if ($i -gt $nameLine -and $current.Trim() -and $currentIndent -le $stepIndent) { break }
    if ($current -match '^(\s*)run:\s*\|') {
        $runLine = $i
        $indent = $Matches[1].Length
        break
    }
}
if (-not $runLine) { throw 'Exact-tree resolver run block is absent.' }

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
else { (Get-Command bash -ErrorAction Stop).Source }

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ("aidotnet-validation-reuse-" + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
$fixtureFailure = $null

try {
    New-Item -ItemType Directory -Path $fixture -Force | Out-Null
    $bin = Join-Path $fixture 'bin'
    $runnerTemp = Join-Path $fixture 'runner'
    $binPosix = if ($IsWindows) {
        '/' + $bin.Substring(0, 1).ToLowerInvariant() + $bin.Substring(2).Replace('\', '/')
    }
    else { $bin }
    New-Item -ItemType Directory -Path $bin, $runnerTemp -Force | Out-Null
    $resolverScript = Join-Path $fixture 'resolver.sh'
    [IO.File]::WriteAllText($resolverScript, ($body -join "`n"), [Text.UTF8Encoding]::new($false))

    $ghStub = @'
#!/usr/bin/env bash
set -euo pipefail
args="$*"
printf '%s\n' "$args" >> "$PROOF_GH_LOG"
case "$args" in
  *"/commits/${GITHUB_SHA}/pulls"*)
    printf '[{"number":9,"merged_at":"2026-01-01T00:00:00Z","merge_commit_sha":"%s","head":{"sha":"%s"}}]\n' "$GITHUB_SHA" "$PROOF_HEAD_SHA"
    ;;
  *"/actions/workflows/sonarcloud.yml/runs"*)
    if [[ "$args" == *"--jq"* ]]; then printf '0\n';
    elif [[ "$args" == *"event=merge_group"* ]]; then printf '{"workflow_runs":[]}\n';
    else printf '{"workflow_runs":[{"id":123,"conclusion":"success"}]}\n'; fi
    ;;
  *"/actions/runs/123"*)
    if [[ "$PROOF_MODE" == 'non-runtime' ]]; then
      printf '[{"artifacts":[{"id":11,"name":"ci-validation-certificate-proof","expired":false,"created_at":"2026-01-01T00:00:00Z"}]}]\n'
    else
      printf '[{"artifacts":[{"id":11,"name":"ci-validation-certificate-proof","expired":false,"created_at":"2026-01-01T00:00:00Z"},{"id":12,"name":"ci-test-analysis-proof","expired":false,"created_at":"2026-01-01T00:00:00Z"},{"id":13,"name":"coverage-proof","expired":false,"created_at":"2026-01-01T00:00:00Z"}]}]\n'
    fi
    ;;
  *"/actions/artifacts/11/zip"*) cat "$PROOF_CERTIFICATE_ZIP" ;;
  *"/actions/artifacts/12/zip"*) cat "$PROOF_ANALYSIS_ZIP" ;;
  *"/git/commits/"*) printf '%s\n' "$PROOF_TREE_SHA" ;;
  *) printf 'unexpected gh invocation: %s\n' "$args" >&2; exit 1 ;;
esac
'@
    $ghPath = Join-Path $bin 'gh'
    [IO.File]::WriteAllText($ghPath, $ghStub, [Text.UTF8Encoding]::new($false))
    $realJq = [string] (& $bash -lc 'command -v jq')
    $realJq = $realJq.Trim()
    if (-not $realJq) { throw 'jq is required for the validation reuse proof.' }
    $jqPath = Join-Path $bin 'jq'
    $jqStub = @'
#!/usr/bin/env bash
"$PROOF_REAL_JQ" "$@" | tr -d '\r'
exit "${PIPESTATUS[0]}"
'@
    [IO.File]::WriteAllText($jqPath, $jqStub, [Text.UTF8Encoding]::new($false))
    & $bash -lc "chmod +x '$($ghPath.Replace('\', '/'))' '$($jqPath.Replace('\', '/'))' '$($resolverScript.Replace('\', '/'))'"
    if ($LASTEXITCODE -ne 0) { throw 'Could not make the command fixtures executable.' }

    function New-ProofZip {
        param([string] $Directory, [string] $FileName, [string] $Json, [string] $ZipPath)
        if (Test-Path -LiteralPath $Directory) { Remove-Item -LiteralPath $Directory -Recurse -Force }
        New-Item -ItemType Directory -Path $Directory -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $Directory $FileName), $Json, [Text.UTF8Encoding]::new($false))
        if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
        Compress-Archive -LiteralPath (Join-Path $Directory $FileName) -DestinationPath $ZipPath
    }

    function Invoke-ReuseCase {
        param(
            [string] $Name,
            [bool] $RequiresValidation,
            [bool] $IncludeRuntimeArtifacts,
            [bool] $ExpectReuse
        )

        $testedSha = '1111111111111111111111111111111111111111'
        $certificateDirectory = Join-Path $fixture "$Name-certificate"
        $analysisDirectory = Join-Path $fixture "$Name-analysis"
        $certificateZip = Join-Path $fixture "$Name-certificate.zip"
        $analysisZip = Join-Path $fixture "$Name-analysis.zip"
        $certificate = [ordered]@{
            schemaVersion = 2
            runId = '123'
            testedSha = $testedSha
            testedTree = 'proof-tree'
            event = 'pull_request'
            requiresValidation = $RequiresValidation
            ciGateConclusion = 'success'
        } | ConvertTo-Json -Compress
        $analysis = [ordered]@{ source = [ordered]@{ commitSha = $testedSha } } |
            ConvertTo-Json -Depth 3 -Compress
        New-ProofZip -Directory $certificateDirectory -FileName validation-certificate.json `
            -Json $certificate -ZipPath $certificateZip
        New-ProofZip -Directory $analysisDirectory -FileName ci-test-analysis.json `
            -Json $analysis -ZipPath $analysisZip

        $output = Join-Path $fixture "$Name-output.txt"
        $summary = Join-Path $fixture "$Name-summary.md"
        $ghLog = Join-Path $fixture "$Name-gh.log"
        $environmentNames = @(
            'GITHUB_EVENT_NAME', 'GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_OUTPUT',
            'GITHUB_STEP_SUMMARY', 'RUNNER_TEMP', 'GH_TOKEN', 'REUSE_WAIT_MINUTES',
            'PROOF_HEAD_SHA', 'PROOF_TREE_SHA', 'PROOF_MODE', 'PROOF_CERTIFICATE_ZIP',
            'PROOF_ANALYSIS_ZIP', 'PROOF_GH_LOG', 'PROOF_REAL_JQ', 'PROOF_BIN'
        )
        $savedEnvironment = @{}
        foreach ($variableName in $environmentNames) {
            $savedEnvironment[$variableName] = [Environment]::GetEnvironmentVariable(
                $variableName, [EnvironmentVariableTarget]::Process)
        }
        try {
            $env:GITHUB_EVENT_NAME = 'push'
            $env:GITHUB_REPOSITORY = 'ooples/AiDotNet'
            $env:GITHUB_SHA = '2222222222222222222222222222222222222222'
            $env:GITHUB_OUTPUT = $output.Replace('\', '/')
            $env:GITHUB_STEP_SUMMARY = $summary.Replace('\', '/')
            $env:RUNNER_TEMP = $runnerTemp.Replace('\', '/')
            $env:GH_TOKEN = 'fixture-token'
            $env:REUSE_WAIT_MINUTES = '0'
            $env:PROOF_HEAD_SHA = '3333333333333333333333333333333333333333'
            $env:PROOF_TREE_SHA = 'proof-tree'
            $env:PROOF_MODE = $(if ($IncludeRuntimeArtifacts) { 'runtime' } else { 'non-runtime' })
            $env:PROOF_CERTIFICATE_ZIP = $certificateZip.Replace('\', '/')
            $env:PROOF_ANALYSIS_ZIP = $analysisZip.Replace('\', '/')
            $env:PROOF_GH_LOG = $ghLog.Replace('\', '/')
            $env:PROOF_REAL_JQ = $realJq
            $env:PROOF_BIN = $binPosix
            $resolverOutput = @(& $bash -c 'export PATH="$PROOF_BIN:$PATH"; exec "$1"' `
                proof $resolverScript.Replace('\', '/') 2>&1)
            if ($LASTEXITCODE -ne 0) { [void] $failures.Add("$Name resolver exited $LASTEXITCODE") }
        }
        finally {
            foreach ($variableName in $environmentNames) {
                [Environment]::SetEnvironmentVariable(
                    $variableName, $savedEnvironment[$variableName], [EnvironmentVariableTarget]::Process)
            }
        }

        $outputs = if (Test-Path -LiteralPath $output) { Get-Content -LiteralPath $output -Raw } else { '' }
        $summaryText = if (Test-Path -LiteralPath $summary) { Get-Content -LiteralPath $summary -Raw } else { '' }
        $ghCalls = if (Test-Path -LiteralPath $ghLog) { (Get-Content -LiteralPath $ghLog) -join ' | ' } else { '' }
        $reused = $outputs -match '(?m)^reuse=true$'
        if ($reused -ne $ExpectReuse) {
            [void] $failures.Add("$Name expected reuse=$ExpectReuse, got reuse=$reused; resolver output: $($resolverOutput -join ' | '); summary: $summaryText; outputs: $outputs; gh: $ghCalls")
        }
        if ($ExpectReuse) {
            $expectedScope = $RequiresValidation.ToString().ToLowerInvariant()
            if ($outputs -notmatch "(?m)^reused_requires_validation=$expectedScope$") {
                [void] $failures.Add("$Name did not publish reused_requires_validation=$expectedScope")
            }
        }
    }

    Invoke-ReuseCase -Name non_runtime_without_test_artifacts -RequiresValidation $false `
        -IncludeRuntimeArtifacts $false -ExpectReuse $true
    Invoke-ReuseCase -Name runtime_with_test_artifacts -RequiresValidation $true `
        -IncludeRuntimeArtifacts $true -ExpectReuse $true
    Invoke-ReuseCase -Name runtime_without_test_artifacts -RequiresValidation $true `
        -IncludeRuntimeArtifacts $false -ExpectReuse $false
}
catch {
    $fixtureFailure = $_
    throw
}
finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    if ($resolvedFixture.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedFixture).StartsWith('aidotnet-validation-reuse-', [StringComparison]::Ordinal)) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force -ErrorAction SilentlyContinue
    }
    elseif ($null -eq $fixtureFailure) { throw "refusing to remove unexpected fixture path '$resolvedFixture'" }
}

if ($failures.Count -gt 0) {
    Write-Host 'Validation reuse mode proof FAILED:'
    foreach ($failure in $failures) { Write-Host "  - $failure" }
    exit 1
}

Write-Host 'Validation reuse mode proof passed (non-runtime certificate reuses without test artifacts; runtime certificate requires them).'
exit 0
