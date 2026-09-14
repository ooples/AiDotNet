<#
.SYNOPSIS
    Exercises the production delta-reuse emission branch against incomplete artifact inventories.
.DESCRIPTION
    Git/selection is covered by Test-TestImpactEndToEnd. This fixture substitutes only that plan
    and GitHub's artifact inventory, then executes the checked-in decision/output branch and its
    real helpers in a child process. No GitHub calls, workflow runs, or remote artifacts are needed.
#>
[CmdletBinding()]
param([string] $ResolverPath = "$PSScriptRoot/Resolve-CiValidationReuse.ps1")

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'ReviewFixtureCleanup.ps1')

enum InvalidArtifactCase {
    Expired
    WrongSha
    WrongSlug
    WrongCase
}

enum DeltaEmissionExpectation {
    Decline
    PartialWithImports
    PartialWithoutImports
    WholeResultReuse
}

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($resolverPath, [ref] $tokens, [ref] $parseErrors)
if ($parseErrors.Count -gt 0) { throw 'The reuse resolver must parse before its production branch is tested.' }
$declarations = @($ast.EndBlock.Statements | Where-Object {
    $_ -is [System.Management.Automation.Language.FunctionDefinitionAst] -or
    $_ -is [System.Management.Automation.Language.TypeDefinitionAst]
} | ForEach-Object { $_.Extent.Text })
$branches = @($ast.EndBlock.Statements | Where-Object {
    $_ -is [System.Management.Automation.Language.IfStatementAst] -and
    $_.Extent.Text.StartsWith('if ($deltaCandidates.Count', [StringComparison]::Ordinal)
})
if ($branches.Count -ne 1) { throw 'The fixture must execute exactly one production delta-emission branch.' }

$sha = '0123456789abcdef0123456789abcdef01234567'
$otherSha = '1123456789abcdef0123456789abcdef01234567'
$valid = @(
    [pscustomobject]@{ name = "test-results-$sha-Integration_E_G"; expired = $false },
    [pscustomobject]@{ name = "coverage-$sha-Integration_E_G"; expired = $false },
    [pscustomobject]@{ name = "test-results-$sha-Unit_10_RL"; expired = $false },
    [pscustomobject]@{ name = "coverage-$sha-Unit_10_RL"; expired = $false }
)
$cases = [System.Collections.Generic.List[object]]::new()
[void] $cases.Add([pscustomobject]@{ Name = 'complete-imports'; Artifacts = $valid
    Expectation = [DeltaEmissionExpectation]::PartialWithImports; Delta = @('Integration D') })
foreach ($missing in 0..3) {
    [void] $cases.Add([pscustomobject]@{ Name = "missing-$missing"
        Expectation = [DeltaEmissionExpectation]::Decline; Delta = @('Integration D')
        Artifacts = @(for ($i = 0; $i -lt $valid.Count; $i++) { if ($i -ne $missing) { $valid[$i] } }) })
}
foreach ($invalid in [Enum]::GetValues([InvalidArtifactCase])) {
    $artifacts = @($valid | ForEach-Object { $_.PSObject.Copy() })
    switch ($invalid) {
        ([InvalidArtifactCase]::Expired) { $artifacts[0].expired = $true }
        ([InvalidArtifactCase]::WrongSha) { $artifacts[0].name = "test-results-$otherSha-Integration_E_G" }
        ([InvalidArtifactCase]::WrongSlug) { $artifacts[0].name = "test-results-$sha-Integration_D" }
        ([InvalidArtifactCase]::WrongCase) { $artifacts[0].name = "test-results-$sha-integration_e_g" }
    }
    [void] $cases.Add([pscustomobject]@{ Name = $invalid.ToString(); Artifacts = $artifacts
        Expectation = [DeltaEmissionExpectation]::Decline; Delta = @('Integration D') })
}
[void] $cases.Add([pscustomobject]@{ Name = 'all-shards-rerun'; Artifacts = @()
    Expectation = [DeltaEmissionExpectation]::PartialWithoutImports
    Delta = @('Integration D', 'Integration E-G', 'Unit - 10 RL') })
[void] $cases.Add([pscustomobject]@{ Name = 'all-shards-rerun-with-unused-artifacts'; Artifacts = $valid
    Expectation = [DeltaEmissionExpectation]::PartialWithoutImports
    Delta = @('Integration D', 'Integration E-G', 'Unit - 10 RL') })
[void] $cases.Add([pscustomobject]@{ Name = 'full-reuse-preserved'; Artifacts = @()
    Expectation = [DeltaEmissionExpectation]::WholeResultReuse; Delta = @('Unrelated') })

$setup = @'
param([string] $CasePath, [string] $GitHubOutput, [string] $MapFile, [string] $ShardManifestFile, [string] $FixtureSha)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:GITHUB_STEP_SUMMARY = ''
'@
$fixturePlan = @'
$case = Get-Content -LiteralPath $CasePath -Raw | ConvertFrom-Json
$prNumber = 42
$candidateEvidence = [pscustomobject]@{
    RunId = 123; TestedSha = $FixtureSha
    RequiresValidation = $true; ArtifactNames = @(Get-UnexpiredArtifactNames -Artifacts @($case.Artifacts))
}
$deltaCandidates = @($candidateEvidence)
$fixtureDecision = Get-DeltaReuseDecision -SelectionEscalated $false -SelectionRequiresValidation $true `
    -DeltaShards @($case.Delta) -PullRequestShards @('Integration D', 'Integration E-G', 'Unit - 10 RL')
$fixturePlan = [pscustomobject]@{
    Decision = $fixtureDecision; Tree = 'fixture-tree'
    Selection = [pscustomobject]@{ routes = @(); reasons = @() }
}
function Get-DeltaPlan { param($Candidate) return $fixturePlan }
'@
$fallback = 'Write-ReuseDecision -DecisionScope None -PrNumber $prNumber'
$scriptText = @($setup) + $declarations + @($fixturePlan, $branches[0].Extent.Text, $fallback)
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $tempRoot ('aidotnet-reuse-review-' + [guid]::NewGuid().ToString('N'))
$failures = [System.Collections.Generic.List[string]]::new()
try {
    New-Item -ItemType Directory -Path $fixture | Out-Null
    $scriptPath = Join-Path $fixture 'production-emission.ps1'
    $mapPath = Join-Path $fixture 'map.json'
    $manifestPath = Join-Path $fixture 'manifest.json'
    Set-Content -LiteralPath $scriptPath -Value ($scriptText -join [Environment]::NewLine) -Encoding utf8
    '{}' | Set-Content -LiteralPath $mapPath -Encoding utf8
    '[]' | Set-Content -LiteralPath $manifestPath -Encoding utf8
    foreach ($case in $cases) {
        $casePath = Join-Path $fixture ($case.Name + '.json')
        $outputPath = Join-Path $fixture ($case.Name + '.outputs')
        $case | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $casePath -Encoding utf8
        $output = @(& pwsh -NoProfile -File $scriptPath -CasePath $casePath -GitHubOutput $outputPath `
            -MapFile $mapPath -ShardManifestFile $manifestPath -FixtureSha $sha 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "$($case.Name): production emission failed: $($output -join [Environment]::NewLine)" }
        $values = @{}
        foreach ($line in Get-Content -LiteralPath $outputPath) {
            $parts = $line -split '=', 2
            $values[$parts[0]] = $parts[1]
        }
        if ($case.Expectation -eq [DeltaEmissionExpectation]::Decline) {
            if (($output -join [Environment]::NewLine) -notmatch 'delta reuse declined:') {
                [void] $failures.Add("$($case.Name): production decline path was not observed: $($output -join [Environment]::NewLine)")
            }
            if ($values.delta_mode -or $values.import_run_id -or $values.import_sha -or
                $values.import_shards -cne '[]' -or $values.partial_shards -cne '[]' -or
                $values.execute_validation -cne 'true' -or $values.reuse_scope -cne 'None') {
                [void] $failures.Add("$($case.Name): missing imported evidence still authorized reuse: $($values | ConvertTo-Json -Compress)")
            }
        }
        elseif ($case.Expectation -eq [DeltaEmissionExpectation]::WholeResultReuse) {
            if ($values.delta_mode -cne 'Reuse' -or $values.reuse_scope -cne 'Validation' -or
                $values.execute_validation -cne 'false' -or $values.execute_quality -cne 'true') {
                [void] $failures.Add('whole-result delta reuse changed despite not importing per-shard artifacts')
            }
        }
        else {
            $expectedImport = if ($case.Expectation -eq [DeltaEmissionExpectation]::PartialWithoutImports) { '[]' } else { '["Integration E-G","Unit - 10 RL"]' }
            $expectedImportRunId = if ($case.Expectation -eq [DeltaEmissionExpectation]::PartialWithoutImports) { '' } else { '123' }
            $expectedImportSha = if ($case.Expectation -eq [DeltaEmissionExpectation]::PartialWithoutImports) { '' } else { $sha }
            $expectedRerun = if ($case.Expectation -eq [DeltaEmissionExpectation]::PartialWithoutImports) {
                '["Integration D","Integration E-G","Unit - 10 RL"]'
            } else { '["Integration D"]' }
            if ($values.delta_mode -cne 'Partial' -or $values.import_run_id -cne $expectedImportRunId -or
                $values.import_sha -cne $expectedImportSha -or $values.import_shards -cne $expectedImport -or
                $values.partial_shards -cne $expectedRerun -or
                $values.execute_validation -cne 'true' -or $values.execute_quality -cne 'true') {
                [void] $failures.Add("$($case.Name): valid partial reuse was not preserved: actual=$($values | ConvertTo-Json -Compress); expected import=$expectedImport run=$expectedImportRunId sha=$expectedImportSha rerun=$expectedRerun")
            }
        }
    }
}
finally {
    Remove-ReviewFixtureDirectory -LiteralPath $fixture -ExpectedLeafPrefix 'aidotnet-reuse-review-'
}
if ($failures.Count -gt 0) {
    foreach ($failure in $failures) { Write-Host $failure }
    exit 1
}
Write-Host "Delta artifact review controls passed: $($cases.Count) production emission cases."
exit 0
