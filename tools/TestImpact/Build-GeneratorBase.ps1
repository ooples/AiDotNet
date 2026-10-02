<#
.SYNOPSIS
  Builds the merge base's test assemblies when a pull request changes a source generator, so TypeImpact can
  compare the generated output of the two builds.

.DESCRIPTION
  A generator reaches runtime only through what it emits. TypeImpact --base-bin compares the content hashes of
  the generated documents in the base and head PDBs and maps each changed output like a changed source; without
  a base build every generator change leaves the plan unresolved and the pull request runs the full matrix.

  Runs only when Select-AffectedTests would narrow at all and the merge commit touches src/AiDotNet.Generators/.
  On success it exports TYPE_IMPACT_BASE_BIN and TYPE_IMPACT_BASE_REPO through GITHUB_ENV. On any failure it
  exports nothing, and the generator change stays unmappable: the full selection, as before.
#>
[CmdletBinding()]
param(
    [string] $Repository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string] $GeneratorRoot = 'src/AiDotNet.Generators/',
    [string] $WorkRoot = $(if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() })
)

$ErrorActionPreference = 'Stop'
if ($env:GITHUB_EVENT_NAME -ne 'pull_request') { Write-Host 'not a pull request: no base build'; exit 0 }
if ($env:REQUIRES_SHARDS -ne 'true') { Write-Host 'no test shards were required: no base build'; exit 0 }
if ($env:COLLECT_COVERAGE_EVERYWHERE -eq 'true') { Write-Host 'coverage-everywhere run: no base build'; exit 0 }

Push-Location $Repository
try {
    # Same base as Select-AffectedTests: the merge commit's first parent, the base branch actually tested.
    $parents = @(((& git rev-list --parents -n 1 HEAD) -join ' ').Split(' ', [StringSplitOptions]::RemoveEmptyEntries))
    if ($LASTEXITCODE -ne 0 -or $parents.Count -ne 3) { Write-Host '::warning::HEAD is not a merge commit: no base build'; exit 0 }
    $touched = @(& git -c core.quotepath=false diff --name-only $parents[1] HEAD -- $GeneratorRoot)
    if ($LASTEXITCODE -ne 0) { Write-Host '::warning::git diff failed: no base build'; exit 0 }
    if ($touched.Count -eq 0) { Write-Host "no change under $GeneratorRoot`: no base build"; exit 0 }
    Write-Host "$($touched.Count) generator file(s) changed; building the merge base $($parents[1]) to compare generated output"

    $base = Join-Path $WorkRoot 'generator-base'
    & git worktree add --detach $base $parents[1]
    if ($LASTEXITCODE -ne 0) { Write-Host '::warning::could not check out the merge base: no base build'; exit 0 }
}
finally {
    Pop-Location
}

$project = Join-Path $base 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
& dotnet build $project -c Release -f net10.0 -m:4
if ($LASTEXITCODE -ne 0) { Write-Host '::warning::the merge base did not build: generator changes stay unmappable'; exit 0 }

$bin = Join-Path $base 'tests/AiDotNet.Tests/bin/Release/net10.0'
if (-not (Test-Path -LiteralPath (Join-Path $bin 'AiDotNetTests.pdb'))) {
    Write-Host '::warning::the merge base build has no AiDotNetTests.pdb: generator changes stay unmappable'
    exit 0
}

# TypeImpact reads only the base's assemblies and PDBs; the intermediate output is several GB on a runner
# whose disk the head build has already filled.
Get-ChildItem -LiteralPath $base -Directory -Recurse -Filter obj -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }

"TYPE_IMPACT_BASE_BIN=$bin" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
"TYPE_IMPACT_BASE_REPO=$base" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
Write-Host "merge base built: $bin"
exit 0
