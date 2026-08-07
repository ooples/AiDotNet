<#
.SYNOPSIS
  Fails when a generated ModelFamily test class is not covered by any shard filter.

.DESCRIPTION
  The generated-layer shards select classes with hand-picked prefixes
  (Generated.A, Generated.B, ... Generated.Y|Generated.Z). That works only while
  someone re-verifies every name after each generator change: a class added later
  under an unlisted letter matches no shard, never runs, and NOTHING reports it.
  A test that silently stops running is worse than a failing one -- the pipeline
  stays green and the coverage loss is invisible.

  So the letters are checked against reality instead of maintained by hand. The
  generated classes do not exist on disk (the source generator emits them at build
  time), so the list comes from `dotnet test --list-tests` against the built
  assembly, and every distinct first letter must appear in some shard filter.

  Reports EVERY uncovered letter before failing, so one CI run tells the whole
  story rather than one letter per run.
#>
[CmdletBinding()]
param(
    [string]$Workflow = '.github/workflows/sonarcloud.yml',
    [string]$Project  = 'tests/AiDotNet.Tests/AiDotNetTests.csproj',
    [string]$Framework = 'net10.0',
    # Accepts a pre-captured listing so CI does not pay for a second discovery run
    # and so this script is testable without a build.
    [string]$ListingFile
)

$ErrorActionPreference = 'Stop'

if ($ListingFile) {
    if (-not (Test-Path $ListingFile)) { throw "listing file not found: $ListingFile" }
    $listing = Get-Content $ListingFile
} else {
    Write-Host "Discovering tests in $Project ($Framework)..."
    $listing = & dotnet test $Project -f $Framework --nologo --list-tests 2>&1
    # A discovery that did not run is not an empty result set. Failing here keeps
    # "no generated tests exist" from reading as "every letter is covered".
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet test --list-tests failed (exit $LASTEXITCODE); cannot verify shard coverage"
    }
}

$generated = $listing |
    Select-String -Pattern 'ModelFamilyTests\.Generated\.([A-Za-z])' -AllMatches |
    ForEach-Object { $_.Matches } |
    ForEach-Object { $_.Groups[1].Value.ToUpperInvariant() } |
    Sort-Object -Unique

if (-not $generated -or $generated.Count -eq 0) {
    throw 'No ModelFamilyTests.Generated.* tests were discovered. Either the generator produced nothing or discovery is misconfigured; both need a human, and neither should pass silently.'
}

$workflowText = Get-Content $Workflow -Raw
$uncovered = @()
foreach ($letter in $generated) {
    if ($workflowText -notmatch [regex]::Escape("Generated.$letter")) {
        $uncovered += $letter
    }
}

Write-Host ("Generated first-letters discovered: " + ($generated -join ', '))

if ($uncovered.Count -gt 0) {
    Write-Host "::error::Generated test classes are not covered by any shard filter: $($uncovered -join ', ')"
    Write-Host "Add these to a shard filter in $Workflow, or extend the last shard of the range to a tail filter."
    exit 1
}

Write-Host 'Shard coverage OK: every generated first-letter is matched by a shard filter.'
exit 0
