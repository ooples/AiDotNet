<#
.SYNOPSIS
  Fails when a generated ModelFamily test class is not covered by any shard filter.

.DESCRIPTION
  The generated-layer shards select classes with hand-picked prefixes
  (Generated.A, Generated.B, ... Generated.MedC, Generated.SAL). That works only
  while someone re-verifies every name after each generator change: a class added
  later under an unlisted prefix matches no shard, never runs, and NOTHING reports
  it. A test that silently stops running is worse than a failing one -- the
  pipeline stays green and the coverage loss is invisible.

  So the prefixes are checked against reality instead of maintained by hand. The
  generated classes do not exist on disk (the source generator emits them at build
  time), so the list comes from `dotnet test --list-tests` against the built
  assembly.

  CLASS NAMES, NOT FIRST LETTERS. An earlier revision reduced each discovered class
  to its first letter and asked whether the workflow text contained
  "Generated.<letter>" anywhere. That cannot detect the loss this script exists to
  prevent: M is covered only by Generated.M2, Generated.MA, Generated.Mea and
  friends, and "Generated.M" is a SUBSTRING of "Generated.MA", so the letter check
  passed while a class named Mzeta* matched no shard at all. The same held for S,
  P, and R. Every discovered class name is now matched against the actual shard
  prefixes parsed out of the workflow.

  Matching is ordinal (case-sensitive) because that is what VSTest's `~` operator
  does. The existing filter set relies on it -- Generated.MEG, Generated.Mel and
  Generated.Mem are three distinct shards that a case-insensitive match would
  collapse.

  Reports EVERY uncovered class before failing, so one CI run tells the whole story
  rather than one class per run.
#>
[CmdletBinding()]
param(
    [string]$Workflow = '.github/workflows/sonarcloud.yml',
    [string]$Project  = 'tests/AiDotNet.Tests/AiDotNetTests.csproj',
    [string]$Framework = 'net10.0',
    # THE CONFIGURATION THE SHARDS ACTUALLY RUN. Without this `dotnet test` defaults
    # to Debug and compiles the whole generated-source-heavy solution a second time,
    # in the same job the workflow already documents as reaching the 16-GB ceiling
    # and getting a compiler killed with exit 137. It also answered for the wrong
    # artifact: a generator condition that differs under Debug produces a coverage
    # verdict that does not describe the assembly the shards test.
    [string]$Configuration = 'Release',
    # Accepts a pre-captured listing so CI does not pay for a second discovery run
    # and so this script is testable without a build.
    [string]$ListingFile
)

$ErrorActionPreference = 'Stop'

if ($ListingFile) {
    if (-not (Test-Path $ListingFile)) { throw "listing file not found: $ListingFile" }
    $listing = Get-Content $ListingFile
} else {
    Write-Host "Discovering tests in $Project ($Framework, $Configuration)..."
    # --no-build: the caller has already built this configuration. Rebuilding is the
    # memory blowup described above, and it is also what let Debug slip in.
    $listing = & dotnet test $Project -f $Framework -c $Configuration --no-build --nologo --list-tests 2>&1
    # A discovery that did not run is not an empty result set. Failing here keeps
    # "no generated tests exist" from reading as "every class is covered".
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet test --list-tests failed (exit $LASTEXITCODE); cannot verify shard coverage"
    }
}

$generated = $listing |
    Select-String -Pattern 'ModelFamilyTests\.Generated\.([A-Za-z0-9_]+)' -AllMatches |
    ForEach-Object { $_.Matches } |
    ForEach-Object { $_.Groups[1].Value } |
    Sort-Object -Unique

if (-not $generated -or @($generated).Count -eq 0) {
    throw 'No ModelFamilyTests.Generated.* tests were discovered. Either the generator produced nothing or discovery is misconfigured; both need a human, and neither should pass silently.'
}

$workflowText = Get-Content $Workflow -Raw

# Every prefix the workflow actually filters on, with the shared "Generated." head
# removed so what remains is compared against the class name.
# -CaseSensitive IS LOAD-BEARING. Sort-Object -Unique is case-INSENSITIVE by default, so it
# collapsed Generated.MA and Generated.Ma into one entry and kept whichever sorted first. The
# comparison below is StartsWith(..., Ordinal), which IS case-sensitive -- so a prefix deleted by
# the dedup could never match, and adding the correctly-cased prefix to the workflow did nothing
# because the dedup ate it before the comparison ran. Measured: 175 prefixes in the workflow
# reported as 144, one FEWER than before the correctly-cased ones were added.
#
# This is the same distinction the header above depends on: Generated.MEG, Generated.Mel and
# Generated.Mem are three separate shards, and a case-insensitive dedup conflates them.
$prefixes = [regex]::Matches($workflowText, 'FullyQualifiedName~Generated\.([A-Za-z0-9_]+)') |
    ForEach-Object { $_.Groups[1].Value } |
    Sort-Object -Unique -CaseSensitive

if (-not $prefixes -or @($prefixes).Count -eq 0) {
    throw "No FullyQualifiedName~Generated.* shard filters were found in $Workflow; the gate has nothing to check against."
}

$uncovered = @()
foreach ($class in $generated) {
    $covered = $false
    foreach ($prefix in $prefixes) {
        if ($class.StartsWith($prefix, [System.StringComparison]::Ordinal)) {
            $covered = $true
            break
        }
    }
    if (-not $covered) { $uncovered += $class }
}

Write-Host ("Generated classes discovered: " + @($generated).Count + "; shard prefixes in workflow: " + @($prefixes).Count)

if ($uncovered.Count -gt 0) {
    Write-Host "::error::Generated test classes are not covered by any shard filter: $($uncovered -join ', ')"
    Write-Host "Add a matching prefix to a shard filter in $Workflow, or extend the last shard of the range to a tail filter."
    exit 1
}

Write-Host "Shard coverage OK: all $(@($generated).Count) generated classes are matched by a shard filter."
exit 0
