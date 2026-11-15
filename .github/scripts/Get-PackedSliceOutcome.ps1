<#
.SYNOPSIS
    Reports one slice of a packed job as its own step: passes when the slice's tests passed, fails otherwise.

.DESCRIPTION
    A packed job (repository variable TEST_IMPACT_REPACK=on; tools/TestImpact/ShardPacking.ps1) runs several
    shard slices in one test step, so the job's own conclusion no longer says which shard failed. The workflow
    runs this once per slice in a step named 'Packed slice N: <shard>'. That step's conclusion is the slice's,
    which is what a person reading the job sees and what the tools that read a run's jobs by shard name use
    (Expand-PackedShardJobs in ShardPacking.ps1, Resolve-CiValidationReuse.ps1).

    The outcome is the one Invoke-Shard.ps1 recorded in the slice's shard-metadata.json. A slice with no
    record, or one still 'cancelled' because the job ran out of time before reaching it, fails.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [ValidateRange(0, 63)] [int] $Slot,
    [string] $ResultsRoot = 'TestResults'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$file = Join-Path (Join-Path $ResultsRoot "slot-$Slot") 'shard-metadata.json'
if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
    Write-Host "::error::Packed slice $($Slot + 1) has no recorded outcome: the test step never started it."
    exit 1
}
try { $metadata = Get-Content -LiteralPath $file -Raw | ConvertFrom-Json }
catch {
    Write-Host "::error::Packed slice $($Slot + 1) has an unreadable outcome record: $($_.Exception.Message)"
    exit 1
}
$outcome = [string] $metadata.testStepOutcome
$seconds = if ($metadata.PSObject.Properties['testWallSeconds']) { ' in {0:N1} min' -f ([double] $metadata.testWallSeconds / 60) } else { '' }
Write-Host "$($metadata.shard): $outcome$seconds"
if ($outcome -ceq 'success') { exit 0 }
if ($outcome -ceq 'cancelled') { Write-Host "::error::Packed slice '$($metadata.shard)' did not run to completion." }
exit 1
