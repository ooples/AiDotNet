<#
.SYNOPSIS
    Answers the only question that decides whether test selection is safe: would it have skipped a
    shard that failed?

.DESCRIPTION
    Selection rests on one assumption - that a shard which executes none of the changed lines would
    have passed anyway. That is an assumption, not a proof, and nothing about a green PR reveals
    when it is wrong: the skipped shard leaves no evidence behind. A selector can be quietly wrong
    for months and every run still looks healthy.

    So it is measured against runs that DID execute everything. For a full run, replay the selection
    that would have been made, then intersect the shards it would have skipped with the shards that
    actually failed. Any shard in that intersection is a MISS - a regression selection would have
    let through.

    Historical audits use the map that was actually in effect, not the map built from the run being
    audited. The only exception is an explicitly labelled FreshCoverage bootstrap/re-seed: the
    certificate policy binds that candidate to the same source tree and complete shard universe,
    refuses it after a historical miss, and requires a genuinely reduced partition. That bootstrap
    does not replace later historical replay audits.

    A miss is never merely informational. A non-zero miss count is an exit code, because a miss
    rate that is only ever printed is a number nobody reads.

    A miss is not automatically a selector bug - a flaky or pre-existing failure in an unrelated
    shard counts as a miss here even though selection was right to skip it. That is intentional:
    the metric errs toward reporting too much, and the names are printed so a human can tell the
    two apart. A clean run of this over time is the evidence that selection is safe to rely on.

.PARAMETER MapFile
    For HistoricalReplay, the shard map that would have been in effect, identified by the source
    marker rather than guessed from chronology. FreshCoverage may supply the same-tree candidate;
    New-ShardMapCertificate.ps1 enforces the distinct provenance rules for those two typed bases.

.PARAMETER OutcomesFile
    JSON array of { shard, outcome } for every shard in the audited run. `outcome` is the shard's
    test step result; 'success' passes, 'failure' fails; any OTHER outcome (cancelled, skipped, neutral) aborts the audit, because a shard that never finished is not evidence in either direction.

.PARAMETER SelectorPath
    Select-Shards.ps1. Invoked as a subprocess so the audit exercises the SHIPPING selector rather
    than a copy of its logic that is free to drift from it. The audit-only unchanged-map switch is
    passed explicitly so an exact map/source tree still exercises a reduced selection; ordinary PR
    selection does not receive that capability and continues to fail closed on an empty diff.

.PARAMETER CurrentChangeBaseSha
    Optional base of the current change under audit. Map-to-HEAD source ranges are still measured
    from the map SHA; this second boundary prevents already-merged selector-control edits from
    being mistaken for edits in every later pull request.

.PARAMETER OutFile
    Optional path for the JSON report.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, ParameterSetName = 'Measure')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Measure')] [string] $OutcomesFile,
    [Parameter(ParameterSetName = 'Measure')] [string] $SelectorPath = "$PSScriptRoot/Select-Shards.ps1",
    [Parameter(ParameterSetName = 'Measure')] [string] $CurrentChangeBaseSha,
    [Parameter(ParameterSetName = 'Measure')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-SelectionMiss {
    <#
        Pure, so the arithmetic can be tested without a map, a git tree or a CI run. Every argument
        is a plain list of shard names.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $AllShards,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $WouldRun,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Failed,
        [Parameter(Mandatory)] [bool] $Escalated
    )

    # An escalated run skips nothing, so it can miss nothing. Counting it as a clean audit would
    # dilute the miss rate with runs that never exercised selection at all, so it is reported
    # separately instead.
    # Built by adding rather than by the collection constructor: an empty [string[]] arrives as
    # $null and the constructor throws "Value cannot be null". That is not a hypothetical - an
    # escalated run passes an empty WouldRun, which is the most common case there is.
    $run = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($s in $WouldRun) { [void] $run.Add([string] $s) }

    # Assigned in two statements, not as an if-expression. `$x = if (...) { @() }` yields $null,
    # because PowerShell unrolls the empty array on the way out - and $null.Count then throws under
    # StrictMode. The escalated branch is exactly that case.
    $wouldSkip = @()
    if (-not $Escalated) {
        $wouldSkip = @($AllShards | Where-Object { -not $run.Contains([string] $_) })
    }

    $skip = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
    foreach ($s in $wouldSkip) { [void] $skip.Add([string] $s) }
    $missed = @($Failed | Where-Object { $skip.Contains([string] $_) } | Sort-Object -Unique)

    $wouldRunShards = @($WouldRun)
    if ($Escalated) { $wouldRunShards = @($AllShards) }

    return [pscustomobject]@{
        Escalated      = $Escalated
        TotalShards    = $AllShards.Count
        WouldRun       = $wouldRunShards.Count
        WouldSkip      = $wouldSkip.Count
        WouldRunShards = $wouldRunShards
        WouldSkipShards = $wouldSkip
        Failed         = @($Failed).Count
        Missed         = $missed
        MissCount      = $missed.Count
    }
}

# ---------------------------------------------------------------- self-test

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) } }

    $all = @('A', 'B', 'C', 'D')

    # 1. A failure in a shard selection would have SKIPPED is a miss - the case the audit exists for.
    $r = Get-SelectionMiss -AllShards $all -WouldRun @('A', 'B') -Failed @('C') -Escalated $false
    Assert-True ($r.MissCount -eq 1) 'a failure in a skipped shard must be reported as a miss'
    Assert-True ($r.Missed -contains 'C') 'the missed shard must be named'
    Assert-True (($r.WouldRunShards -join ',') -eq 'A,B') 'selected shard identities were not preserved'
    Assert-True (($r.WouldSkipShards -join ',') -eq 'C,D') 'skipped shard identities were not preserved'

    # 2. A failure in a shard selection would have RUN is not a miss. Without this the audit could
    #    report every failure as a miss and still pass check 1, which would make it useless.
    $r = Get-SelectionMiss -AllShards $all -WouldRun @('A', 'B') -Failed @('A') -Escalated $false
    Assert-True ($r.MissCount -eq 0) 'a failure in a selected shard is not a miss'

    # 3. Mixed: only the skipped half counts.
    $r = Get-SelectionMiss -AllShards $all -WouldRun @('A', 'B') -Failed @('A', 'D') -Escalated $false
    Assert-True ($r.MissCount -eq 1 -and $r.Missed -contains 'D') 'only failures in skipped shards count'

    # 4. An escalated run skips nothing, so it cannot miss - even when shards failed.
    $r = Get-SelectionMiss -AllShards $all -WouldRun @() -Failed @('C', 'D') -Escalated $true
    Assert-True ($r.MissCount -eq 0) 'an escalated run cannot miss'
    Assert-True ($r.WouldSkip -eq 0) 'an escalated run skips nothing'
    Assert-True ($r.WouldRun -eq 4) 'an escalated run runs everything'
    Assert-True (($r.WouldRunShards -join ',') -eq 'A,B,C,D') 'an escalated run did not record the full selected universe'

    # 5. A green run misses nothing regardless of how much it skipped.
    $r = Get-SelectionMiss -AllShards $all -WouldRun @('A') -Failed @() -Escalated $false
    Assert-True ($r.MissCount -eq 0) 'no failures means no misses'
    Assert-True ($r.WouldSkip -eq 3) 'skipped count must still be reported'

    # 6. Ordinal comparison. Shard names differing only in case are different shards, and a
    #    case-insensitive match would silently treat a skipped shard as run.
    $r = Get-SelectionMiss -AllShards @('Alpha', 'alpha') -WouldRun @('Alpha') -Failed @('alpha') -Escalated $false
    Assert-True ($r.MissCount -eq 1) 'shard matching must be ordinal'

    if ($failures.Count -gt 0) {
        Write-Host 'Measure-SelectionMiss self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Measure-SelectionMiss self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- audit

$outcomes = @(Get-Content -LiteralPath $OutcomesFile -Raw | ConvertFrom-Json)
if ($outcomes.Count -eq 0) { throw "No shard outcomes in $OutcomesFile - nothing to audit." }

$allShards = [System.Collections.Generic.List[string]]::new()
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($outcome in $outcomes) {
    if (-not $outcome.PSObject.Properties['shard'] -or -not $outcome.PSObject.Properties['outcome']) {
        throw 'Every outcome must contain shard and outcome.'
    }
    $name = [string] $outcome.shard
    $conclusion = [string] $outcome.outcome
    if ([string]::IsNullOrWhiteSpace($name)) { throw 'Outcome shard names must be nonempty.' }
    if (-not $seen.Add($name)) { throw "Duplicate outcome for shard '$name'." }
    if ($conclusion -notin @('success', 'failure')) {
        throw "Shard '$name' has unauditable outcome '$conclusion'."
    }
    [void] $allShards.Add($name)
}
$failed = @($outcomes | Where-Object { [string] $_.outcome -ne 'success' } | ForEach-Object { [string] $_.shard })

$selectionFile = Join-Path ([System.IO.Path]::GetTempPath()) "selection-audit-$PID.json"
$selectorArguments = @{
    MapFile = $MapFile
    ExpectedShards = @($allShards)
    AuditUnchangedMap = $true
    OutFile = $selectionFile
}
if ($CurrentChangeBaseSha) {
    $selectorArguments.BaseSha = $CurrentChangeBaseSha
}
& $SelectorPath @selectorArguments | Out-Null
$selection = Get-Content -LiteralPath $selectionFile -Raw | ConvertFrom-Json
Remove-Item -LiteralPath $selectionFile -ErrorAction SilentlyContinue

$result = Get-SelectionMiss -AllShards $allShards `
                            -WouldRun @($selection.shards) `
                            -Failed $failed `
                            -Escalated ([bool] $selection.escalate)

if ($result.Escalated) {
    Write-Host "audit: selection would have ESCALATED ($($result.Failed) shard(s) failed) - nothing skipped, nothing to miss"
} else {
    Write-Host "audit: would run $($result.WouldRun)/$($result.TotalShards), skip $($result.WouldSkip); $($result.Failed) shard(s) failed"
}

if ($result.MissCount -gt 0) {
    Write-Host "::error::selection would have SKIPPED $($result.MissCount) shard(s) that failed:"
    foreach ($m in $result.Missed) { Write-Host "  MISSED: $m" }
} elseif ($result.Failed -eq 0) {
    Write-Host 'audit: complete clean matrix; zero observed selection misses'
} else {
    Write-Host 'audit: no misses'
}

if ($OutFile) { $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutFile -Encoding utf8 }

# The miss count is the exit code's whole point: a safety metric that only ever gets printed is a
# safety metric nobody reads.
if ($result.MissCount -gt 0) { exit 1 }
exit 0
