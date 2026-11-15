<#
.SYNOPSIS
    The rolling map's part of a selection: select with it beside the weekly map, compare, and decide which
    of the two selections the matrix uses. Also, after the matrix: what the rolling map would have dropped.

.DESCRIPTION
    SELECT (default parameter set). Called by the Select step right after the certified weekly map (the
    anchor) has produced AnchorSelectionFile, and only when repository variable TEST_IMPACT_ROLLING_MAP is
    shadow or on.

      shadow  The anchor's selection file is never opened for writing. The rolling selection is written
              beside it and both are compared in ComparisonFile.
      on      The rolling selection replaces AnchorSelectionFile (the anchor's is kept as AnchorCopyFile)
              only when the rolling map is certified for the anchor in use, its selector run succeeded, and
              the two agree on whether the change needs validation and shards at all.

    Every other outcome - no rolling artifact on the anchor's run, a certificate that does not match the map
    bytes or the anchor, a digest past the age limit, a selector failure - leaves the anchor's selection in
    place, so a rolling map that cannot be trusted never selects less than the certified weekly map does.
    The anchor itself was validated by the caller before this runs; when the certified-map resolver refuses
    the anchor (revoked, unclassified, expired) there is no anchor selection and this script is not reached.

    OUTCOME (-OutcomeComparisonFile). After the matrix: which of the shards the rolling map would not have
    run failed. In shadow mode they all ran, so a failure among them is a regression the rolling map would
    have let through; it is reported as a warning and in the summary. Report-only.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [AllowEmptyString()] [string] $Mode,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $AnchorMapFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [long] $AnchorRunId,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $RollingDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $AnchorSelectionFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string[]] $ExpectedShards,
    [Parameter(ParameterSetName = 'Select')] [string] $ShardManifestFile,
    [Parameter(ParameterSetName = 'Select')] [string] $PullRequestHeadSha,
    [Parameter(ParameterSetName = 'Select')] [string] $MapLineageRef,
    [Parameter(ParameterSetName = 'Select')] [string] $RollingSelectionFile = 'selection-rolling.json',
    [Parameter(ParameterSetName = 'Select')] [string] $AnchorCopyFile = 'selection-anchor.json',
    [Parameter(ParameterSetName = 'Select')] [string] $ComparisonFile = 'rolling-comparison.json',
    [Parameter(ParameterSetName = 'Select')] [string] $SelectorPath = "$PSScriptRoot/Select-Shards.ps1",
    [Parameter(Mandatory, ParameterSetName = 'Outcome')] [string] $OutcomeComparisonFile,
    [Parameter(Mandatory, ParameterSetName = 'Outcome')] [string] $JobsFile,
    [Parameter(ParameterSetName = 'Outcome')] [string] $OutcomeFile = 'rolling-shadow-outcome.json',
    [Parameter(ParameterSetName = 'Outcome')] [string] $SummaryFile = $env:GITHUB_STEP_SUMMARY
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RollingShardMap.ps1"

if ($PSCmdlet.ParameterSetName -ceq 'Outcome') {
    $decision = Get-Content -LiteralPath $OutcomeComparisonFile -Raw | ConvertFrom-Json
    $outcomes = @{}
    foreach ($job in @(Get-Content -LiteralPath $JobsFile -Raw | ConvertFrom-Json)) {
        $name = [string] $job.name
        if (-not $name.StartsWith('Tests (', [StringComparison]::Ordinal)) { continue }
        $split = $name.IndexOf(') - ', [StringComparison]::Ordinal)
        if ($split -ge 0) { $outcomes[$name.Substring($split + 4)] = [string] $job.conclusion }
    }
    $outcome = Get-RollingShadowOutcome -Decision $decision -Outcomes $outcomes
    $outcome | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $OutcomeFile -Encoding utf8
    $lines = [System.Collections.Generic.List[string]]::new()
    [void] $lines.Add("### Rolling coverage map: outcome ($($outcome.mode) mode)")
    [void] $lines.Add('')
    if (@($outcome.dropped).Count -eq 0) {
        [void] $lines.Add('The rolling map would have run every shard the weekly map ran.')
    }
    elseif ($outcome.decides -ceq 'rolling') {
        [void] $lines.Add("The rolling map decided this matrix and left out $(@($outcome.dropped).Count) shard(s) the weekly map would have run. They did not run, so they have no outcome here; the nightly full run is their check.")
    }
    elseif (@($outcome.droppedThenFailed).Count -gt 0) {
        [void] $lines.Add("**$(@($outcome.droppedThenFailed).Count) shard(s) the rolling map would NOT have run failed in this run:**")
        [void] $lines.Add('')
        foreach ($name in $outcome.droppedThenFailed) { [void] $lines.Add("- $name") }
        [void] $lines.Add('')
        [void] $lines.Add('Each is either a regression the rolling map would have let through or a failure unrelated to the change (flaky, pre-existing); either way it needs a look before the rolling map is turned on.')
        Write-Host "::warning::rolling map shadow: $(@($outcome.droppedThenFailed).Count) shard(s) it would have dropped failed: $(@($outcome.droppedThenFailed) -join ', ')"
    }
    else {
        $notRun = @($outcome.droppedNotRun)
        $ran = @($outcome.dropped).Count - $notRun.Count
        [void] $lines.Add("The rolling map would not have run $(@($outcome.dropped).Count) shard(s) the weekly map selected. $ran ran and none of them failed.")
        if ($notRun.Count -gt 0) {
            [void] $lines.Add("$($notRun.Count) of them have no outcome in this run, so they are no evidence: $($notRun -join ', ')")
        }
    }
    if ($SummaryFile) { Add-Content -LiteralPath $SummaryFile -Value ($lines -join "`n") -Encoding utf8 }
    Write-Host ($lines -join "`n")
    exit 0
}

$resolvedMode = Get-RollingMapMode $Mode
if ($resolvedMode -ceq 'off') {
    Write-Host 'rolling map: off'
    exit 0
}

$anchorSelection = Get-Content -LiteralPath $AnchorSelectionFile -Raw | ConvertFrom-Json
$mapFile = Join-Path $RollingDirectory 'rolling-shard-map.json'
$certificateFile = Join-Path $RollingDirectory 'rolling-certification.json'
$problem = ''
try {
    if (-not (Test-Path -LiteralPath $mapFile) -or -not (Test-Path -LiteralPath $certificateFile)) {
        $refused = Join-Path $RollingDirectory 'rolling-decision.json'
        $why = if (Test-Path -LiteralPath $refused) { " ($([string] (Get-Content -LiteralPath $refused -Raw | ConvertFrom-Json).disposition))" } else { '' }
        throw "the anchor's map run $AnchorRunId published no certified rolling map$why"
    }
    Assert-CertifiedRollingShardMap -RollingMap (Get-Content -LiteralPath $mapFile -Raw | ConvertFrom-Json) `
        -RollingMapHash (Get-RollingFileHash -Path $mapFile) `
        -Certificate (Get-Content -LiteralPath $certificateFile -Raw | ConvertFrom-Json) `
        -AnchorMap (Get-Content -LiteralPath $AnchorMapFile -Raw | ConvertFrom-Json) -AnchorRunId $AnchorRunId
}
catch { $problem = $_.Exception.Message }

$rollingSelection = $null
$seconds = $null
if (-not $problem) {
    try {
        Remove-Item -LiteralPath $RollingSelectionFile -ErrorAction SilentlyContinue
        $arguments = @{ MapFile = $mapFile; ExpectedShards = $ExpectedShards; OutFile = $RollingSelectionFile }
        if ($ShardManifestFile) { $arguments.ShardManifestFile = $ShardManifestFile }
        if ($PullRequestHeadSha) { $arguments.PullRequestHeadSha = $PullRequestHeadSha }
        if ($MapLineageRef) { $arguments.MapLineageRef = $MapLineageRef }
        $watch = [Diagnostics.Stopwatch]::StartNew()
        & $SelectorPath @arguments
        $seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 1)
        if ($LASTEXITCODE -ne 0) { throw "the selector exited $LASTEXITCODE" }
        $rollingSelection = Get-Content -LiteralPath $RollingSelectionFile -Raw | ConvertFrom-Json
    }
    catch { $problem = "the rolling selection failed ($($_.Exception.Message))" }
}

$decision = Get-RollingSelectionDecision -Mode $resolvedMode -AnchorSelection $anchorSelection `
    -RollingSelection $rollingSelection -RollingProblem $problem -AllShards $ExpectedShards
$decision | Add-Member -NotePropertyName rollingSelectionSeconds -NotePropertyValue $seconds
$decision | Add-Member -NotePropertyName anchorRunId -NotePropertyValue $AnchorRunId
$decision | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ComparisonFile -Encoding utf8
Write-Host "rolling map ($resolvedMode): $($decision.reason)"
if ($null -ne $decision.comparison.rolling) {
    Write-Host "  weekly map: $($decision.comparison.anchor.shardCount) shard(s); rolling map: $($decision.comparison.rolling.shardCount); rolling would not run $(@($decision.comparison.droppedByRolling).Count), adds $(@($decision.comparison.addedByRolling).Count)"
}

if ($decision.decides -ceq 'rolling') {
    # The last thing done, and only here: the anchor's selection is kept, then replaced in one rename.
    Copy-Item -LiteralPath $AnchorSelectionFile -Destination $AnchorCopyFile -Force
    $staged = "$AnchorSelectionFile.rolling"
    Copy-Item -LiteralPath $RollingSelectionFile -Destination $staged -Force
    Move-Item -LiteralPath $staged -Destination $AnchorSelectionFile -Force
}
exit 0
