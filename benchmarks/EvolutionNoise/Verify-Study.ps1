param([Parameter(Mandatory = $true)][string] $Report)
$ErrorActionPreference = 'Stop'
$study = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
if ($study.Schema -ne 'us06-consumer-noise-study-v1' -or $study.Valid -ne $true -or $study.Runs.Count -ne 18) {
    throw 'Missing, failed or incomplete fixed consumer study.'
}
$expectedSeeds = '1103,2207,3301,4409,5519,6607'
if (($study.Seeds -join ',') -ne $expectedSeeds -or ($study.ScreeningThresholds -join ',') -ne '0.5,0.98') {
    throw 'The predeclared design changed.'
}
$fits = 0; $invocations = 0; $checks = 0; $cost = 0
foreach ($run in $study.Runs) {
    if ($run.Valid -ne $true -or $run.Screen.IsComplete -ne $true -or $run.Ledger.Unknown -ne 0 -or $run.Ledger.MaximumViolated) {
        throw 'A failed or unknown measurement cannot validate a preset.'
    }
    $batches = @($run.Screen.Entries | ForEach-Object { $_.Measurements })
    $batches += @($run.Screen.Audit.Entries | ForEach-Object { $_.FullEvaluation })
    foreach ($challenge in @($run.Challenge, $run.ReverseChallenge)) {
        if ($null -eq $challenge) { continue }
        $batches += @($challenge.CandidateSearch, $challenge.IncumbentSearch, $challenge.CandidateConfirmation, $challenge.IncumbentConfirmation) | Where-Object { $null -ne $_ }
    }
    $samples = @($batches | ForEach-Object { $_.Samples })
    if (@($samples | Group-Object { $_.Context.SampleIdentity } | Where-Object Count -gt 1).Count -ne 0) {
        throw 'A workflow reused a fitness sample identity.'
    }
    $charged = ($samples | Measure-Object ChargedCostUnits -Sum).Sum
    $physical = $run.Checks
    if ($run.Workload -eq 'ridge-regression') {
        $physical += $run.Fits; $fits += $run.Fits
        if ($run.Checks -ne $run.Fits -or $run.Observations.Count -ne $run.Fits -or
            !$run.Challenge.IsConfirmed -or $run.ReverseChallenge.IsConfirmed) { throw 'Ridge fresh-fit/confirmation contract failed.' }
    } elseif ($run.Workload -eq 'trusted-sorting') {
        $physical += $run.Invocations; $invocations += $run.Invocations
        if ($run.Invocations -ne 2 * $run.Checks -or $run.Observations.Count -ne $run.Checks) { throw 'Warmup accounting failed.' }
    } else { throw 'Unexpected workload.' }
    if ($charged -ne $physical -or $charged -ne $run.Ledger.Spent.cost_units) { throw 'Physical work did not reconcile with receipts.' }
    $checks += $run.Checks; $cost += $charged
}
$conservative = @($study.Runs | Where-Object { $_.Workload -eq 'ridge-regression' -and $_.ScreenThreshold -eq .5 })
$aggressive = @($study.Runs | Where-Object { $_.Workload -eq 'ridge-regression' -and $_.ScreenThreshold -eq .98 })
$timing = @($study.Runs | Where-Object Workload -eq 'trusted-sorting')
foreach ($group in @(@{Rows=$conservative}, @{Rows=$aggressive}, @{Rows=$timing})) {
    if (($group.Rows.Seed -join ',') -ne $expectedSeeds) { throw 'Dropped, duplicated or reordered fixed study roots.' }
}
if (@($conservative | Where-Object { !$_.PresetApproved -or $_.Screen.Audit.FalseRejectionRateUpper -gt .1 }).Count -gt 0) {
    throw 'The declared conservative ridge preset did not meet its validation threshold.'
}
if (@($aggressive | Where-Object { $_.PresetApproved -or $_.Screen.Audit.FalseRejectionRateLower -le .1 }).Count -gt 0 -or
    @($timing | Where-Object { $_.PresetApproved -or $_.Screen.Audit.FalseRejectionRateLower -le .1 }).Count -gt 0) {
    throw 'An intentionally harmful negative-control preset escaped the rejection audit.'
}
Write-Output "PASS: 18 fixed runs; $fits fresh fits; $invocations timing invocations; $checks correctness checks; $cost charged calls."
Write-Output 'Conservative ridge preset accepted on 6/6 roots; aggressive ridge and timing presets rejected on 12/12 roots.'
