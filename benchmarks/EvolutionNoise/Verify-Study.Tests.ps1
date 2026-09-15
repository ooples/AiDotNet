param([Parameter(Mandatory = $true)][string] $Report)
$ErrorActionPreference = 'Stop'
$json = Get-Content -LiteralPath $Report -Raw
$mutations = @(
    { param($s) $s.Runs = @($s.Runs | Select-Object -Skip 1) },
    { param($s) $s.Runs[0].Ledger.Spent.cost_units++ },
    { param($s) $s.Runs[0].PresetApproved = $false },
    { param($s) $s.Runs[6].PresetApproved = $true },
    { param($s) $s.Seeds[0] = 99 },
    { param($s) $s.Runs[0].Challenge.CandidateSearch.Samples[1].Context.SampleIdentity = $s.Runs[0].Challenge.CandidateSearch.Samples[0].Context.SampleIdentity }
)
foreach ($mutation in $mutations) {
    $changed = $json | ConvertFrom-Json
    & $mutation $changed
    $temporaryReport = [IO.Path]::GetTempFileName()
    try {
        [IO.File]::WriteAllText($temporaryReport, ($changed | ConvertTo-Json -Depth 100))
        $refused = $false
        try { & "$PSScriptRoot/Verify-Study.ps1" -Report $temporaryReport | Out-Null }
        catch { $refused = $true }
        if (!$refused) { throw 'A corrupted study was incorrectly accepted.' }
    } finally {
        Remove-Item -LiteralPath $temporaryReport
    }
}
Write-Output 'PASS: all 6 corrupt study variants were rejected.'
