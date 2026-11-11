<#
.SYNOPSIS
Rewrites src/ElementWiseBaseline.txt, the shrink-only inventory of the element-wise gate (ADNELEM001/ADNELEM002).

.DESCRIPTION
Builds AiDotNet for net10.0 with ElementWiseRecord=true, which leaves the baseline out so the analyzer reports every
per-element tensor indexer and INumericOperations call inside a loop, collects them from the SARIF log, and writes one
"count<TAB>rule|member" line per member, sorted. Run it after fixing occurrences: ADNELEM003 fails the build while the
file still lists more than the code contains. The gate does not allow the file to grow; review a diff that adds lines.

.PARAMETER Repository
The repository root. Defaults to the parent of tools/.
#>
param(
    [string] $Repository = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
)

$ErrorActionPreference = 'Stop'
$project = Join-Path $Repository 'src/AiDotNet.csproj'
$baselinePath = Join-Path $Repository 'src/ElementWiseBaseline.txt'
$sarif = Join-Path ([System.IO.Path]::GetTempPath()) "elementwise-$([guid]::NewGuid().ToString('N')).sarif"

# The build fails by design (every occurrence is an error in record mode); the SARIF log is what matters.
& dotnet build $project -c Release -f net10.0 -v q "-p:ElementWiseRecord=true" "-p:ErrorLog=$sarif%2Cversion=2.1" | Out-Null
if (-not (Test-Path $sarif)) { throw "The build produced no SARIF log at $sarif." }

$log = Get-Content $sarif -Raw | ConvertFrom-Json
$seen = [System.Collections.Generic.HashSet[string]]::new()
$counts = [System.Collections.Generic.Dictionary[string, int]]::new([System.StringComparer]::Ordinal)
foreach ($run in $log.runs) {
    foreach ($result in $run.results) {
        if ($result.ruleId -notin @('ADNELEM001', 'ADNELEM002')) { continue }
        $location = $result.locations[0].physicalLocation
        $signature = '{0}|{1}|{2}|{3}' -f $result.ruleId, $location.artifactLocation.uri, $location.region.startLine, $location.region.startColumn
        if (-not $seen.Add($signature)) { continue }
        if ($result.message.text -notmatch 'inside a loop in (.*?)\. Use ') { throw "Unrecognized message: $($result.message.text)" }
        $key = "$($result.ruleId)|$($Matches[1])"
        $counts[$key] = 1 + ($(if ($counts.ContainsKey($key)) { $counts[$key] } else { 0 }))
    }
}
Remove-Item $sarif

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# Element-wise gate inventory (ADNELEM001 tensor indexer, ADNELEM002 INumericOperations call, inside a loop).')
$lines.Add('# Shrink-only: a member may not exceed its count. Regenerate with tools/ElementWise/Update-ElementWiseBaseline.ps1.')
foreach ($key in ($counts.Keys | Sort-Object -CaseSensitive)) { $lines.Add("$($counts[$key])`t$key") }
[System.IO.File]::WriteAllLines($baselinePath, $lines, [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote $($counts.Count) members ($($seen.Count) occurrences) to $baselinePath"
