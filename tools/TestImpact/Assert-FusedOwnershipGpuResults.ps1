<# Strict acceptance for the fused-persistence native-ownership GPU cases. Skipped/missing GPU cases fail the lane. #>
[CmdletBinding()]
param([Parameter(Mandatory)][string] $Trx)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$class = 'AiDotNet.Tests.UnitTests.Training.FusedPersistenceOwnershipTests'
$expected = @(
    'Probe_owns_snapshots_preserves_sources_and_releases_native_storage(gpu: False)'
    'Probe_owns_snapshots_preserves_sources_and_releases_native_storage(gpu: True)'
    'Completed_fused_steps_release_probe_storage_and_match_eager_updates(gpu: False)'
    'Completed_fused_steps_release_probe_storage_and_match_eager_updates(gpu: True)'
) | ForEach-Object { "$class.$_" }
$settings = [System.Xml.XmlReaderSettings]::new()
$settings.DtdProcessing = [System.Xml.DtdProcessing]::Prohibit
$settings.XmlResolver = $null
$reader = [System.Xml.XmlReader]::Create((Resolve-Path -LiteralPath $Trx).Path, $settings)
$doc = [System.Xml.XmlDocument]::new()
$doc.XmlResolver = $null
try { $doc.Load($reader) } finally { $reader.Dispose() }
$ns = [System.Xml.XmlNamespaceManager]::new($doc.NameTable)
$ns.AddNamespace('t','http://microsoft.com/schemas/VisualStudio/TeamTest/2010')
$counters = $doc.SelectSingleNode('/t:TestRun/t:ResultSummary/t:Counters', $ns)
if ($null -eq $counters) { throw 'Missing fused-ownership counters.' }
foreach ($name in @('total','executed','passed')) {
    if ($counters.GetAttribute($name) -cne [string]$expected.Count) { throw "Fused-ownership counter $name is $($counters.GetAttribute($name)), expected $($expected.Count)." }
}
foreach ($name in @('failed','error','timeout','aborted','inconclusive','notExecuted','notRunnable')) {
    if ($counters.GetAttribute($name) -cne '0') { throw "Fused-ownership counter $name was not zero." }
}
$results = @($doc.SelectNodes('/t:TestRun/t:Results/t:UnitTestResult', $ns))
$seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($result in $results) {
    $testName = $result.GetAttribute('testName')
    if ($result.GetAttribute('outcome') -cne 'Passed') { throw "Fused-ownership case not Passed ($($result.GetAttribute('outcome'))): $testName" }
    if ($expected -cnotcontains $testName -or -not $seen.Add($testName)) { throw "Unexpected/duplicate fused-ownership case: $testName" }
}
$missing = @($expected | Where-Object { -not $seen.Contains($_) })
if ($missing.Count -ne 0) { throw "Missing fused-ownership cases: $($missing -join '; ')" }
Write-Host "Fused-persistence ownership GPU cases PASSED: $($expected.Count) exact cases executed (GPU cases with native tracking)."
