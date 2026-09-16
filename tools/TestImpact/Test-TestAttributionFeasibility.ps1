<#
.SYNOPSIS
    Executable capability probe; does NOT enable selective execution or issue certificates.
.DESCRIPTION
    Identical aggregate production coverage can represent different test ownership.
    Isolated method runs provide a control and measure their process/collector overhead.
    Success means the probe assertions passed, NOT that per-test attribution is feasible.
#>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum AttributionCapability { AggregateOnly; MethodIsolated }

$project = Join-Path $PSScriptRoot 'fixtures/TestAttribution/Tests/AttributionTests.csproj'
$results = Join-Path ([IO.Path]::GetTempPath()) "test-attribution-$([guid]::NewGuid().ToString('N'))"
[void] (New-Item -ItemType Directory -Path $results)
$observations = [Collections.Generic.List[object]]::new()

function Assert-Probe([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}

function Invoke-ProbeRun {
    param([string] $Name, [bool] $Swap, [bool] $Coverage, [string] $Method = '')
    $destination = Join-Path $results $Name
    $env:ATTRIBUTION_SWAP = if ($Swap) { '1' } else { '0' }
    $arguments = @('test', $project, '-c', 'Release', '--no-build', '--no-restore',
        '--results-directory', $destination, '--logger', 'trx;LogFileName=results.trx',
        '--blame-hang-timeout', '30s')
    if ($Method) { $arguments += @('--filter', "FullyQualifiedName=AttributionTests.OwnershipTests.$Method") }
    if ($Coverage) {
        $arguments += @('--collect:XPlat Code Coverage', '--',
            'DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.Format=opencover',
            'DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.SingleHit=true',
            'DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.Include=[AttributionSubject]*')
    }
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & dotnet @arguments | Out-Host
    Assert-Probe ($LASTEXITCODE -eq 0) "$Name execution failed."
    $timer.Stop()
    [xml] $trx = Get-Content -LiteralPath (Join-Path $destination 'results.trx') -Raw
    $cases = @($trx.SelectNodes('//*[local-name()="UnitTestResult"]'))
    $expected = if ($Method -ceq 'First') { 2 } elseif ($Method -ceq 'Second') { 1 } else { 3 }
    Assert-Probe ($cases.Count -eq $expected) "$Name executed $($cases.Count), expected $expected cases."
    Assert-Probe (@($cases | Where-Object { $_.outcome -cne 'Passed' }).Count -eq 0) "$Name contains non-passing cases."
    $definitions = @($trx.SelectNodes('//*[local-name()="TestMethod"]'))
    $methods = @($definitions | ForEach-Object { [string] $_.name } | Sort-Object -Unique)
    $expectedMethods = if ($Method) { @($Method) } else { @('First', 'Second') }
    Assert-Probe (($methods -join ',') -ceq ($expectedMethods -join ',')) "$Name executed unexpected methods."
    $points = @()
    $visitedMethods = @()
    $references = 0
    if ($Coverage) {
        $reports = @(Get-ChildItem -LiteralPath $destination -Filter coverage.opencover.xml -Recurse -File)
        # The TRX logger copies collector attachments into its own result directory.
        # Accept byte-identical copies, but never silently pick one of conflicting reports.
        Assert-Probe ($reports.Count -gt 0) "$Name has no coverage report."
        $hashes = @($reports | ForEach-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } | Sort-Object -Unique)
        Assert-Probe ($hashes.Count -eq 1) "$Name has conflicting coverage reports."
        [xml] $report = Get-Content -LiteralPath $reports[0].FullName -Raw
        $points = @($report.SelectNodes('//SequencePoint') | Where-Object { [int] $_.vc -gt 0 } |
            ForEach-Object { "$($_.fileid):$($_.sl):$($_.sc):$($_.el):$($_.ec)" } | Sort-Object -Unique)
        Assert-Probe ($points.Count -gt 0) "$Name has no production coverage."
        $visitedMethods = @($report.SelectNodes('//Method[SequencePoints/SequencePoint[@vc!="0"]]/Name') |
            ForEach-Object { $_.InnerText } | Sort-Object)
        $references = @($report.SelectNodes('//TrackedMethodRef')).Count
        foreach ($fixtureMethod in @('Setup', 'Cleanup')) {
            Assert-Probe (@($visitedMethods | Where-Object { $_ -match "::$fixtureMethod\(" }).Count -eq 1) `
                "$Name lost shared fixture $fixtureMethod coverage."
        }
    }
    $observation = [pscustomobject]@{
        name = $Name; seconds = [math]::Round($timer.Elapsed.TotalSeconds, 3)
        caseCount = $cases.Count; testNames = @($cases | ForEach-Object { [string] $_.testName } | Sort-Object)
        visitedPoints = $points; visitedMethods = $visitedMethods; trackedReferences = $references
    }
    $observations.Add($observation)
    return $observation
}

$priorSwap = $env:ATTRIBUTION_SWAP
try {
    # Small exploratory build is the explicitly agreed feasibility gate, not a model-suite run.
    dotnet build $project -c Release --nologo -m:2 -p:UseSharedCompilation=false
    Assert-Probe ($LASTEXITCODE -eq 0) 'Attribution fixture build failed.'
    $plain = Invoke-ProbeRun -Name plain -Swap $false -Coverage $false
    $normal = Invoke-ProbeRun -Name combined -Swap $false -Coverage $true
    $swapped = Invoke-ProbeRun -Name combined-swapped -Swap $true -Coverage $true
    $first = Invoke-ProbeRun -Name first -Swap $false -Coverage $true -Method First
    $second = Invoke-ProbeRun -Name second -Swap $false -Coverage $true -Method Second
    $firstSwapped = Invoke-ProbeRun -Name first-swapped -Swap $true -Coverage $true -Method First

    Assert-Probe (($plain.testNames -join '|') -ceq ($normal.testNames -join '|')) 'Instrumentation changed executed test identities.'
    Assert-Probe (($normal.visitedPoints -join '|') -ceq ($swapped.visitedPoints -join '|')) 'Combined coverage did not remain identical.'
    Assert-Probe (($first.visitedPoints -join '|') -cne ($firstSwapped.visitedPoints -join '|')) 'Ownership swap control failed.'
    foreach ($control in @(@($first, 'Left', 'Right'), @($second, 'Right', 'Left'), @($firstSwapped, 'Right', 'Left'))) {
        Assert-Probe (@($control[0].visitedMethods | Where-Object { $_ -match "::$($control[1])\(" }).Count -eq 1) 'Expected isolated dependency absent.'
        Assert-Probe (@($control[0].visitedMethods | Where-Object { $_ -match "::$($control[2])\(" }).Count -eq 0) 'Unrelated isolated dependency included.'
    }
    Assert-Probe ($normal.trackedReferences -eq 0 -and $swapped.trackedReferences -eq 0) `
        'Collector now emits attribution references; reassess capability rather than assuming aggregate-only.'

    $evidence = [ordered]@{
        schemaVersion = 1
        capability = ([AttributionCapability]::AggregateOnly).ToString()
        isolatedControl = ([AttributionCapability]::MethodIsolated).ToString()
        productionSelectionEnabled = $false
        observations = $observations.ToArray()
        isolatedPairSeconds = [math]::Round(($first.seconds + $second.seconds), 3)
        limitations = @('Single local sample, not a repository performance estimate.',
            'Parallel attribution, worker attribution, and production selection are not proven.',
            'A capability probe is not a validation certificate.')
    }
    $evidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $results 'evidence.json') -Encoding utf8
    Write-Host "Probe assertions passed. Capability: AggregateOnly. Production selection remains disabled. Evidence: $results"
}
finally { $env:ATTRIBUTION_SWAP = $priorSwap }
