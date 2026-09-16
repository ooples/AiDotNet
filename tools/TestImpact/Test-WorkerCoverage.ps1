<# Runs a tiny executable before/after proof, not the AiDotNet model suite. #>
[CmdletBinding()]
param([switch] $NoBuild)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$fixture = Join-Path $PSScriptRoot 'fixtures/WorkerCoverage'
$project = Join-Path $fixture 'Parent/Parent.csproj'
$parent = Join-Path $fixture 'Parent/bin/Release/net10.0'
$worker = Join-Path $fixture 'Worker/bin/Release/net10.0'
if (-not $NoBuild) {
    dotnet build $project -c Release -m:2 -p:UseSharedCompilation=false --nologo
    if ($LASTEXITCODE -ne 0) { throw 'Worker coverage fixture build failed.' }
}
$results = Join-Path ([IO.Path]::GetTempPath()) "worker-coverage-$([Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $results | Out-Null
$originalWorker = Join-Path $results 'Shared.original.dll'
$workerAssembly = Join-Path $worker 'Shared.dll'
Copy-Item -LiteralPath $workerAssembly -Destination $originalWorker
$previousWorker = $env:WORKER_COVERAGE_FIXTURE_DLL
$parentHash = (Get-FileHash (Join-Path $parent 'Shared.dll')).Hash
try {
    # Reset any link left by a previous run without writing through that link.
    Remove-Item -LiteralPath $workerAssembly
    Copy-Item -LiteralPath $originalWorker -Destination $workerAssembly
    $env:WORKER_COVERAGE_FIXTURE_DLL = Join-Path $worker 'Worker.dll'
    foreach ($connected in @($false, $true)) {
        if ($connected) {
            & "$PSScriptRoot/Connect-WorkerCoverage.ps1" -ParentDirectory $parent `
                -WorkerDirectory $worker -AssemblyName Shared.dll
            if ($LASTEXITCODE -ne 0) { throw 'Worker coverage connection failed.' }
        }
        $phase = if ($connected) { 'after' } else { 'before' }
        $phaseResults = Join-Path $results $phase
        dotnet test $project -c Release --no-build --no-restore `
            --collect:'XPlat Code Coverage' --results-directory $phaseResults -- `
            DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.Format=opencover `
            DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.SingleHit=true `
            'DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.Include=[Shared]*'
        if ($LASTEXITCODE -ne 0) { throw "$phase worker fixture failed." }
        $reports = @(Get-ChildItem -LiteralPath $phaseResults -Filter coverage.opencover.xml -Recurse -File)
        if ($reports.Count -ne 1) { throw "$phase must produce exactly one coverage report." }
        [xml] $report = Get-Content -LiteralPath $reports[0].FullName -Raw
        $points = @($report.SelectNodes('//Method[contains(Name,"WorkerOnly::Calculate")]/SequencePoints/SequencePoint'))
        $visited = @($points | Where-Object { [int] $_.vc -gt 0 }).Count
        $expectedVisited = if ($connected) { $points.Count } else { 0 }
        if ($points.Count -lt 2 -or $visited -ne $expectedVisited) {
            throw "${phase}: expected worker-only coverage=$expectedVisited, observed $visited/$($points.Count)."
        }
        Write-Host "$phase worker-only coverage: $visited/$($points.Count)"
    }
    # Path confusion, a different build, and non-DLL input must never substitute bytes.
    foreach ($name in @('../Shared.dll', 'Shared.pdb')) {
        $rejected = $false
        try { & "$PSScriptRoot/Connect-WorkerCoverage.ps1" -ParentDirectory $parent -WorkerDirectory $worker -AssemblyName $name }
        catch { $rejected = $true }
        if (-not $rejected) { throw "Unsafe assembly name accepted: $name" }
    }
    $rejected = $false
    try { & "$PSScriptRoot/Connect-WorkerCoverage.ps1" -ParentDirectory $parent -WorkerDirectory $parent -AssemblyName Shared.dll }
    catch { $rejected = $true }
    if (-not $rejected) { throw 'Same parent/worker path accepted.' }
    Remove-Item -LiteralPath $workerAssembly
    Copy-Item -LiteralPath (Join-Path $parent 'Parent.dll') -Destination $workerAssembly
    $mismatchedHash = (Get-FileHash -LiteralPath $workerAssembly).Hash
    $rejected = $false
    try { & "$PSScriptRoot/Connect-WorkerCoverage.ps1" -ParentDirectory $parent -WorkerDirectory $worker -AssemblyName Shared.dll }
    catch { $rejected = $true }
    if (-not $rejected -or (Get-FileHash -LiteralPath $workerAssembly).Hash -cne $mismatchedHash) {
        throw 'A mismatched worker binary was accepted or overwritten.'
    }
    if ((Get-FileHash (Join-Path $parent 'Shared.dll')).Hash -cne $parentHash) {
        throw 'Coverage did not restore the original parent binary.'
    }
    Write-Host "Worker coverage proof passed; reports retained in $results"
}
finally {
    $env:WORKER_COVERAGE_FIXTURE_DLL = $previousWorker
    # These are individual fixture output files, never recursive workspace cleanup.
    if (Test-Path -LiteralPath $workerAssembly -PathType Leaf) { Remove-Item -LiteralPath $workerAssembly }
    Copy-Item -LiteralPath $originalWorker -Destination $workerAssembly
}
exit 0
