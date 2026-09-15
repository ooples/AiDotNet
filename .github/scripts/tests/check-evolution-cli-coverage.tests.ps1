$ErrorActionPreference = 'Stop'
$gate = Join-Path $PSScriptRoot '../check-evolution-cli-coverage.ps1'
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('aidotnet-coverage-gate-' + [guid]::NewGuid().ToString('N'))
$null = New-Item -ItemType Directory -Path $fixtureRoot
function Assert-Rejected([scriptblock]$action) {
    $rejected = $false
    try { & $action | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Expected coverage validation to reject this fixture.' }
}
try {
    $good = '<coverage><packages><package name="aidotnet-evolve"><classes><class filename="ProgramBenchmark.cs"><lines><line number="1" hits="1"/></lines></class></classes></package></packages></coverage>'
    $goodPath = Join-Path $fixtureRoot 'good.xml'
    $copyPath = Join-Path $fixtureRoot 'attachment.xml'
    $badPath = Join-Path $fixtureRoot 'bad.xml'
    [IO.File]::WriteAllText($goodPath, $good)
    Copy-Item -LiteralPath $goodPath -Destination $copyPath
    & $gate -CoveragePath $goodPath | Out-Null
    & $gate -CoveragePath @($goodPath, $copyPath) | Out-Null
    Assert-Rejected { & $gate -CoveragePath @() }
    Assert-Rejected { & $gate -CoveragePath (Join-Path $fixtureRoot 'missing.xml') }
    foreach ($bad in @($good.Replace('hits="1"', 'hits="0"'), $good.Replace('aidotnet-evolve', 'wrong-module'),
        $good.Replace('ProgramBenchmark.cs', 'Different.cs'))) {
        [IO.File]::WriteAllText($badPath, $bad)
        Assert-Rejected { & $gate -CoveragePath $badPath }
        Assert-Rejected { & $gate -CoveragePath @($goodPath, $badPath) }
    }
    Write-Output 'Coverage gate: 10 positive/adversarial checks passed.'
}
finally {
    $resolved = [IO.Path]::GetFullPath($fixtureRoot)
    if ([IO.Path]::GetDirectoryName($resolved).TrimEnd([IO.Path]::DirectorySeparatorChar) -ne
        [IO.Path]::GetTempPath().TrimEnd([IO.Path]::DirectorySeparatorChar) -or
        -not [IO.Path]::GetFileName($resolved).StartsWith('aidotnet-coverage-gate-')) {
        throw 'Refusing cleanup outside the owned coverage fixture directory.'
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
