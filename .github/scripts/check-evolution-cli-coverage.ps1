param([Parameter(Mandatory = $true)][string]$CoveragePath)
$ErrorActionPreference = 'Stop'
[xml]$coverageDocument = Get-Content -Raw -LiteralPath $CoveragePath
$packages = @($coverageDocument.coverage.packages.package)
if ($packages.Count -ne 1 -or $packages[0].name -cne 'aidotnet-evolve') {
    throw 'Expected coverage for aidotnet-evolve alone; verify the Coverlet include filter.'
}
$lines = @($packages[0].classes.class |
    Where-Object { $_.filename -match '(^|[/\\])ProgramBenchmark\.cs$' } |
    ForEach-Object { $_.lines.line } | Group-Object number)
if ($lines.Count -eq 0) { throw 'No executable ProgramBenchmark.cs lines were measured.' }
$covered = @($lines | Where-Object { @($_.Group | Where-Object { [long]$_.hits -gt 0 }).Count -gt 0 }).Count
if ($covered * 10 -lt $lines.Count * 9) {
    throw "ProgramBenchmark.cs coverage below 90%: $covered/$($lines.Count)."
}
Write-Output "ProgramBenchmark.cs: $covered/$($lines.Count) executable lines covered; module filter verified."
