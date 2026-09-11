param([Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string[]]$CoveragePath)
$ErrorActionPreference = 'Stop'
# VSTest may copy the same collector attachment beneath its TRX In/ directory. Accept byte-identical
# copies only; distinct reports still require explicit aggregation rather than arbitrary first-file selection.
$reportGroups = @($CoveragePath | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 } | Group-Object Hash)
if ($reportGroups.Count -ne 1) { throw 'Expected one distinct CLI coverage report.' }
[xml]$coverageDocument = Get-Content -Raw -LiteralPath $reportGroups[0].Group[0].Path
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
