param(
    [string]$SnapshotPath,
    [string]$OutDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-LatestSnapshotPath {
    $snapshotsRoot = Join-Path $PSScriptRoot "..\\BenchmarkDotNet.Artifacts\\results\\snapshots"
    if (-not (Test-Path $snapshotsRoot)) {
        return $null
    }

    $latest = Get-ChildItem $snapshotsRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        return $null
    }

    return $latest.FullName
}

function Convert-ToNanoseconds {
    param([string]$value)

    if ([string]::IsNullOrWhiteSpace($value)) {
        return $null
    }

    $microSign = [char]0x00B5
    $muSign = [char]0x03BC
    $clean = $value.Trim().Trim('"') -replace '\\s+', ' '

    if ($clean -notmatch '^([0-9.,]+)\\s*([\\p{L}]+)$') {
        throw "Unsupported time format: $value"
    }

    $numberText = $matches[1] -replace ',', ''
    $unit = $matches[2].Replace($microSign, 'u').Replace($muSign, 'u')

    $number = [double]::Parse($numberText, [Globalization.CultureInfo]::InvariantCulture)
    $multiplier = switch ($unit) {
        "ns" { 1.0 }
        "us" { 1000.0 }
        "ms" { 1000000.0 }
        "s" { 1000000000.0 }
        default { throw "Unsupported unit '$unit' in value '$value'" }
    }

    return $number * $multiplier
}

function Normalize-Size {
    param([string]$size)

    if ($null -eq $size) {
        return ""
    }

    return $size.Trim()
}

function Compare-Benchmarks {
    param(
        [string]$CsvPath,
        [string]$CompetitorName,
        [string]$CompetitorPrefix,
        [hashtable]$BaseMap
    )

    if (-not (Test-Path $CsvPath)) {
        throw "Missing CSV: $CsvPath"
    }

    $rows = Import-Csv $CsvPath
    $competitorRows = @{}

    foreach ($row in $rows) {
        $key = "{0}|{1}" -f $row.Method, (Normalize-Size $row.size)
        $competitorRows[$key] = $row
    }

    $results = @()
    foreach ($row in $rows | Where-Object { $_.Method -like "AiDotNet_*" }) {
        $base = $row.Method.Substring("AiDotNet_".Length)
        if (-not $BaseMap.ContainsKey($base)) {
            Write-Warning "No competitor map for $($row.Method) in $CompetitorName"
            continue
        }

        $compBase = $BaseMap[$base]
        $compMethod = "{0}_{1}" -f $CompetitorPrefix, $compBase
        $sizeKey = Normalize-Size $row.size
        $compKey = "{0}|{1}" -f $compMethod, $sizeKey
        $compRow = $competitorRows[$compKey]

        if (-not $compRow) {
            Write-Warning "Missing $CompetitorName row for $compMethod (size '$sizeKey')"
            continue
        }

        $aiNs = Convert-ToNanoseconds $row.Mean
        $compNs = Convert-ToNanoseconds $compRow.Mean

        if ($null -eq $aiNs -or $null -eq $compNs) {
            Write-Warning "Missing mean for $($row.Method) vs $compMethod"
            continue
        }

        $ratio = $aiNs / $compNs
        $deltaPercent = (($aiNs - $compNs) / $compNs) * 100.0
        $winner = if ($aiNs -le $compNs) { "AiDotNet" } else { $CompetitorName }

        $results += [pscustomobject]@{
            Competitor = $CompetitorName
            Benchmark = $base
            Size = $sizeKey
            AiDotNetUs = [math]::Round($aiNs / 1000.0, 3)
            CompetitorUs = [math]::Round($compNs / 1000.0, 3)
            AiOverComp = [math]::Round($ratio, 3)
            DeltaPercent = [math]::Round($deltaPercent, 1)
            Winner = $winner
        }
    }

    return $results
}

if (-not $SnapshotPath) {
    $SnapshotPath = Get-LatestSnapshotPath
}

if (-not $SnapshotPath -or -not (Test-Path $SnapshotPath)) {
    throw "Snapshot path not found. Pass -SnapshotPath or run scripts/snapshot-cpu-benchmarks.ps1."
}

if (-not $OutDir) {
    $OutDir = Join-Path $SnapshotPath "comparisons"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$configs = @(
    @{
        Name = "TorchSharp"
        Csv = "AiDotNetBenchmarkTests.TorchSharpCpuComparisonBenchmarks-report.csv"
        Prefix = "TorchSharp"
        BaseMap = @{
            TensorAdd = "Add"
            TensorMultiply = "Multiply"
            TensorSum = "Sum"
            TensorMean = "Mean"
            TensorMatMul = "MatMul"
            ReLU = "ReLU"
            Sigmoid = "Sigmoid"
            Conv2D = "Conv2D"
        }
    },
    @{
        Name = "TensorFlow"
        Csv = "AiDotNetBenchmarkTests.TensorFlowCpuComparisonBenchmarks-report.csv"
        Prefix = "TensorFlow"
        BaseMap = @{
            TensorAdd = "Add"
            TensorMultiply = "Multiply"
            TensorSum = "ReduceSum"
            TensorMean = "ReduceMean"
            TensorMatMul = "MatMul"
            ReLU = "ReLU"
            Sigmoid = "Sigmoid"
            Conv2D = "Conv2D"
        }
    },
    @{
        Name = "ML.NET"
        Csv = "AiDotNetBenchmarkTests.MlNetCpuComparisonBenchmarks-report.csv"
        Prefix = "MlNet"
        BaseMap = @{
            TensorAdd = "Add"
            TensorMultiply = "Multiply"
            TensorSum = "Sum"
            TensorMean = "Mean"
        }
    }
)

$allResults = @()
foreach ($config in $configs) {
    $csvPath = Join-Path $SnapshotPath $config.Csv
    $allResults += Compare-Benchmarks -CsvPath $csvPath -CompetitorName $config.Name -CompetitorPrefix $config.Prefix -BaseMap $config.BaseMap
}

$csvOut = Join-Path $OutDir "comparison-summary.csv"
$allResults | Sort-Object Competitor, AiOverComp -Descending | Export-Csv -NoTypeInformation -Path $csvOut

$mdOut = Join-Path $OutDir "comparison-summary.md"
$lines = @()
$lines += "# CPU Comparison Summary"
$lines += ""
$lines += "Snapshot: $SnapshotPath"
$lines += ""
foreach ($group in $allResults | Group-Object Competitor) {
    $lines += "## $($group.Name)"
    $lines += ""
    $lines += "| Benchmark | Size | AiDotNet us | $($group.Name) us | Ai/Comp | Delta % | Winner |"
    $lines += "| --- | --- | --- | --- | --- | --- | --- |"
    foreach ($row in ($group.Group | Sort-Object AiOverComp -Descending)) {
        $lines += "| $($row.Benchmark) | $($row.Size) | $($row.AiDotNetUs) | $($row.CompetitorUs) | $($row.AiOverComp) | $($row.DeltaPercent) | $($row.Winner) |"
    }
    $lines += ""
}

$lines | Out-File -FilePath $mdOut -Encoding ASCII

Write-Host "Wrote comparison summary:" $csvOut
Write-Host "Wrote comparison summary:" $mdOut
