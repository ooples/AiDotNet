# Report Slow Tests - Shared script for analyzing test execution times
# Used by both net8.0 and net471 test jobs in CI/CD pipeline

Write-Host "=== COMPREHENSIVE TEST TIMING DIAGNOSTICS ===" -ForegroundColor Cyan
$trxFiles = Get-ChildItem -Path "TestResults" -Recurse -Filter "*.trx" -ErrorAction SilentlyContinue
if ($trxFiles.Count -eq 0) {
  Write-Host "No TRX files found"
  exit 0
}
$allTests = @()
$totalExecutionTime = 0
foreach ($trx in $trxFiles) {
  [xml]$xml = Get-Content $trx.FullName
  $ns = @{t = "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
  $results = Select-Xml -Xml $xml -XPath "//t:UnitTestResult" -Namespace $ns
  foreach ($result in $results) {
    $duration = $result.Node.duration
    if ($duration) {
      $ts = [TimeSpan]::Parse($duration)
      $parts = $result.Node.testName -split '\.'
      $className = if ($parts.Count -ge 2) { $parts[-2] } else { "Unknown" }
      $allTests += [PSCustomObject]@{
        Name = $result.Node.testName
        Duration = $ts.TotalSeconds
        Outcome = $result.Node.outcome
        ClassName = $className
      }
      $totalExecutionTime += $ts.TotalSeconds
    }
  }
}
Write-Host ""
Write-Host "=== TEST EXECUTION SUMMARY ===" -ForegroundColor Cyan
Write-Host ("Total tests: {0}" -f $allTests.Count)
Write-Host ("Total execution time: {0:N1} seconds ({1:N1} minutes)" -f $totalExecutionTime, ($totalExecutionTime / 60))
if ($allTests.Count -gt 0) {
  $avgTime = $totalExecutionTime / $allTests.Count
  Write-Host ("Average test time: {0:N2} seconds" -f $avgTime)
}
Write-Host ""
Write-Host "=== TOP 20 SLOWEST TESTS ===" -ForegroundColor Yellow
Write-Host "  Target: 2 seconds per test" -ForegroundColor Cyan
$allTests | Sort-Object Duration -Descending | Select-Object -First 20 | ForEach-Object {
  $color = if ($_.Duration -gt 30) { "Red" } elseif ($_.Duration -gt 10) { "Magenta" } elseif ($_.Duration -gt 2) { "Yellow" } else { "Green" }
  Write-Host ("  [{0,8:N2}s] {1} ({2})" -f $_.Duration, $_.Name, $_.Outcome) -ForegroundColor $color
}
Write-Host ""
Write-Host "=== TIME BY TEST CLASS (Top 15) ===" -ForegroundColor Cyan
$byClass = $allTests | Group-Object ClassName | ForEach-Object {
  [PSCustomObject]@{
    ClassName = $_.Name
    TestCount = $_.Count
    TotalTime = ($_.Group | Measure-Object Duration -Sum).Sum
    AvgTime = ($_.Group | Measure-Object Duration -Average).Average
    MaxTime = ($_.Group | Measure-Object Duration -Maximum).Maximum
  }
} | Sort-Object TotalTime -Descending | Select-Object -First 15
foreach ($class in $byClass) {
  $color = if ($class.TotalTime -gt 300) { "Red" } elseif ($class.TotalTime -gt 120) { "Yellow" } else { "White" }
  Write-Host ("  [{0,7:N1}s total, {1,6:N2}s avg, {2,6:N1}s max] {3} ({4} tests)" -f $class.TotalTime, $class.AvgTime, $class.MaxTime, $class.ClassName, $class.TestCount) -ForegroundColor $color
}
$criticalTests = $allTests | Where-Object { $_.Duration -gt 30 }
if ($criticalTests.Count -gt 0) {
  Write-Host ""
  Write-Host "=== CRITICAL: TESTS > 30 SECONDS (15x over 2s target - extreme optimization needed) ===" -ForegroundColor Red
  $criticalTests | Sort-Object Duration -Descending | ForEach-Object {
    Write-Host ("  [BOTTLENECK] {0,7:N1}s - {1}" -f $_.Duration, $_.Name) -ForegroundColor Red
  }
  Write-Host "These tests indicate severe performance issues in the code under test." -ForegroundColor Red
}
$warningTests = $allTests | Where-Object { $_.Duration -gt 10 -and $_.Duration -le 30 }
if ($warningTests.Count -gt 0) {
  Write-Host ""
  Write-Host "=== WARNING: TESTS 10-30 SECONDS (5-15x over 2s target - high priority optimization) ===" -ForegroundColor Yellow
  $warningTests | Sort-Object Duration -Descending | ForEach-Object {
    Write-Host ("  [WARNING] {0,7:N1}s - {1}" -f $_.Duration, $_.Name) -ForegroundColor Yellow
  }
}
$optimizeCandidates = $allTests | Where-Object { $_.Duration -gt 2 -and $_.Duration -le 10 }
if ($optimizeCandidates.Count -gt 0) {
  Write-Host ""
  Write-Host "=== OPTIMIZE: TESTS 2-10 SECONDS (over 2s target - optimization candidates) ===" -ForegroundColor Cyan
  $optimizeCandidates | Sort-Object Duration -Descending | ForEach-Object {
    Write-Host ("  [OPTIMIZE] {0,7:N1}s - {1}" -f $_.Duration, $_.Name) -ForegroundColor Cyan
  }
}
