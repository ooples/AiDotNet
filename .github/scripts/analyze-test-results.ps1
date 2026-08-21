[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$ResultsPath,

  [Parameter(Mandatory = $true)]
  [string]$OutputDirectory,

  [string]$JobsPath,
  [string]$JobsJson,
  [string]$BaselinePath,
  [string]$Repository = $env:GITHUB_REPOSITORY,
  [string]$CommitSha = $env:GITHUB_SHA,
  [string]$RunUrl,
  [string[]]$MissingFailedShards = @(),
  [switch]$FailOnRegression
)

$ErrorActionPreference = 'Stop'

function Get-IntegerAttribute {
  param($Node, [string]$Name)
  if ($null -eq $Node -or $null -eq $Node.Attributes[$Name]) { return 0 }
  $value = 0
  if ([int]::TryParse($Node.Attributes[$Name].Value, [ref]$value)) { return $value }
  return 0
}

function ConvertTo-ShardKey {
  param([string]$Name)
  if (-not $Name) { return '' }
  return (($Name -replace '[\\/:*?"<>|\s-]+', '_').Trim('_')).ToLowerInvariant()
}

function Get-ShardCategory {
  param([string]$Name)
  switch -Regex ($Name) {
    '^Unit\b|^Unit_' { return 'Unit' }
    '^Integration\b|^Integration_' { return 'Integration' }
    '^ModelFamily\b|^ModelFamily_' { return 'Model family' }
    '^Generated\b|^Generated_' { return 'Generated' }
    '^Component\b|^Component_' { return 'Component' }
    default { return 'Other' }
  }
}

function Get-FailureCategory {
  param([string]$Message, [string]$StackTrace)
  $text = "$Message`n$StackTrace"
  switch -Regex ($text) {
    '(?i)timed?\s*out|timeout|blame-hang|TaskCanceledException' { return 'Timeout or cancellation' }
    '(?i)OutOfMemory|out of memory|cannot allocate|allocation failed|OOM' { return 'Memory or resource exhaustion' }
    '(?i)\bNaN\b|non[- ]finite|\bInfinity\b|overflow|underflow' { return 'Numerical stability' }
    '(?i)shape|dimension|reshape|broadcast|length mismatch|rank mismatch' { return 'Shape or tensor contract' }
    '(?i)gradient|parameter.*(?:change|update)|training|loss (?:grew|increase|reduce)' { return 'Training or gradient behavior' }
    '(?i)serializ|deserializ|round.?trip|clone' { return 'State, clone, or serialization' }
    '(?i)Assert\.|Xunit\.|Expected:|Actual:' { return 'Assertion mismatch' }
    default { return 'Unhandled exception or other' }
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][string]$Value, [int]$MaximumLength = 200)
  if ($null -eq $Value) { return '' }
  $flat = ($Value -replace '\r?\n', ' ' -replace '\s+', ' ').Trim()
  if ($flat.Length -gt $MaximumLength) { $flat = $flat.Substring(0, $MaximumLength) + '...' }
  return ($flat -replace '\|', '\|')
}

function Add-Count {
  param([hashtable]$Table, [string]$Key, [int]$Amount = 1)
  if (-not $Table.ContainsKey($Key)) { $Table[$Key] = 0 }
  $Table[$Key] += $Amount
}

function Convert-CountsToObject {
  param([hashtable]$Counts)
  $ordered = [ordered]@{}
  foreach ($key in @($Counts.Keys | Sort-Object)) { $ordered[$key] = $Counts[$key] }
  return [PSCustomObject]$ordered
}

if (-not (Test-Path -LiteralPath $ResultsPath -PathType Container)) {
  throw "ResultsPath does not exist or is not a directory: $ResultsPath"
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$resolvedResults = (Resolve-Path -LiteralPath $ResultsPath).Path

$testJobs = @()
if ($JobsJson -or ($JobsPath -and (Test-Path -LiteralPath $JobsPath -PathType Leaf))) {
  $jobDocument = if ($JobsJson) {
    $JobsJson | ConvertFrom-Json
  } else {
    Get-Content -LiteralPath $JobsPath -Raw | ConvertFrom-Json
  }
  $allJobs = if ($null -ne $jobDocument.jobs) { @($jobDocument.jobs) } else { @($jobDocument) }
  foreach ($job in $allJobs) {
    if ([string]$job.name -notmatch '^Tests \(net10\.0\) - (.+)$') { continue }
    $testJobs += [PSCustomObject]@{
      name = $Matches[1]
      key = ConvertTo-ShardKey $Matches[1]
      status = [string]$job.status
      conclusion = [string]$job.conclusion
      url = [string]$job.html_url
    }
  }
}

$jobByKey = @{}
foreach ($job in $testJobs) { $jobByKey[$job.key] = $job }

$counterTotals = [ordered]@{
  total = 0
  executed = 0
  passed = 0
  failed = 0
  skipped = 0
  notExecuted = 0
}
$parseErrors = New-Object System.Collections.Generic.List[object]
$failureRecords = New-Object System.Collections.Generic.List[object]
$resultShardKeys = New-Object 'System.Collections.Generic.HashSet[string]'
$trxFiles = @(Get-ChildItem -LiteralPath $resolvedResults -Recurse -Filter '*.trx' -File -ErrorAction SilentlyContinue)

foreach ($trx in $trxFiles) {
  $relative = $trx.FullName.Substring($resolvedResults.Length).TrimStart(
    [IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
  $firstSegment = ($relative -split '[\\/]')[0]
  $artifactKey = $firstSegment -replace '^(?:test-results|coverage)-[0-9a-f]+-', ''
  $shardKey = ConvertTo-ShardKey $artifactKey
  if (-not $shardKey) { $shardKey = ConvertTo-ShardKey $trx.Directory.Name }
  [void]$resultShardKeys.Add($shardKey)
  $shardName = if ($jobByKey.ContainsKey($shardKey)) { $jobByKey[$shardKey].name } else { $artifactKey }

  try {
    [xml]$document = Get-Content -LiteralPath $trx.FullName -Raw
  }
  catch {
    $parseErrors.Add([PSCustomObject]@{ file = $relative; error = $_.Exception.Message })
    continue
  }

  $counters = $document.SelectSingleNode('//*[local-name()="Counters"]')
  $counterTotals.total += Get-IntegerAttribute $counters 'total'
  $counterTotals.executed += Get-IntegerAttribute $counters 'executed'
  $counterTotals.passed += Get-IntegerAttribute $counters 'passed'
  $counterTotals.failed += Get-IntegerAttribute $counters 'failed'
  $counterTotals.skipped += (Get-IntegerAttribute $counters 'notExecuted') +
    (Get-IntegerAttribute $counters 'inconclusive')

  $definitions = @{}
  foreach ($unitTest in @($document.SelectNodes('//*[local-name()="TestDefinitions"]/*[local-name()="UnitTest"]'))) {
    $method = $unitTest.SelectSingleNode('./*[local-name()="TestMethod"]')
    $id = [string]$unitTest.id
    if ($id -and $null -ne $method) {
      $definitions[$id] = [PSCustomObject]@{
        className = [string]$method.className
        methodName = [string]$method.name
      }
    }
  }

  foreach ($result in @($document.SelectNodes('//*[local-name()="UnitTestResult" and @outcome="Failed"]'))) {
    $definition = $definitions[[string]$result.testId]
    $className = if ($null -ne $definition) { $definition.className } else { '' }
    $methodName = if ($null -ne $definition) { $definition.methodName } else { [string]$result.testName }
    $displayName = [string]$result.testName
    $identity = if ($className) { "$className::$displayName" } else { $displayName }
    $methodIdentity = if ($className) { "$className::$methodName" } else { $methodName }
    $messageNode = $result.SelectSingleNode('.//*[local-name()="ErrorInfo"]/*[local-name()="Message"]')
    $stackNode = $result.SelectSingleNode('.//*[local-name()="ErrorInfo"]/*[local-name()="StackTrace"]')
    $message = if ($null -ne $messageNode) { [string]$messageNode.InnerText } else { '' }
    $stack = if ($null -ne $stackNode) { [string]$stackNode.InnerText } else { '' }

    $failureRecords.Add([PSCustomObject]@{
      identity = $identity
      methodIdentity = $methodIdentity
      className = $className
      methodName = $methodName
      displayName = $displayName
      shard = $shardName
      shardCategory = Get-ShardCategory $shardName
      failureCategory = Get-FailureCategory $message $stack
      message = (($message -split '\r?\n')[0]).Trim()
    })
  }
}

$counterTotals.notExecuted = [Math]::Max(0, $counterTotals.total - $counterTotals.executed)

$uniqueFailures = @()
foreach ($group in @($failureRecords | Group-Object identity | Sort-Object Name)) {
  $first = $group.Group[0]
  $uniqueFailures += [PSCustomObject]@{
    identity = $first.identity
    methodIdentity = $first.methodIdentity
    displayName = $first.displayName
    className = $first.className
    methodName = $first.methodName
    shards = @($group.Group.shard | Sort-Object -Unique)
    shardCategory = $first.shardCategory
    failureCategory = $first.failureCategory
    message = $first.message
    occurrences = $group.Count
  }
}

$failedJobs = @($testJobs | Where-Object { $_.conclusion -and $_.conclusion -ne 'success' -and $_.conclusion -ne 'skipped' })
if ($testJobs.Count -eq 0) {
  $failedJobs = @($uniqueFailures | ForEach-Object { $_.shards } | Sort-Object -Unique | ForEach-Object {
    [PSCustomObject]@{ name = $_; key = ConvertTo-ShardKey $_; status = 'completed'; conclusion = 'failure'; url = '' }
  })
}
foreach ($missingShard in $MissingFailedShards) {
  if ($failedJobs.name -contains $missingShard) { continue }
  $failedJobs += [PSCustomObject]@{
    name = $missingShard
    key = ConvertTo-ShardKey $missingShard
    status = 'completed'
    conclusion = 'cancelled'
    url = ''
  }
}

$missingResultShards = @($testJobs | Where-Object { -not $resultShardKeys.Contains($_.key) })
$failedMethodCount = @($uniqueFailures.methodIdentity | Sort-Object -Unique).Count
$shardCategoryCounts = @{}
foreach ($job in $failedJobs) { Add-Count $shardCategoryCounts (Get-ShardCategory $job.name) }
$failureCategoryCounts = @{}
foreach ($failure in $uniqueFailures) { Add-Count $failureCategoryCounts $failure.failureCategory }
$shardReports = New-Object System.Collections.Generic.List[object]
if ($testJobs.Count -gt 0) { $shardsForReport = $testJobs } else { $shardsForReport = $failedJobs }
foreach ($job in @($shardsForReport | Sort-Object name)) {
  $shardReports.Add([PSCustomObject][ordered]@{
    name = $job.name
    status = $job.status
    conclusion = $job.conclusion
    url = $job.url
    hasResults = $resultShardKeys.Contains([string]$job.key)
  })
}

$baseline = $null
if ($BaselinePath -and (Test-Path -LiteralPath $BaselinePath -PathType Leaf)) {
  $baseline = Get-Content -LiteralPath $BaselinePath -Raw | ConvertFrom-Json
}

$currentFailureIds = @($uniqueFailures.identity)
$baselineFailureIds = if ($null -ne $baseline) { @($baseline.failures | ForEach-Object { [string]$_.identity }) } else { @() }
$newFailureIds = @($currentFailureIds | Where-Object { $_ -notin $baselineFailureIds } | Sort-Object -Unique)
$resolvedFailureIds = @($baselineFailureIds | Where-Object { $_ -notin $currentFailureIds } | Sort-Object -Unique)
$persistentFailureIds = @($currentFailureIds | Where-Object { $_ -in $baselineFailureIds } | Sort-Object -Unique)
$currentFailedShardNames = @($failedJobs.name | Sort-Object -Unique)
$baselineFailedShardNames = if ($null -ne $baseline) { @($baseline.shards | Where-Object { $_.conclusion -and $_.conclusion -ne 'success' -and $_.conclusion -ne 'skipped' } | ForEach-Object { [string]$_.name } | Sort-Object -Unique) } else { @() }
$newFailedShards = @($currentFailedShardNames | Where-Object { $_ -notin $baselineFailedShardNames })
$resolvedFailedShards = @($baselineFailedShardNames | Where-Object { $_ -notin $currentFailedShardNames })

$baselineAvailable = $null -ne $baseline
$hasRegression = $baselineAvailable -and ($newFailureIds.Count -gt 0 -or $newFailedShards.Count -gt 0)
$regressionStatus = if (-not $baselineAvailable) { 'baseline-unavailable' } elseif ($hasRegression) { 'regressed' } elseif ($resolvedFailureIds.Count -gt 0 -or $resolvedFailedShards.Count -gt 0) { 'improved' } else { 'stable' }

$report = [ordered]@{
  schemaVersion = 1
  generatedUtc = [DateTime]::UtcNow.ToString('o')
  source = [ordered]@{ repository = $Repository; commitSha = $CommitSha; runUrl = $RunUrl }
  summary = [ordered]@{
    trxFiles = $trxFiles.Count
    totalTests = $counterTotals.total
    executedTests = $counterTotals.executed
    passedTests = $counterTotals.passed
    failedResultRecords = $failureRecords.Count
    uniqueFailedTests = $uniqueFailures.Count
    uniqueFailedMethods = $failedMethodCount
    skippedTests = $counterTotals.skipped
    notExecutedTests = $counterTotals.notExecuted
    failedShards = $currentFailedShardNames.Count
    missingResultShards = $missingResultShards.Count
    unparseableTrxFiles = $parseErrors.Count
  }
  categories = [ordered]@{
    failedShards = Convert-CountsToObject $shardCategoryCounts
    failedTests = Convert-CountsToObject $failureCategoryCounts
  }
  shards = $shardReports.ToArray()
  failures = $uniqueFailures
  diagnostics = [ordered]@{
    missingResultShards = @($missingResultShards.name)
    parseErrors = $parseErrors.ToArray()
  }
  regression = [ordered]@{
    baselineAvailable = $baselineAvailable
    status = $regressionStatus
    baselineCommitSha = if ($baselineAvailable) { [string]$baseline.source.commitSha } else { '' }
    newFailedTests = $newFailureIds
    resolvedFailedTests = $resolvedFailureIds
    persistentFailedTests = $persistentFailureIds
    newFailedShards = $newFailedShards
    resolvedFailedShards = $resolvedFailedShards
  }
}

$jsonPath = Join-Path $OutputDirectory 'ci-test-analysis.json'
$markdownPath = Join-Path $OutputDirectory 'ci-test-analysis.md'
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('## Aggregate test analysis')
$lines.Add('')
$lines.Add("Regression status: **$regressionStatus**")
$lines.Add('')
$lines.Add('| Metric | Current | Baseline | Delta |')
$lines.Add('|---|---:|---:|---:|')
$baselineFailedTests = if ($baselineAvailable) { [int]$baseline.summary.uniqueFailedTests } else { 0 }
$baselineFailedMethods = if ($baselineAvailable) { [int]$baseline.summary.uniqueFailedMethods } else { 0 }
$baselineFailedShards = if ($baselineAvailable) { [int]$baseline.summary.failedShards } else { 0 }
$baselineLabel = if ($baselineAvailable) { $baselineFailedTests } else { 'n/a' }
$deltaLabel = if ($baselineAvailable) { $uniqueFailures.Count - $baselineFailedTests } else { 'n/a' }
$lines.Add("| Unique failing test cases | $($uniqueFailures.Count) | $baselineLabel | $deltaLabel |")
$baselineLabel = if ($baselineAvailable) { $baselineFailedMethods } else { 'n/a' }
$deltaLabel = if ($baselineAvailable) { $failedMethodCount - $baselineFailedMethods } else { 'n/a' }
$lines.Add("| Unique failing test methods | $failedMethodCount | $baselineLabel | $deltaLabel |")
$baselineLabel = if ($baselineAvailable) { $baselineFailedShards } else { 'n/a' }
$deltaLabel = if ($baselineAvailable) { $currentFailedShardNames.Count - $baselineFailedShards } else { 'n/a' }
$lines.Add("| Failing shards | $($currentFailedShardNames.Count) | $baselineLabel | $deltaLabel |")
$lines.Add("| Failed result records | $($failureRecords.Count) | n/a | n/a |")
$lines.Add("| Executed / discovered | $($counterTotals.executed) / $($counterTotals.total) | n/a | n/a |")
$lines.Add("| Shards without a result artifact | $($missingResultShards.Count) | n/a | n/a |")
$lines.Add('')

if ($missingResultShards.Count -gt 0) {
  $lines.Add(':warning: The following shards produced no downloadable TRX, so all test counts are floors:')
  $lines.Add('')
  foreach ($shard in $missingResultShards) { $lines.Add("- $($shard.name) ($($shard.conclusion))") }
  $lines.Add('')
}

$lines.Add('### Failure categories')
$lines.Add('')
$lines.Add('| Category | Unique failing tests |')
$lines.Add('|---|---:|')
foreach ($entry in @($failureCategoryCounts.GetEnumerator() |
    Sort-Object -Property @{ Expression = 'Value'; Descending = $true }, Name)) {
  $lines.Add("| $($entry.Name) | $($entry.Value) |")
}
$lines.Add('')
$lines.Add('### Failing shard categories')
$lines.Add('')
$lines.Add('| Category | Failing shards |')
$lines.Add('|---|---:|')
foreach ($entry in @($shardCategoryCounts.GetEnumerator() |
    Sort-Object -Property @{ Expression = 'Value'; Descending = $true }, Name)) {
  $lines.Add("| $($entry.Name) | $($entry.Value) |")
}
$lines.Add('')

if ($baselineAvailable) {
  $lines.Add("### New failures ($($newFailureIds.Count))")
  $lines.Add('')
  if ($newFailureIds.Count -eq 0) { $lines.Add('_None._') }
  foreach ($id in $newFailureIds) {
    $failure = $uniqueFailures | Where-Object identity -eq $id | Select-Object -First 1
    $lines.Add(('- `{0}` - {1}' -f (ConvertTo-MarkdownCell $id 240),
        (ConvertTo-MarkdownCell $failure.message 240)))
  }
  $lines.Add('')
  $lines.Add("### Resolved baseline failures ($($resolvedFailureIds.Count))")
  $lines.Add('')
  if ($resolvedFailureIds.Count -eq 0) { $lines.Add('_None._') }
  foreach ($id in $resolvedFailureIds) {
    $lines.Add(('- `{0}`' -f (ConvertTo-MarkdownCell $id 240)))
  }
  $lines.Add('')
}

$lines.Add("<details><summary>All current failing tests ($($uniqueFailures.Count))</summary>")
$lines.Add('')
foreach ($failure in $uniqueFailures) {
  $lines.Add(('- **{0}** - `{1}` - {2}' -f $failure.failureCategory,
      (ConvertTo-MarkdownCell $failure.identity 240),
      (ConvertTo-MarkdownCell $failure.message 240)))
}
$lines.Add('')
$lines.Add('</details>')

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8
if ($env:GITHUB_STEP_SUMMARY) { $lines | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY -Encoding utf8 }

Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
Write-Host "Unique failed tests: $($uniqueFailures.Count); failed shards: $($currentFailedShardNames.Count); status: $regressionStatus"

if ($FailOnRegression -and $hasRegression) { exit 2 }
