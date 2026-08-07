# Report Failed Tests - surfaces the COMPLETE list of failed tests for a shard.
#
# Why this exists: a red shard runs to completion and reports every failure, but
# those failures are interleaved with thousands of "Passed" lines in an 80k-line
# job log. Finding "which tests actually failed, and why" meant scrolling the raw
# log or downloading the TRX artifact and parsing it by hand. This script reads
# the TRX the run already produced and writes a compact digest of ONLY the
# failures -- test name + error message -- to both the console and the job's
# GitHub Step Summary, so the full per-shard failure list is one click away on
# the run page.
#
# It intentionally does NOT change the job's pass/fail outcome: it is a reporter,
# wired as an `if: failure()` step after the test run. It exits 0 always so a
# reporting hiccup never masks or overrides the real test result.

$ErrorActionPreference = 'Stop'

function Add-Summary {
  param([string]$Line)
  Write-Host $Line
  if ($env:GITHUB_STEP_SUMMARY) {
    Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $Line
  }
}

# --blame-hang kills the whole test host when any single test exceeds the hang timeout, and
# --blame writes a Sequence_*.xml naming the test that was executing when it died. Everything
# queued behind that test never runs and never appears in the TRX, so a shard truncated this way
# reports a handful of failures and looks like it merely has a handful of failures.
#
# That is the single most misleading state this pipeline can produce: it makes a shard look nearly
# green when most of its suite never executed, and it is why fixing the visible failures kept
# revealing new ones. Detect it and say so, loudly, before anything else.
$sequenceFiles = Get-ChildItem -Path 'TestResults' -Recurse -Filter 'Sequence_*.xml' -ErrorAction SilentlyContinue
$hangVictim = $null
if ($sequenceFiles) {
  foreach ($seq in $sequenceFiles) {
    try {
      [xml]$seqXml = Get-Content $seq.FullName
      # The last <UnitTestElement> is the test that was still running when the host was killed.
      $elements = $seqXml.SelectNodes('//UnitTestElement')
      if ($elements -and $elements.Count -gt 0) {
        $last = $elements[$elements.Count - 1]
        $hangVictim = "$($last.source)::$($last.FullyQualifiedName)".TrimStart(':')
        if (-not $last.FullyQualifiedName) { $hangVictim = $last.InnerText }
      }
    } catch {
      $hangVictim = '(Sequence file present but unreadable)'
    }
  }
}

$trxFiles = Get-ChildItem -Path 'TestResults' -Recurse -Filter '*.trx' -ErrorAction SilentlyContinue

if ($hangVictim) {
  Add-Summary '## :rotating_light: THIS SHARD WAS TRUNCATED -- the failure list below is INCOMPLETE'
  Add-Summary ''
  Add-Summary "The test host was killed by ``--blame-hang`` while executing:"
  Add-Summary ''
  Add-Summary ('    ' + $hangVictim)
  Add-Summary ''
  Add-Summary 'Every test queued after it **never ran and is not reported anywhere**. Treat the'
  Add-Summary 'counts below as a FLOOR, not a total -- this shard can go green on the listed'
  Add-Summary 'failures and still be hiding an unknown number behind the hang. Fix the hang first;'
  Add-Summary 'nothing else about this shard can be trusted until it runs to completion.'
  Add-Summary ''
}

if (-not $trxFiles -or $trxFiles.Count -eq 0) {
  # No TRX means the test host died before it could flush results (OOM-kill,
  # StackOverflow, AccessViolation). That is a genuinely different failure shape
  # from ordinary assertion failures, so say so explicitly rather than printing
  # an empty list that reads as "nothing failed".
  Add-Summary '## Failed test digest'
  Add-Summary ''
  Add-Summary '_No TRX file was produced._ The test host most likely terminated'
  Add-Summary 'abnormally (out-of-memory kill, StackOverflow, or access violation)'
  Add-Summary 'before results were written, so the failing test cannot be named from'
  Add-Summary 'results alone. Check the tail of the run log and any `Sequence_*.xml`'
  Add-Summary 'blame file for the last test that started.'
  exit 0
}

$ns = @{ t = 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010' }
$failed = New-Object System.Collections.Generic.List[object]

foreach ($trx in $trxFiles) {
  [xml]$xml = Get-Content $trx.FullName
  $results = Select-Xml -Xml $xml -XPath "//t:UnitTestResult[@outcome='Failed']" -Namespace $ns
  foreach ($result in $results) {
    $node = $result.Node
    $message = ''
    $msgNode = Select-Xml -Xml $node -XPath './/t:Output/t:ErrorInfo/t:Message' -Namespace $ns
    if ($msgNode) {
      # Collapse to the first line: the exception type + message is the
      # discriminator; the full stack trace stays in the TRX artifact.
      $message = ($msgNode.Node.InnerText -split "`n")[0].Trim()
    }
    $failed.Add([PSCustomObject]@{ Name = $node.testName; Message = $message })
  }
}

Add-Summary '## Failed test digest'
Add-Summary ''

if ($failed.Count -eq 0) {
  # The step is only reached on job failure, so an empty failure set here means
  # the job failed for a NON-test reason (coverage upload, a post-step, the
  # runner). Flag that so it is not mistaken for a flake.
  Add-Summary 'The TRX recorded no failed tests, yet the job failed. The cause is'
  Add-Summary 'outside the test results themselves (a post-test step, coverage, or'
  Add-Summary 'the runner). Check the step log directly.'
  exit 0
}

Add-Summary ("**$($failed.Count) failed test(s).** Grouped by error, most-common first.")

# Executed-vs-discovered. xUnit's TRX ResultSummary carries both, and when they disagree the
# shard did not finish -- the same truncation blame-hang causes, but also what a plain host crash
# or an OOM part-way through leaves behind. Reporting the gap turns "this shard has 3 failures"
# into "this shard has 3 failures and 812 tests that never ran", which are very different facts.
try {
  [xml]$firstTrx = Get-Content $trxFiles[0].FullName
  $counters = $firstTrx.SelectSingleNode('//*[local-name()="Counters"]')
  if ($counters) {
    $total    = [int]$counters.total
    $executed = [int]$counters.executed
    if ($total -gt 0 -and $executed -lt $total) {
      Add-Summary ''
      Add-Summary (":warning: **Only $executed of $total discovered tests executed -- $($total - $executed) never ran.** " +
                   'This shard is TRUNCATED; the failure list is a floor, not a total.')
    }
  }
} catch {
  # Reporter, never a gate -- a malformed TRX must not mask the failures we did parse.
}

Add-Summary ''

# Group by error message so 25 failures that share one root cause read as one
# line item with a count, not 25 near-identical rows. This is what turns a wall
# of failures into "LLMTime: 25x reshape 64->32" at a glance.
$byMessage = $failed | Group-Object Message | Sort-Object Count -Descending
foreach ($group in $byMessage) {
  $errorText = if ($group.Name) { $group.Name } else { '(no error message captured)' }
  Add-Summary ("### {0}x  {1}" -f $group.Count, $errorText)
  Add-Summary ''
  foreach ($item in $group.Group) {
    Add-Summary ("- {0}" -f $item.Name)
  }
  Add-Summary ''
}

# Always succeed: this is a reporter, not a gate.
exit 0
