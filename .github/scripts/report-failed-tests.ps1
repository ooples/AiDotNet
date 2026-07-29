# Report Failed Tests - surfaces the COMPLETE list of failed tests for a shard.
#
# Why this exists: a red shard runs to completion and reports every failure, but
# those failures are interleaved with thousands of "Passed" lines in an 80k-line
# job log. Finding "which tests actually failed, and why" meant scrolling the raw
# log or downloading the TRX artifact and parsing it by hand. This script reads
# the TRX the run already produced and writes a compact digest of ONLY the
# failures — test name + error message — to both the console and the job's
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

$trxFiles = Get-ChildItem -Path 'TestResults' -Recurse -Filter '*.trx' -ErrorAction SilentlyContinue
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
