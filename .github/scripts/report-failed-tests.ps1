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
    # THE CONSOLE WRITE HAPPENS FIRST AND THE FILE WRITE CANNOT KILL THE SCRIPT.
    # $ErrorActionPreference is Stop, and GitHub caps the step-summary file, so a
    # large grouped failure list that exceeds the cap made Add-Content terminating.
    # That aborted the reporter before its `exit 0`, which changed the job outcome
    # the header promises this script never touches. A summary that cannot be
    # written is a degraded report, not a failed job.
    try {
      Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $Line -ErrorAction Stop
    } catch {
      $script:SummaryWriteFailed = $true
    }
  }
}

# Set by Add-Summary when the step-summary file rejects a write. Reported once at
# the end rather than per line, so a capped file does not produce one warning per
# failure in the digest.
$script:SummaryWriteFailed = $false

# EVERY EXIT PATH IS 0. The body runs inside a script block so an unhandled
# terminating error anywhere in it is caught here instead of propagating a
# nonzero exit code out of an `if: failure()` reporting step.
$reportBody = {

# --blame-hang kills the whole test host when any single test exceeds the hang timeout, and
# --blame writes a Sequence file naming the test that was executing when it died. Everything
# queued behind that test never runs and never appears in the TRX, so a shard truncated this way
# reports a handful of failures and looks like it merely has a handful of failures.
#
# That is the single most misleading state this pipeline can produce: it makes a shard look nearly
# green when most of its suite never executed, and it is why fixing the visible failures kept
# revealing new ones. Detect it and say so, loudly, before anything else.
# Sequence*.xml, not Sequence_*.xml: vstest writes Sequence.xml for a HANG and
# Sequence_<guid>.xml for a CRASH. The old filter matched only the crash spelling, and the element
# name below was wrong for both, so this reporter never identified a victim at all.
$sequenceFiles = Get-ChildItem -Path 'TestResults' -Recurse -Filter 'Sequence*.xml' -ErrorAction SilentlyContinue
$hangVictim = $null
$seqKind = 'hang'
if ($sequenceFiles) {
  foreach ($seq in $sequenceFiles) {
    try {
      [xml]$seqXml = Get-Content $seq.FullName
      # Real vstest output, captured from a forced FailFast under --blame-crash:
      #
      #   <TestSequence>
      #     <Test Name="..." DisplayName="..." Source="....dll" Completed="True"  />
      #     <Test Name="..." DisplayName="..." Source="....dll" Completed="False" />
      #   </TestSequence>
      #
      # The element is <Test>, never <UnitTestElement>. Prefer the entry explicitly marked
      # Completed="False" -- that IS the test that never finished -- and fall back to the last
      # entry only if no such marker is present.
      # The FILENAME distinguishes the two deaths: vstest writes Sequence.xml for a hang and
      # Sequence_<guid>.xml for a crash. They need different wording downstream -- calling a crash
      # a hang sends the reader looking for a deadlock that does not exist.
      if ($seq.Name -match '^Sequence_.+\.xml$') { $seqKind = 'crash' } else { $seqKind = 'hang' }
      $elements = $seqXml.SelectNodes('//Test')
      if ($elements -and $elements.Count -gt 0) {
        $victim = $null
        foreach ($el in $elements) {
          if ($el.Completed -and $el.Completed -eq 'False') { $victim = $el }
        }
        if (-not $victim) { $victim = $elements[$elements.Count - 1] }
        $name = if ($victim.Name) { $victim.Name } elseif ($victim.DisplayName) { $victim.DisplayName } else { $victim.InnerText }
        $hangVictim = "$($victim.Source)::$name".TrimStart(':')
      }
    } catch {
      # A PARSE FAILURE IS NOT A HANG. Assigning a placeholder here left a truthy
      # $hangVictim, so an unreadable sequence file printed the full "THIS SHARD
      # WAS TRUNCATED" banner -- telling the reader most of the suite never ran
      # and nothing in the shard can be trusted, on no evidence at all. The two
      # states are reported separately now.
      $seqParseError = $_.Exception.Message
    }
  }
}

$trxFiles = Get-ChildItem -Path 'TestResults' -Recurse -Filter '*.trx' -ErrorAction SilentlyContinue

if ($seqParseError) {
  # Reported as what it is: the reporter could not read the sequence file. It says
  # nothing about whether the shard completed, so it must not imply truncation.
  Add-Summary '## :warning: Sequence file present but unreadable'
  Add-Summary ''
  Add-Summary 'The hang-victim could not be determined. This does NOT mean the shard was truncated.'
  Add-Summary ''
  Add-Summary ('    ' + $seqParseError)
  Add-Summary ''
}

if ($hangVictim) {
  Add-Summary '## :rotating_light: THIS SHARD WAS TRUNCATED -- the failure list below is INCOMPLETE'
  Add-Summary ''
  if ($seqKind -eq 'crash') {
    Add-Summary 'The test host CRASHED (``--blame-crash``) while executing:'
  } else {
    Add-Summary 'The test host was killed by ``--blame-hang`` (no progress for the timeout) while executing:'
  }
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
  Add-Summary 'results alone. Check the tail of the run log and any `Sequence*.xml`'
  Add-Summary 'blame file for the last test that started.'
  return
}

$ns = @{ t = 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010' }
$failed = New-Object System.Collections.Generic.List[object]

$unparseableTrx = New-Object System.Collections.Generic.List[string]
$hostLifecycleDiagnostics = New-Object System.Collections.Generic.List[string]
$hostLifecycleDiagnosticKeys = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::Ordinal)

foreach ($trx in $trxFiles) {
  # THE REPORTER MUST SURVIVE THE STATE IT EXISTS TO EXPLAIN. With
  # $ErrorActionPreference = 'Stop', an unparseable TRX made this cast a
  # TERMINATING error: the script died here and never printed the digest or
  # reached its own `exit 0`. The same host crash that truncates a TRX is
  # precisely the crash this script was written to report, so a malformed file is
  # the expected input, not an exceptional one.
  $xml = $null
  try {
    [xml]$xml = Get-Content $trx.FullName -Raw
  } catch {
    $unparseableTrx.Add("$($trx.Name): $($_.Exception.Message)")
    continue
  }
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

  # Counters alone cannot prove that the host completed cleanly. VSTest can
  # write executed=total, flush every assertion result, and then append a
  # RunInfo saying that the active run was aborted because the test host
  # crashed. That exact shape previously looked like an ordinary short failure
  # list. Only lifecycle-specific text is promoted here; ordinary xUnit [FAIL]
  # RunInfo records remain represented by the failed-test digest above.
  foreach ($runInfo in @($xml.SelectNodes('//*[local-name()="RunInfo"]'))) {
    $textNode = $runInfo.SelectSingleNode('./*[local-name()="Text"]')
    $runInfoText = if ($textNode) { [string] $textNode.InnerText } else { [string] $runInfo.InnerText }
    $flatRunInfo = ($runInfoText -replace '\s+', ' ').Trim()
    if ($flatRunInfo -match '(?i)\b(?:active\s+)?test\s+run\s+was\s+aborted\b' -or
        $flatRunInfo -match '(?i)\btest\s+host(?:\s+process)?\b.*\b(?:crash(?:ed)?|terminat(?:ed|ion)|abort(?:ed)?|exited\s+unexpectedly)\b') {
      $outcome = if ($runInfo.outcome) { [string] $runInfo.outcome } else { 'Error' }
      $diagnostic = "[$outcome] $flatRunInfo"
      if ($hostLifecycleDiagnosticKeys.Add($diagnostic)) {
        $hostLifecycleDiagnostics.Add($diagnostic)
      }
    }
  }
}

# COUNTERS ARE READ BEFORE THE EMPTY-FAILURE BRANCH, NOT AFTER IT.
# This check used to live below the `$failed.Count -eq 0` early return, so the one
# case it exists to catch could never reach it: a host OOM-killed after 40 passing
# tests and before any failure was recorded leaves a TRX whose Counters read
# executed=40 total=852 and whose failure list is EMPTY. The script then printed
# "the cause is outside the test results themselves", which is the exact opposite
# of the truth -- 812 tests never ran.
# Executed-vs-discovered. xUnit's TRX ResultSummary carries both, and when they disagree the
# shard did not finish -- the same truncation blame-hang causes, but also what a plain host crash
# or an OOM part-way through leaves behind. Reporting the gap turns "this shard has 3 failures"
# into "this shard has 3 failures and 812 tests that never ran", which are very different facts.
# SUMMED ACROSS EVERY TRX, not read from the first. The failure list above
# aggregates all of them, so reading counters from $trxFiles[0] alone put the two
# scopes in disagreement and produced a wrong answer in both directions: a
# complete first file hid a later truncation entirely, and a truncated first file
# reported a "never ran" count for the whole shard that was only ever true of one
# part of it.
$total = 0
$executed = 0
$counterParseErrors = New-Object System.Collections.Generic.List[string]
foreach ($trx in $trxFiles) {
  try {
    [xml]$doc = Get-Content $trx.FullName -Raw
    $counters = $doc.SelectSingleNode('//*[local-name()="Counters"]')
    if ($counters) {
      $total    += [int]$counters.total
      $executed += [int]$counters.executed
    }
  } catch {
    # Reporter, never a gate -- a malformed TRX must not mask the failures we did
    # parse. But it is RECORDED: an empty catch left no way to tell "the counts
    # matched" from "the check never ran", which is the difference between a
    # trustworthy total and an unknown one.
    $counterParseErrors.Add("$($trx.Name): $($_.Exception.Message)")
  }
}

Add-Summary '## Failed test digest'
Add-Summary ''

if ($hostLifecycleDiagnostics.Count -gt 0) {
  Add-Summary ':rotating_light: **The test host terminated abnormally. This shard is INCOMPLETE even though its TRX counters may show every discovered test as executed.**'
  Add-Summary ''
  foreach ($diagnostic in $hostLifecycleDiagnostics) {
    Add-Summary ('    ' + $diagnostic)
  }
  Add-Summary ''
  Add-Summary 'Treat the assertion list below as diagnostic evidence, not proof that the shard completed cleanly.'
  Add-Summary ''
}

if ($failed.Count -eq 0) {
  # The step is only reached on job failure, so an empty failure set here means
  # the job failed for a NON-test reason (coverage upload, a post-step, the
  # runner). Flag that so it is not mistaken for a flake.
  # ORDER MATTERS, AND UNPARSEABLE COMES FIRST. With a TRX that could not be parsed, the
  # failed-test set is UNKNOWN -- so concluding 'the cause is outside the test results' is a
  # statement the script has no evidence for, and the malformed-TRX warning that would have
  # said so sat after the `return` below and never ran.
  if ($unparseableTrx.Count -gt 0) {
    Add-Summary (":warning: **$($unparseableTrx.Count) result file(s) could not be parsed**, so the " +
                 'failed-test set for this shard is UNKNOWN. The digest below is incomplete; this is' +
                 ' NOT evidence that the failure lies outside the test results.')
    foreach ($e in $unparseableTrx) { Add-Summary ('    ' + $e) }
  }
  elseif ($hostLifecycleDiagnostics.Count -gt 0) {
    Add-Summary 'No failed test result was recorded before the host lifecycle failure above.'
  }
  elseif ($total -gt 0 -and $executed -lt $total) {
    Add-Summary (":warning: **Only $executed of $total discovered tests executed -- $($total - $executed) never ran.** " +
                 'This shard is TRUNCATED. No failure was recorded because the host died before' +
                 ' one could be written, not because the suite passed.')
  } else {
    Add-Summary 'The TRX recorded no failed tests, yet the job failed. The cause is'
    Add-Summary 'outside the test results themselves (a post-test step, coverage, or'
    Add-Summary 'the runner). Check the step log directly.'
  }
  if ($counterParseErrors.Count -gt 0) {
    Add-Summary (":warning: **The executed-vs-discovered check was skipped for $($counterParseErrors.Count) result file(s).** " +
                 'Whether this shard was truncated is UNKNOWN.')
    foreach ($e in $counterParseErrors) { Add-Summary ('    ' + $e) }
  }
  return
}

Add-Summary ("**$($failed.Count) failed test(s).** Grouped by error, most-common first.")

if ($total -gt 0 -and $executed -lt $total) {
  Add-Summary ''
  Add-Summary (":warning: **Only $executed of $total discovered tests executed -- $($total - $executed) never ran.** " +
               'This shard is TRUNCATED; the failure list is a floor, not a total.')
}

if ($counterParseErrors.Count -gt 0) {
  Add-Summary ''
  Add-Summary (":warning: **The executed-vs-discovered check was skipped for $($counterParseErrors.Count) result file(s).** " +
               'The totals below may be incomplete.')
  foreach ($e in $counterParseErrors) { Add-Summary ('    ' + $e) }
}

if ($unparseableTrx.Count -gt 0) {
  Add-Summary ''
  Add-Summary (":warning: **$($unparseableTrx.Count) result file(s) could not be parsed** and contributed no failures to the digest below.")
  foreach ($e in $unparseableTrx) { Add-Summary ('    ' + $e) }
}

Add-Summary ''

# Group by error message so 25 failures that share one root cause read as one
# line item with a count, not 25 near-identical rows. This is what turns a wall
# of failures into "LLMTime: 25x reshape 64->32" at a glance.
$byMessage = $failed | Group-Object Message | Sort-Object Count -Descending
foreach ($group in $byMessage) {
  $errorText = if ($group.Name) { $group.Name } else { '(no error message captured)' }

  # FENCED AND CAPPED. Assertion messages routinely contain backticks, asterisks,
  # underscores and pipes; interpolated raw into a `###` heading GitHub renders
  # them as Markdown, so the primary output of this script arrives mangled or
  # partly invisible. An uncapped message also becomes a heading long enough to
  # push the test list off screen.
  #
  # The fence is sized to the longest backtick run in the text, because a
  # three-backtick span cannot contain a three-backtick sequence -- the usual
  # single-backtick wrap breaks on exactly the messages most likely to need it.
  $MaxHeadingChars = 160
  $flat = ($errorText -replace '\s+', ' ').Trim()
  if ($flat.Length -gt $MaxHeadingChars) {
    $flat = $flat.Substring(0, $MaxHeadingChars) + '...'
  }
  $longestRun = 0
  foreach ($m in [regex]::Matches($flat, '`+')) {
    if ($m.Value.Length -gt $longestRun) { $longestRun = $m.Value.Length }
  }
  $fence = '`' * ($longestRun + 1)
  # A span whose content begins or ends with a backtick needs a padding space,
  # which Markdown strips on render.
  $pad = if ($flat.StartsWith('`') -or $flat.EndsWith('`')) { ' ' } else { '' }
  Add-Summary ("### {0}x  {1}{2}{3}{4}{1}" -f $group.Count, $fence, $pad, $flat, $pad)
  Add-Summary ''
  foreach ($item in $group.Group) {
    Add-Summary ("- {0}" -f $item.Name)
  }
  Add-Summary ''
}

# Always succeed: this is a reporter, not a gate.
}

try {
  & $reportBody
} catch {
  # A reporter that throws must still not decide the job. Say what broke, in the
  # log, and leave the real test result standing.
  Write-Host "::warning::report-failed-tests.ps1 could not complete: $($_.Exception.Message)"
} finally {
  if ($script:SummaryWriteFailed) {
    Write-Host '::warning::One or more lines could not be written to GITHUB_STEP_SUMMARY' +
               ' (the file is capped). The console output above is complete.'
  }
}

# Unconditional, and the last statement in the file.
exit 0
