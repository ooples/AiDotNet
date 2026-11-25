<#
.SYNOPSIS
    Classifies one Heavy Timeout chunk as passed, out of resources, host died, or failed (#2087, #2447).

.DESCRIPTION
    Only Passed keeps the lane green, and Passed is earned from the result file, not from the exit code:
    the chunk's TRX must exist, parse, say the run completed, list as many results as its own counters
    claim and at least as many as discovery listed for the chunk, and hold no failure or run-level error.

    Everything else fails the lane (FailsLane):

    - ResourceExhausted: every failed test ran out of memory (OutOfMemoryException under the lane's GC heap
      cap) or disk. Until #2447 this was "Envelope" and did NOT fail the lane, on the reasoning that a
      paper-scale model had outgrown the 16 GB runner. That hid defects for weeks: FLUX parameter IO wrote
      5.9x the model to disk, a detector serialized its backbone twice, an optimizer step built ten
      parameter-sized temporaries. Each was reported green as "exceeds runner". It is kept as its own
      outcome only so the summary says which kind of failure it was.
    - HostDied: the test host or the chunk process did not finish its run. A signal exit (SIGKILL from the
      kernel OOM killer, SIGSEGV from an access violation, SIGABRT), a missing, empty or truncated result
      file, an aborted run, a run-level error such as "Test host process crashed", or a result file that
      holds fewer results than it should.
    - Failed: an assertion, a per-test timeout, the chunk's wall-clock cap, or a run that failed without a
      failed test.

.PARAMETER TrxPath
    The chunk's TRX file.

.PARAMETER ExitCode
    The chunk's dotnet test exit code, or the lane's text for its wall-clock cap ("timeout after N min").

.PARAMETER ExpectedTests
    How many tests discovery listed for the chunk's classes. A result file with fewer results is a host that
    stopped part-way. 0 skips the comparison.

.PARAMETER SelfTest
    Runs the classifier against synthetic TRX files and exits non-zero on any wrong verdict.
#>
[CmdletBinding()]
param(
    [string] $TrxPath,
    [string] $ExitCode,
    [int] $ExpectedTests = 0,
    [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum HeavyChunkOutcome { Passed; ResourceExhausted; HostDied; Failed }

# Exact shapes only: a message that merely mentions OOM somewhere is not evidence the test did nothing else.
$script:ExhaustionPatterns = @(
    '^System\.OutOfMemoryException : '
    '^System\.IO\.IOException : No space left on device'
    "^System\.AggregateException : One or more errors occurred\.( \(Exception of type 'System\.OutOfMemoryException' was thrown\.\))+\s*$"
)

# A "does not throw" test records the exception and asserts it is null, so an out-of-memory error reaches the TRX as
# an assertion whose Actual value is the exception: SegmentationTrainingRobustnessTests.OpenVocabSAM_MultiStepTrain
# failed the 2026-10-19 lane that way. Only that exact shape counts, with the exception as the WHOLE Actual value;
# any other exception recorded the same way is still a failure.
# The assertion's own lines, in order: 'Expected: null' and then an Actual value that IS the exception, so an OOM
# named inside some other exception's message does not count.
$script:RecordedExpectedPattern = '^\s*Expected:\s+null\s*$'
$script:RecordedActualPattern = '^\s*Actual:\s+System\.OutOfMemoryException: '

function Test-ResourceExhaustionMessage([string] $Message) {
    $first = ($Message -split "`r?`n")[0].Trim()
    foreach ($pattern in $script:ExhaustionPatterns) {
        if ($first -match $pattern) { return $true }
    }
    $lines = $Message -split "`r?`n"
    if ($first -eq 'Assert.Null() Failure: Value is not null' -and $lines.Count -ge 3 -and
        $lines[1] -match $script:RecordedExpectedPattern -and $lines[2] -match $script:RecordedActualPattern) {
        return $true
    }
    return $false
}

# What a non-0/1 exit of the chunk means. 128+N is a process ended by signal N; the Windows codes are the
# same deaths as NTSTATUS values, for a lane or a local run on Windows.
function Get-ExitDescription([string] $ExitCode) {
    $code = 0L
    if (-not [long]::TryParse($ExitCode, [ref] $code)) { return "exit $ExitCode" }
    switch ($code) {
        134 { return 'exit 134 (SIGABRT)' }
        137 { return 'exit 137 (SIGKILL: kernel OOM killer or timeout escalation)' }
        139 { return 'exit 139 (SIGSEGV: access violation)' }
        143 { return 'exit 143 (SIGTERM)' }
        -1073741819 { return 'exit 0xC0000005 (access violation)' }
        3221225477 { return 'exit 0xC0000005 (access violation)' }
        -1073741571 { return 'exit 0xC00000FD (stack overflow)' }
        3221225725 { return 'exit 0xC00000FD (stack overflow)' }
        -1073740791 { return 'exit 0xC0000409 (fail fast)' }
        3221226505 { return 'exit 0xC0000409 (fail fast)' }
    }
    if ($code -gt 128 -and $code -lt 256) { return "exit $code (signal $($code - 128))" }
    return "exit $code"
}

function Get-HeavyChunkOutcome {
    param([string] $TrxPath, [string] $ExitCode, [int] $ExpectedTests = 0)

    $result = [pscustomobject]@{
        Outcome = [HeavyChunkOutcome]::Failed; Exhausted = 0; Failures = @(); Reason = ''; FailsLane = $true
    }
    $complete = {
        param([HeavyChunkOutcome] $Outcome, [string] $Reason)
        $result.Outcome = $Outcome
        $result.Reason = $Reason
        $result.FailsLane = $Outcome -ne [HeavyChunkOutcome]::Passed
        return $result
    }

    if ($ExitCode -ne '0' -and $ExitCode -ne '1') {
        # The lane's own wall-clock cap is a slow chunk; any other exit is a process that was ended.
        if ($ExitCode -like 'timeout*') { return & $complete Failed "exit $ExitCode" }
        return & $complete HostDied (Get-ExitDescription $ExitCode)
    }

    # From here the exit code is 0 or 1, and neither is believed without the result file: a host that died
    # after its last test, or a filter that matched nothing, also exits 0.
    if (-not $TrxPath -or -not (Test-Path -LiteralPath $TrxPath)) {
        return & $complete HostDied "no result file (exit $ExitCode)"
    }

    $trx = $null
    try {
        [xml] $trx = Get-Content -LiteralPath $TrxPath -Raw
    }
    catch {
        Write-Verbose "Result file $TrxPath did not parse: $($_.Exception.Message)"
        return & $complete HostDied "truncated or unreadable result file (exit $ExitCode)"
    }
    # An empty or declaration-only file parses to a document with no root element, and reading the
    # namespace off it below would throw out of the whole lane rather than fail this one chunk.
    if ($null -eq $trx -or $null -eq $trx.DocumentElement) {
        return & $complete HostDied "empty result file (exit $ExitCode)"
    }

    $ns = [System.Xml.XmlNamespaceManager]::new($trx.NameTable)
    $ns.AddNamespace('t', $trx.DocumentElement.NamespaceURI)
    $summary = $trx.SelectSingleNode('//t:ResultSummary', $ns)
    if ($null -eq $summary) { return & $complete HostDied 'result file has no run summary' }
    $runOutcome = $summary.GetAttribute('outcome')
    if ($runOutcome -eq 'Aborted') { return & $complete HostDied 'test run aborted' }

    # A crashed or hung host can be recorded only as a run-level error, beside results that all passed or
    # all ran out of memory. xUnit also echoes every failed test there as "[xUnit.net ...] <test> [FAIL]",
    # which the result rows below already judge, so only the other errors are evidence on their own.
    foreach ($info in @($trx.SelectNodes("//t:RunInfo[@outcome='Error']", $ns))) {
        $textNode = $info.SelectSingleNode('t:Text', $ns)
        $text = if ($null -eq $textNode) { '' } else { $textNode.InnerText.Trim() }
        if ($text -notmatch '^\[xUnit\.net [\d:.]+\]\s+\S+ \[FAIL\]$') {
            return & $complete HostDied "run error: $(($text -split "`r?`n")[0])"
        }
    }

    $results = @($trx.SelectNodes('//t:UnitTestResult', $ns))
    $counters = $summary.SelectSingleNode('t:Counters', $ns)
    if ($null -eq $counters) { return & $complete HostDied 'result file has no counters' }
    $total = 0
    if (-not [int]::TryParse($counters.GetAttribute('total'), [ref] $total)) {
        return & $complete HostDied 'result file has no test total'
    }
    if ($results.Count -ne $total) {
        return & $complete HostDied "result file holds $($results.Count) of its $total result(s)"
    }
    if ($ExpectedTests -gt 0 -and $results.Count -lt $ExpectedTests) {
        return & $complete HostDied "$($results.Count) of $ExpectedTests discovered test(s) reported a result"
    }

    # Anything that is not a pass or a skip is a failure, whatever dotnet test exited with.
    $failedResults = @($results | Where-Object { $_.GetAttribute('outcome') -notin @('Passed', 'NotExecuted') })

    if ($ExitCode -eq '0') {
        if ($results.Count -eq 0) { return & $complete Failed 'exit 0 with no test result: nothing ran' }
        if ($failedResults.Count -gt 0) {
            return & $complete Failed "exit 0 with $($failedResults.Count) failed test(s)"
        }
        if ($runOutcome -notin @('Completed', 'Passed')) {
            return & $complete Failed "exit 0 but the run ended '$runOutcome'"
        }
        return & $complete Passed ''
    }

    if ($failedResults.Count -eq 0) {
        # dotnet test exited 1 with nothing failed: the run itself broke.
        return & $complete Failed 'exit 1 with no failed test'
    }

    $other = @()
    foreach ($node in $failedResults) {
        $messageNode = $node.SelectSingleNode('t:Output/t:ErrorInfo/t:Message', $ns)
        $message = if ($null -eq $messageNode) { '' } else { $messageNode.InnerText }
        if ($node.GetAttribute('outcome') -eq 'Failed' -and (Test-ResourceExhaustionMessage $message)) {
            $result.Exhausted++
        }
        else { $other += $node.GetAttribute('testName') }
    }

    $result.Failures = $other
    if ($other.Count -eq 0) {
        return & $complete ResourceExhausted "$($result.Exhausted) test(s) ran out of memory or disk"
    }
    return & $complete Failed "$($other.Count) failed test(s)"
}

function Invoke-SelfTest {
    $root = Join-Path ([IO.Path]::GetTempPath()) ('heavy-chunk-' + [Guid]::NewGuid().ToString('N'))
    [void] [System.IO.Directory]::CreateDirectory($root)
    $problems = [System.Collections.Generic.List[string]]::new()

    # -Total overrides the counter the file claims, for a file that lost results after its summary was written.
    function New-Trx([string] $Name, [string] $RunOutcome, [object[]] $Results, [string[]] $RunErrors = @(), [int] $Total = -1) {
        $rows = foreach ($r in $Results) {
            $message = [System.Security.SecurityElement]::Escape($r.Message)
            "<UnitTestResult testName=`"$($r.Name)`" outcome=`"$($r.Outcome)`"><Output><ErrorInfo><Message>$message</Message></ErrorInfo></Output></UnitTestResult>"
        }
        $infos = foreach ($e in $RunErrors) {
            "<RunInfo computerName=`"runner`" outcome=`"Error`"><Text>$([System.Security.SecurityElement]::Escape($e))</Text></RunInfo>"
        }
        if ($Total -lt 0) { $Total = @($Results).Count }
        $path = Join-Path $root "$Name.trx"
        $xml = "<?xml version=`"1.0`" encoding=`"utf-8`"?><TestRun xmlns=`"http://microsoft.com/schemas/VisualStudio/TeamTest/2010`"><Results>$($rows -join '')</Results><ResultSummary outcome=`"$RunOutcome`"><Counters total=`"$Total`" /><RunInfos>$($infos -join '')</RunInfos></ResultSummary></TestRun>"
        Set-Content -LiteralPath $path -Value $xml -Encoding utf8NoBOM
        return $path
    }

    function Assert-Outcome([string] $Case, [HeavyChunkOutcome] $Expected, [string] $Trx, [string] $Exit, [int] $ExpectedTests = 0) {
        $verdict = Get-HeavyChunkOutcome -TrxPath $Trx -ExitCode $Exit -ExpectedTests $ExpectedTests
        if ($verdict.Outcome -ne $Expected) {
            [void] $problems.Add("${Case}: expected $Expected, got $($verdict.Outcome) ($($verdict.Reason))")
        }
        # The lane's exit follows FailsLane, so it is pinned for every case: only a pass may keep it green.
        if ($verdict.FailsLane -ne ($Expected -ne [HeavyChunkOutcome]::Passed)) {
            [void] $problems.Add("${Case}: FailsLane is $($verdict.FailsLane) for $($verdict.Outcome)")
        }
    }

    $oom = "System.OutOfMemoryException : Exception of type 'System.OutOfMemoryException' was thrown."
    $aggOom = "System.AggregateException : One or more errors occurred. (Exception of type 'System.OutOfMemoryException' was thrown.) (Exception of type 'System.OutOfMemoryException' was thrown.)"
    $disk = "System.IO.IOException : No space left on device : '/tmp/aidotnet-streaming-pool/backing.bin'"
    $recordedOom = "Assert.Null() Failure: Value is not null`nExpected: null`nActual:   System.OutOfMemoryException: Exception of type 'System.OutOfMemoryException' was thrown."
    # A non-OOM Actual whose message carries a diagnostic line shaped like a recorded OOM.
    $recordedOtherWithOomLine = "Assert.Null() Failure: Value is not null`nExpected: null`nActual:   System.ArgumentException: Shapes do not match.`nActual: System.OutOfMemoryException: pool exhausted earlier"
    $recordedOther = "Assert.Null() Failure: Value is not null`nExpected: null`nActual:   System.ArgumentException: Shapes do not match."
    $pass = @{ Name = 'P'; Outcome = 'Passed'; Message = '' }
    $skip = @{ Name = 'S'; Outcome = 'NotExecuted'; Message = '' }
    try {
        # A pass is read from the result file.
        $threePassed = New-Trx 'passed' 'Completed' @($pass, $pass, $skip)
        Assert-Outcome 'passed' Passed $threePassed '0'
        Assert-Outcome 'passed_with_expected_count' Passed $threePassed '0' 3

        # Exit 0 is not a pass on its own (#2447): the host can die after its last test, or run nothing.
        Assert-Outcome 'exit_zero_without_trx_path' HostDied '' '0'
        Assert-Outcome 'exit_zero_missing_trx' HostDied (Join-Path $root 'absent.trx') '0'
        Assert-Outcome 'exit_zero_nothing_ran' Failed (New-Trx 'nothing' 'Completed' @()) '0'
        Assert-Outcome 'exit_zero_fewer_than_discovered' HostDied $threePassed '0' 7
        Assert-Outcome 'exit_zero_fewer_than_counted' HostDied (New-Trx 'short' 'Completed' @($pass, $pass) @() 5) '0'
        Assert-Outcome 'exit_zero_with_failed_row' Failed (New-Trx 'zero-failed' 'Completed' @(
                $pass, @{ Name = 'A'; Outcome = 'Failed'; Message = 'Assert.True() Failure' })) '0'
        Assert-Outcome 'exit_zero_with_host_crash' HostDied (New-Trx 'zero-crash' 'Completed' @($pass) @(
                'The active test run was aborted. Reason: Test host process crashed')) '0'
        Assert-Outcome 'exit_zero_aborted_run' HostDied (New-Trx 'zero-aborted' 'Aborted' @($pass)) '0'

        # A result file cut off mid-write, as a killed host leaves it: a real document minus its tail.
        $truncated = Join-Path $root 'truncated.trx'
        $whole = Get-Content -LiteralPath $threePassed -Raw
        Set-Content -LiteralPath $truncated -Value $whole.Substring(0, [int]($whole.Length * 0.6)) -NoNewline -Encoding utf8NoBOM
        Assert-Outcome 'truncated_trx_exit_zero' HostDied $truncated '0'
        Assert-Outcome 'truncated_trx_exit_one' HostDied $truncated '1'
        # An empty or declaration-only file parses, then has nothing to read a namespace from. It must
        # fail this chunk, not throw out of the lane.
        $empty = Join-Path $root 'empty.trx'
        Set-Content -LiteralPath $empty -Value '' -NoNewline
        Assert-Outcome 'empty_trx' HostDied $empty '1'
        Assert-Outcome 'empty_trx_exit_zero' HostDied $empty '0'
        $declOnly = Join-Path $root 'decl.trx'
        Set-Content -LiteralPath $declOnly -Value '<?xml version="1.0" encoding="utf-8"?>' -Encoding utf8NoBOM
        Assert-Outcome 'declaration_only_trx' HostDied $declOnly '1'
        Assert-Outcome 'missing_trx' HostDied (Join-Path $root 'absent.trx') '1'

        # Out of memory or disk is its own outcome and fails the lane (#2447); it was green as "Envelope".
        Assert-Outcome 'all_oom' ResourceExhausted (New-Trx 'oom' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom },
                @{ Name = 'B'; Outcome = 'Failed'; Message = $aggOom },
                @{ Name = 'C'; Outcome = 'Failed'; Message = $disk },
                @{ Name = 'D'; Outcome = 'Passed'; Message = '' })) '1'
        Assert-Outcome 'oom_with_fail_echo' ResourceExhausted (New-Trx 'echo' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom }) @(
                '[xUnit.net 00:02:07.32]     AiDotNet.Tests.SomeModelTests.Metadata_ShouldExist [FAIL]')) '1'
        Assert-Outcome 'recorded_oom' ResourceExhausted (New-Trx 'recorded' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $recordedOom })) '1'
        Assert-Outcome 'oom_with_host_crash' HostDied (New-Trx 'crash' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom }) @(
                'The active test run was aborted. Reason: Test host process crashed')) '1'
        Assert-Outcome 'oom_then_host_stopped' HostDied (New-Trx 'oom-short' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom })) '1' 4

        Assert-Outcome 'oom_and_assertion' Failed (New-Trx 'mixed' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom },
                @{ Name = 'B'; Outcome = 'Failed'; Message = 'Training did not reduce loss: initial=1, final=2.' })) '1'
        Assert-Outcome 'timeout_is_a_failure' Failed (New-Trx 'timeout' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = 'Test execution timed out after 120000 milliseconds' })) '1'
        Assert-Outcome 'aggregate_with_other_inner' Failed (New-Trx 'agg' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = "System.AggregateException : One or more errors occurred. (Exception of type 'System.OutOfMemoryException' was thrown.) (Index was outside the bounds of the array.)" })) '1'
        Assert-Outcome 'oom_mentioned_in_assertion' Failed (New-Trx 'mention' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = 'Assert.Throws() Failure: expected System.OutOfMemoryException : none thrown' })) '1'
        Assert-Outcome 'recorded_other_exception' Failed (New-Trx 'recorded-other' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $recordedOther })) '1'
        Assert-Outcome 'recorded_other_exception_with_oom_line' Failed (New-Trx 'recorded-other-oom-line' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $recordedOtherWithOomLine })) '1'
        # A row the host never finished is not an out-of-memory test, whatever message it carries.
        Assert-Outcome 'oom_message_on_aborted_row' Failed (New-Trx 'aborted-row' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Aborted'; Message = $oom })) '1'
        Assert-Outcome 'aborted_run' HostDied (New-Trx 'aborted' 'Aborted' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom })) '1'
        Assert-Outcome 'no_failed_test' Failed (New-Trx 'none' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Passed'; Message = '' })) '1'

        # The chunk process itself was ended. The result file, if any, is not consulted.
        Assert-Outcome 'chunk_timeout' Failed (New-Trx 'cap' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom })) 'timeout after 60 min'
        Assert-Outcome 'sigkill_oom_killer' HostDied $threePassed '137'
        Assert-Outcome 'sigsegv_access_violation' HostDied $threePassed '139'
        Assert-Outcome 'sigabrt' HostDied '' '134'
        Assert-Outcome 'windows_access_violation' HostDied '' '-1073741819'
        Assert-Outcome 'unknown_exit' HostDied $threePassed '2'
        foreach ($case in @(@('137', 'SIGKILL'), @('139', 'SIGSEGV'), @('-1073741819', 'access violation'))) {
            $reason = (Get-HeavyChunkOutcome -TrxPath '' -ExitCode $case[0]).Reason
            if ($reason -notmatch $case[1]) { [void] $problems.Add("exit $($case[0]) is described as '$reason'") }
        }
    }
    finally {
        try { [System.IO.Directory]::Delete($root, $true) } catch { Write-Verbose "Could not clean $root" }
    }

    if ($problems.Count -gt 0) {
        Write-Host 'Measure-HeavyChunkOutcome self-test FAILED:'
        foreach ($problem in $problems) { Write-Host "  - $problem" }
        exit 1
    }
    Write-Host 'Measure-HeavyChunkOutcome self-test passed.'
    exit 0
}

if ($SelfTest) { Invoke-SelfTest }

Get-HeavyChunkOutcome -TrxPath $TrxPath -ExitCode $ExitCode -ExpectedTests $ExpectedTests
