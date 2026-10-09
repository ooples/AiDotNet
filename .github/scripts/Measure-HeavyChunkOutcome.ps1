<#
.SYNOPSIS
    Classifies one Heavy Timeout chunk as passed, out of the runner's envelope, or failed (#2087).

.DESCRIPTION
    Paper-scale models in the nightly lane need more memory than the 16 GB hosted runner has: Wan2.2's
    DiT alone is about 4.5B parameters, 18 GB of float weights before gradients and Adam state. Under the
    lane's GC heap cap they throw OutOfMemoryException, and on 2026-10-07 that was 573 of 621 failures,
    so the lane was red every night and the few real defects among them were unreadable.

    A chunk whose every failed test ran out of memory or disk is reported as Envelope, which does not
    fail the lane. Anything else that failed -- an assertion, a timeout, a crashed or aborted host, a
    missing or unreadable result file, an AggregateException holding anything but OOM -- is Failed.

.PARAMETER TrxPath
    The chunk's TRX file.

.PARAMETER ExitCode
    The chunk's dotnet test exit code, or the lane's text for a timeout or SIGKILL.

.PARAMETER SelfTest
    Runs the classifier against synthetic TRX files and exits non-zero on any wrong verdict.
#>
[CmdletBinding()]
param(
    [string] $TrxPath,
    [string] $ExitCode,
    [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum HeavyChunkOutcome { Passed; Envelope; Failed }

# Exact shapes only: a message that merely mentions OOM somewhere is not evidence the test did nothing else.
$script:EnvelopePatterns = @(
    '^System\.OutOfMemoryException : '
    '^System\.IO\.IOException : No space left on device'
    "^System\.AggregateException : One or more errors occurred\.( \(Exception of type 'System\.OutOfMemoryException' was thrown\.\))+\s*$"
)

function Test-EnvelopeMessage([string] $Message) {
    $first = ($Message -split "`r?`n")[0].Trim()
    foreach ($pattern in $script:EnvelopePatterns) {
        if ($first -match $pattern) { return $true }
    }
    return $false
}

function Get-HeavyChunkOutcome {
    param([string] $TrxPath, [string] $ExitCode)

    $result = [pscustomobject]@{ Outcome = [HeavyChunkOutcome]::Failed; Envelope = 0; Failures = @(); Reason = '' }
    if ($ExitCode -eq '0') {
        $result.Outcome = [HeavyChunkOutcome]::Passed
        return $result
    }
    if ($ExitCode -ne '1') {
        $result.Reason = "exit $ExitCode"
        return $result
    }
    if (-not $TrxPath -or -not (Test-Path -LiteralPath $TrxPath)) {
        $result.Reason = 'no result file'
        return $result
    }

    try {
        [xml] $trx = Get-Content -LiteralPath $TrxPath -Raw
    }
    catch {
        $result.Reason = 'unreadable result file'
        return $result
    }
    # An empty or truncated file parses to a document with no root element, and reading the
    # namespace off it below would throw out of the whole lane rather than fail this one chunk.
    if ($null -eq $trx -or $null -eq $trx.DocumentElement) {
        $result.Reason = 'unreadable result file'
        return $result
    }

    $ns = [System.Xml.XmlNamespaceManager]::new($trx.NameTable)
    $ns.AddNamespace('t', $trx.DocumentElement.NamespaceURI)
    $summary = $trx.SelectSingleNode('//t:ResultSummary', $ns)
    if ($null -eq $summary -or $summary.GetAttribute('outcome') -eq 'Aborted') {
        $result.Reason = 'test run aborted'
        return $result
    }

    # A crashed or hung host can be recorded only as a run-level error, beside results that are all
    # envelope failures. xUnit also echoes every failed test there as "[xUnit.net ...] <test> [FAIL]",
    # which the result rows below already judge, so only the other errors are evidence on their own.
    foreach ($info in @($trx.SelectNodes("//t:RunInfo[@outcome='Error']", $ns))) {
        $textNode = $info.SelectSingleNode('t:Text', $ns)
        $text = if ($null -eq $textNode) { '' } else { $textNode.InnerText.Trim() }
        if ($text -notmatch '^\[xUnit\.net [\d:.]+\]\s+\S+ \[FAIL\]$') {
            $result.Reason = "run error: $(($text -split "`r?`n")[0])"
            return $result
        }
    }

    $failedResults = @($trx.SelectNodes("//t:UnitTestResult[@outcome='Failed']", $ns))
    if ($failedResults.Count -eq 0) {
        # dotnet test exited 1 with nothing failed: the run itself broke, which is not the envelope.
        $result.Reason = 'exit 1 with no failed test'
        return $result
    }

    $other = @()
    foreach ($node in $failedResults) {
        $messageNode = $node.SelectSingleNode('t:Output/t:ErrorInfo/t:Message', $ns)
        $message = if ($null -eq $messageNode) { '' } else { $messageNode.InnerText }
        if (Test-EnvelopeMessage $message) { $result.Envelope++ }
        else { $other += $node.GetAttribute('testName') }
    }

    $result.Failures = $other
    if ($other.Count -eq 0) {
        $result.Outcome = [HeavyChunkOutcome]::Envelope
        $result.Reason = "$($result.Envelope) test(s) exceeded the runner's memory or disk"
    }
    else {
        $result.Reason = "$($other.Count) failed test(s)"
    }
    return $result
}

function Invoke-SelfTest {
    $root = Join-Path ([IO.Path]::GetTempPath()) ('heavy-chunk-' + [Guid]::NewGuid().ToString('N'))
    [void] [System.IO.Directory]::CreateDirectory($root)
    $problems = [System.Collections.Generic.List[string]]::new()

    function New-Trx([string] $Name, [string] $RunOutcome, [object[]] $Results, [string[]] $RunErrors = @()) {
        $rows = foreach ($r in $Results) {
            $message = [System.Security.SecurityElement]::Escape($r.Message)
            "<UnitTestResult testName=`"$($r.Name)`" outcome=`"$($r.Outcome)`"><Output><ErrorInfo><Message>$message</Message></ErrorInfo></Output></UnitTestResult>"
        }
        $infos = foreach ($e in $RunErrors) {
            "<RunInfo computerName=`"runner`" outcome=`"Error`"><Text>$([System.Security.SecurityElement]::Escape($e))</Text></RunInfo>"
        }
        $path = Join-Path $root "$Name.trx"
        $xml = "<?xml version=`"1.0`" encoding=`"utf-8`"?><TestRun xmlns=`"http://microsoft.com/schemas/VisualStudio/TeamTest/2010`"><Results>$($rows -join '')</Results><ResultSummary outcome=`"$RunOutcome`"><RunInfos>$($infos -join '')</RunInfos></ResultSummary></TestRun>"
        Set-Content -LiteralPath $path -Value $xml -Encoding utf8NoBOM
        return $path
    }

    function Assert-Outcome([string] $Case, [HeavyChunkOutcome] $Expected, [string] $Trx, [string] $Exit) {
        $actual = (Get-HeavyChunkOutcome -TrxPath $Trx -ExitCode $Exit).Outcome
        if ($actual -ne $Expected) { [void] $problems.Add("${Case}: expected $Expected, got $actual") }
    }

    $oom = "System.OutOfMemoryException : Exception of type 'System.OutOfMemoryException' was thrown."
    $aggOom = "System.AggregateException : One or more errors occurred. (Exception of type 'System.OutOfMemoryException' was thrown.) (Exception of type 'System.OutOfMemoryException' was thrown.)"
    $disk = "System.IO.IOException : No space left on device : '/tmp/aidotnet-streaming-pool/backing.bin'"
    try {
        Assert-Outcome 'exit_zero' Passed '' '0'
        Assert-Outcome 'all_oom' Envelope (New-Trx 'oom' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom },
                @{ Name = 'B'; Outcome = 'Failed'; Message = $aggOom },
                @{ Name = 'C'; Outcome = 'Failed'; Message = $disk },
                @{ Name = 'D'; Outcome = 'Passed'; Message = '' })) '1'
        Assert-Outcome 'oom_with_fail_echo' Envelope (New-Trx 'echo' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom }) @(
                '[xUnit.net 00:02:07.32]     AiDotNet.Tests.SomeModelTests.Metadata_ShouldExist [FAIL]')) '1'
        Assert-Outcome 'oom_with_host_crash' Failed (New-Trx 'crash' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom }) @(
                'The active test run was aborted. Reason: Test host process crashed')) '1'
        Assert-Outcome 'oom_and_assertion' Failed (New-Trx 'mixed' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom },
                @{ Name = 'B'; Outcome = 'Failed'; Message = 'Training did not reduce loss: initial=1, final=2.' })) '1'
        Assert-Outcome 'timeout_is_a_failure' Failed (New-Trx 'timeout' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = 'Test execution timed out after 120000 milliseconds' })) '1'
        Assert-Outcome 'aggregate_with_other_inner' Failed (New-Trx 'agg' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = "System.AggregateException : One or more errors occurred. (Exception of type 'System.OutOfMemoryException' was thrown.) (Index was outside the bounds of the array.)" })) '1'
        Assert-Outcome 'oom_mentioned_in_assertion' Failed (New-Trx 'mention' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = 'Assert.Throws() Failure: expected System.OutOfMemoryException : none thrown' })) '1'
        Assert-Outcome 'aborted_run' Failed (New-Trx 'aborted' 'Aborted' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom })) '1'
        Assert-Outcome 'no_failed_test' Failed (New-Trx 'none' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Passed'; Message = '' })) '1'
        Assert-Outcome 'missing_trx' Failed (Join-Path $root 'absent.trx') '1'
        # A host killed mid-write leaves an empty or root-less file; it parses, then has nothing to
        # read a namespace from. It must fail this chunk, not throw out of the lane.
        $empty = Join-Path $root 'empty.trx'
        Set-Content -LiteralPath $empty -Value '' -NoNewline
        Assert-Outcome 'empty_trx' Failed $empty '1'
        $declOnly = Join-Path $root 'decl.trx'
        Set-Content -LiteralPath $declOnly -Value '<?xml version="1.0" encoding="utf-8"?>' -Encoding utf8NoBOM
        Assert-Outcome 'declaration_only_trx' Failed $declOnly '1'
        Assert-Outcome 'chunk_timeout' Failed (New-Trx 'cap' 'Failed' @(
                @{ Name = 'A'; Outcome = 'Failed'; Message = $oom })) 'timeout after 20 min'
        Assert-Outcome 'sigkill' Failed '' 'SIGKILL (timeout escalation or OOM)'
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

Get-HeavyChunkOutcome -TrxPath $TrxPath -ExitCode $ExitCode
