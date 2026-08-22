<#
.SYNOPSIS
Builds a machine-readable test ledger from TRX files and, when a baseline is
available, evaluates AiDotNet's regression policy.

.DESCRIPTION
The script intentionally treats a shard with missing/unparseable/incomplete
TRX as incomplete. A short failure list from a killed test host is never
allowed to look like an improvement.

The comparison policy is the repository's hybrid rule:
  * explicitly fixed failures must outnumber new failures;
  * incomplete shards must not increase;
  * a previously-green shard may not become red/incomplete; and
  * a new failure on a touched type/test method is a hard regression.

It always writes ledger.json, comparison.json, and summary.md before returning
a policy failure, so the final CI status remains diagnosable from one artifact.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $CurrentResultsPath,

    [string] $BaselineResultsPath,
    [string] $BaselineLedgerPath,

    [Parameter(Mandatory = $true)]
    [string] $OutputDirectory,

    [string] $CurrentSha = $env:GITHUB_SHA,
    [string] $BaselineSha,
    [string] $RepositoryPath = '.',
    [switch] $FailOnPolicy
)

$ErrorActionPreference = 'Stop'
$schemaVersion = 1

function ConvertTo-ShardKey {
    param([string] $Name)
    return ($Name -replace '[\\/:*?"<>|\s-]+', '_').Trim('_')
}

function Get-IntAttribute {
    param($Node, [string] $Name)
    $value = $Node.GetAttribute($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { return 0 }
    $parsed = 0
    if ([int]::TryParse($value, [ref] $parsed)) { return $parsed }
    return 0
}

function Get-ShardContainers {
    param([string] $Root)

    if (-not (Test-Path -LiteralPath $Root)) { return @() }

    $rootItem = Get-Item -LiteralPath $Root
    $children = @(Get-ChildItem -LiteralPath $rootItem.FullName -Directory -ErrorAction SilentlyContinue)
    $rootHasResults = @(Get-ChildItem -LiteralPath $rootItem.FullName -Recurse -Filter '*.trx' -File -ErrorAction SilentlyContinue).Count -gt 0
    $rootHasMetadata = @(Get-ChildItem -LiteralPath $rootItem.FullName -Recurse -Filter 'shard-metadata.json' -File -ErrorAction SilentlyContinue).Count -gt 0

    # actions/download-artifact keeps one directory per artifact. Synthetic/local
    # callers commonly pass one result tree directly, so support both layouts.
    if ($children.Count -eq 0 -or (($rootHasResults -or $rootHasMetadata) -and
        -not ($children | Where-Object { $_.Name -match '^coverage-[0-9a-f]+-' }))) {
        return @($rootItem)
    }

    return @($children | Where-Object {
        @(Get-ChildItem -LiteralPath $_.FullName -Recurse -Filter '*.trx' -File -ErrorAction SilentlyContinue).Count -gt 0 -or
        @(Get-ChildItem -LiteralPath $_.FullName -Recurse -Filter 'shard-metadata.json' -File -ErrorAction SilentlyContinue).Count -gt 0
    })
}

function Read-TestLedger {
    param([string] $Root, [string] $Sha)

    $shards = New-Object System.Collections.Generic.List[object]
    $allTests = New-Object System.Collections.Generic.List[object]

    foreach ($container in @(Get-ShardContainers $Root)) {
        $metadataFile = Get-ChildItem -LiteralPath $container.FullName -Recurse -Filter 'shard-metadata.json' -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        $metadata = $null
        $metadataError = $null
        if ($metadataFile) {
            try { $metadata = Get-Content -LiteralPath $metadataFile.FullName -Raw | ConvertFrom-Json }
            catch { $metadataError = $_.Exception.Message }
        }

        $displayName = if ($metadata -and $metadata.shard) { [string] $metadata.shard } else { $container.Name }
        $key = if ($metadata -and $metadata.slug) {
            [string] $metadata.slug
        } else {
            ConvertTo-ShardKey ($container.Name -replace '^coverage-[0-9a-f]+-', '')
        }

        $trxFiles = @(Get-ChildItem -LiteralPath $container.FullName -Recurse -Filter '*.trx' -File -ErrorAction SilentlyContinue)
        $parseErrors = New-Object System.Collections.Generic.List[string]
        if ($metadataError) { $parseErrors.Add("shard-metadata.json: $metadataError") }
        $total = 0
        $executed = 0
        $notExecuted = 0
        $aborted = 0
        $failedCounter = 0

        foreach ($trx in $trxFiles) {
            try {
                [xml] $document = Get-Content -LiteralPath $trx.FullName -Raw
                $counters = $document.SelectSingleNode('//*[local-name()="Counters"]')
                if ($counters) {
                    $total += Get-IntAttribute $counters 'total'
                    $executed += Get-IntAttribute $counters 'executed'
                    $notExecuted += Get-IntAttribute $counters 'notExecuted'
                    $aborted += Get-IntAttribute $counters 'aborted'
                    $failedCounter += Get-IntAttribute $counters 'failed'
                } else {
                    $parseErrors.Add("$($trx.Name): no ResultSummary/Counters element")
                }

                $definitions = @{}
                foreach ($unitTest in @($document.SelectNodes('//*[local-name()="UnitTest"]'))) {
                    $method = $unitTest.SelectSingleNode('./*[local-name()="TestMethod"]')
                    if ($method -and $unitTest.id) {
                        $className = [string] $method.className
                        $methodName = [string] $method.name
                        $definitions[[string] $unitTest.id] = if ($className) { "$className.$methodName" } else { $methodName }
                    }
                }

                foreach ($result in @($document.SelectNodes('//*[local-name()="UnitTestResult"]'))) {
                    $display = [string] $result.testName
                    $fullyQualified = $definitions[[string] $result.testId]
                    if ([string]::IsNullOrWhiteSpace($fullyQualified)) { $fullyQualified = $display }
                    $identity = if ($display -and $display -ne $fullyQualified) {
                        "$fullyQualified::$display"
                    } else {
                        $fullyQualified
                    }
                    if ([string]::IsNullOrWhiteSpace($identity)) {
                        $identity = "unknown::$($trx.Name)::$($result.executionId)"
                    }

                    $messageNode = $result.SelectSingleNode('.//*[local-name()="ErrorInfo"]/*[local-name()="Message"]')
                    $message = if ($messageNode) { ([string] $messageNode.InnerText -split "`r?`n")[0].Trim() } else { '' }
                    $allTests.Add([PSCustomObject]@{
                        identity = $identity
                        fullyQualifiedName = $fullyQualified
                        displayName = $display
                        outcome = [string] $result.outcome
                        shard = $key
                        duration = [string] $result.duration
                        message = $message
                    })
                }
            } catch {
                $parseErrors.Add("$($trx.Name): $($_.Exception.Message)")
            }
        }

        $shardTests = @($allTests | Where-Object shard -eq $key)
        $failures = @($shardTests | Where-Object outcome -eq 'Failed')
        $distinctFailures = @($failures | Group-Object identity | ForEach-Object { $_.Group[0] })
        $passedIds = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::Ordinal)
        foreach ($passed in @($shardTests | Where-Object outcome -eq 'Passed')) {
            [void] $passedIds.Add([string] $passed.identity)
        }
        $rerunPassedFailures = @($distinctFailures | Where-Object {
            $passedIds.Contains([string] $_.identity)
        })
        $confirmedFailures = @($distinctFailures | Where-Object {
            -not $passedIds.Contains([string] $_.identity)
        })
        $notExecutedResults = @($allTests | Where-Object {
            $_.shard -eq $key -and $_.outcome -eq 'NotExecuted'
        }).Count
        $accountedNotExecuted = [Math]::Max($notExecuted, $notExecutedResults)
        $missingTrx = $trxFiles.Count -eq 0
        # VSTest intentionally records skipped tests as notExecuted. Those are fully accounted
        # results, not truncation. A killed host leaves tests unaccounted (or marks them aborted),
        # which is the incomplete condition the policy needs to catch.
        $counterGap = $total -gt 0 -and ($executed + $accountedNotExecuted) -lt $total
        $abnormalTermination = $aborted -gt 0
        $stepOutcome = if ($metadata -and $metadata.testStepOutcome) { [string] $metadata.testStepOutcome } else { '' }
        $nonTestFailure = $stepOutcome -and $stepOutcome -ne 'success' -and $failures.Count -eq 0
        $incomplete = $missingTrx -or $parseErrors.Count -gt 0 -or $counterGap -or
            $abnormalTermination -or $nonTestFailure
        $status = if ($incomplete) { 'Incomplete' } elseif ($failures.Count -gt 0 -or $failedCounter -gt 0) { 'Failed' } else { 'Passed' }
        $policyStatus = if ($incomplete) {
            'Incomplete'
        } elseif ($confirmedFailures.Count -gt 0 -or $failedCounter -gt $distinctFailures.Count) {
            'Failed'
        } else {
            'Passed'
        }

        $shards.Add([PSCustomObject]@{
            key = $key
            name = $displayName
            status = $status
            policyStatus = $policyStatus
            total = $total
            executed = $executed
            notExecuted = $accountedNotExecuted
            aborted = $aborted
            failed = [Math]::Max($failures.Count, $failedCounter)
            confirmedFailed = $confirmedFailures.Count
            rerunPassedFailures = $rerunPassedFailures.Count
            missingTrx = $missingTrx
            parseErrors = @($parseErrors)
            testStepOutcome = $stepOutcome
        })
    }

    $distinctTests = @($allTests | Sort-Object identity, outcome, shard -Unique)
    return [PSCustomObject]@{
        schemaVersion = $schemaVersion
        sha = $Sha
        generatedUtc = [DateTime]::UtcNow.ToString('o')
        shards = @($shards | Sort-Object key)
        tests = $distinctTests
    }
}

function Read-LedgerFile {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Baseline ledger not found: $Path" }
    $ledger = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ([int] $ledger.schemaVersion -ne $schemaVersion) {
        throw "Unsupported baseline ledger schema '$($ledger.schemaVersion)'; expected '$schemaVersion'."
    }
    return $ledger
}

function Get-TouchedTokens {
    param([string] $Repo, [string] $Base, [string] $Head)

    $tokens = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::OrdinalIgnoreCase)
    if (-not $Base -or -not $Head -or -not (Test-Path -LiteralPath (Join-Path $Repo '.git'))) { return @() }

    $generic = @('Dispose', 'Initialize', 'CreateNetwork', 'GetParameters', 'SetParameters',
        'UpdateParameters', 'Forward', 'Backward', 'Predict', 'Train', 'Clone')
    try {
        $files = & git -C $Repo diff --name-only "$Base...$Head" -- '*.cs'
        if ($LASTEXITCODE -ne 0) { throw "git diff could not compare $Base...$Head" }
        foreach ($file in @($files)) {
            $name = [IO.Path]::GetFileNameWithoutExtension([string] $file)
            if ($name.Length -ge 5) { [void] $tokens.Add($name) }
        }

        $diff = & git -C $Repo diff --unified=0 "$Base...$Head" -- '*.cs'
        if ($LASTEXITCODE -ne 0) { throw "git diff could not inspect $Base...$Head" }
        $currentFile = $null
        foreach ($diffLine in @($diff)) {
            if ($diffLine -match '^\+\+\+ b/(.+\.cs)$') {
                $currentFile = Join-Path $Repo $Matches[1]
                continue
            }

            if ($diffLine -match '^@@ -[^ ]+ \+(\d+)' -and $currentFile -and
                (Test-Path -LiteralPath $currentFile)) {
                # Git's built-in C# hunk context commonly names only the enclosing class. Walk
                # backward from the changed line to capture the actual test/method containing the
                # edit, so a common base-test change is still recognized as touched when the TRX
                # identity belongs to a generated derived fixture.
                $lineNumber = [int] $Matches[1]
                $source = @(Get-Content -LiteralPath $currentFile)
                $lowerBound = [Math]::Max(0, $lineNumber - 201)
                for ($index = [Math]::Min($source.Count - 1, $lineNumber - 1); $index -ge $lowerBound; $index--) {
                    if ($source[$index] -match '^\s*(?:(?:public|protected|private|internal|static|virtual|override|sealed|async|partial|new)\s+)+(?:[A-Za-z_][A-Za-z0-9_<>,.?\[\]]*\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\(') {
                        $methodName = $Matches[1]
                        if ($methodName.Length -ge 5 -and $generic -notcontains $methodName) {
                            [void] $tokens.Add($methodName)
                        }
                        break
                    }
                }
            }
        }
        foreach ($line in @($diff | Where-Object { $_ -match '^[+-](?![+-])' })) {
            foreach ($match in [regex]::Matches($line, '\b(?:class|struct|interface|record)\s+([A-Za-z_][A-Za-z0-9_]*)|\b(?:Task|ValueTask|void|bool|double|float|int|long|Tensor<[^>]+>)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')) {
                $token = if ($match.Groups[1].Success) { $match.Groups[1].Value } else { $match.Groups[2].Value }
                if ($token.Length -ge 5 -and $generic -notcontains $token) { [void] $tokens.Add($token) }
            }
        }
    } catch {
        Write-Warning "Could not derive touched test/type tokens: $($_.Exception.Message)"
    }
    return @($tokens | Sort-Object)
}

function Test-MatchesToken {
    param($Failure, [string[]] $Tokens)
    $haystack = "$($Failure.identity) $($Failure.fullyQualifiedName) $($Failure.displayName)"
    foreach ($token in $Tokens) {
        if ($haystack.IndexOf($token, [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $true }
    }
    return $false
}

function Write-JsonFile {
    param($Value, [string] $Path)
    $Value | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Get-LedgerStatistics {
    param($Ledger)

    $failures = @($Ledger.tests | Where-Object outcome -eq 'Failed')
    return [PSCustomObject]@{
        shardCount = @($Ledger.shards).Count
        passedShards = @($Ledger.shards | Where-Object status -eq 'Passed').Count
        failedShards = @($Ledger.shards | Where-Object status -eq 'Failed').Count
        incompleteShards = @($Ledger.shards | Where-Object status -eq 'Incomplete').Count
        policyPassedShards = @($Ledger.shards | Where-Object {
            $status = if ($_.PSObject.Properties['policyStatus']) { $_.policyStatus } else { $_.status }
            $status -eq 'Passed'
        }).Count
        policyFailedShards = @($Ledger.shards | Where-Object {
            $status = if ($_.PSObject.Properties['policyStatus']) { $_.policyStatus } else { $_.status }
            $status -eq 'Failed'
        }).Count
        rerunPassedFailures = [int](@($Ledger.shards | Measure-Object -Property rerunPassedFailures -Sum).Sum)
        reportedFailureResults = [int](@($Ledger.shards | Measure-Object -Property failed -Sum).Sum)
        distinctFailures = @($failures | Group-Object identity).Count
        totalTests = [int](@($Ledger.shards | Measure-Object -Property total -Sum).Sum)
        executedTests = [int](@($Ledger.shards | Measure-Object -Property executed -Sum).Sum)
    }
}

function Get-FailureCategories {
    param([object[]] $Failures)

    return @($Failures |
        Group-Object {
            if ([string]::IsNullOrWhiteSpace($_.message)) { return '<no error message>' }
            # Values, durations, indices and seeds differ across tests even when the assertion and
            # root cause are identical. Normalize numeric literals so one loss-domain bug across
            # many model families appears as one category instead of dozens of one-off messages.
            return ([string] $_.message) -replace '(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?', '<n>'
        } |
        Sort-Object -Property @{ Expression = 'Count'; Descending = $true }, @{ Expression = 'Name'; Ascending = $true } |
        ForEach-Object {
            [PSCustomObject]@{
                message = [string] $_.Name
                count = $_.Count
                examples = @($_.Group | Select-Object -First 5 | ForEach-Object { [string] $_.identity })
            }
        })
}

function ConvertTo-MarkdownCell {
    param([string] $Value)
    if ($null -eq $Value) { return '' }
    return (($Value -replace '\|', '\|') -replace "`r?`n", ' ').Trim()
}

function Add-MarkdownList {
    param([System.Collections.Generic.List[string]] $Lines, [string] $Heading, [object[]] $Items, [int] $Limit = 100)
    $Lines.Add("### $Heading")
    $Lines.Add('')
    if (-not $Items -or $Items.Count -eq 0) {
        $Lines.Add('_None._')
        $Lines.Add('')
        return
    }
    foreach ($item in @($Items | Select-Object -First $Limit)) { $Lines.Add("- $item") }
    if ($Items.Count -gt $Limit) { $Lines.Add("- ...and $($Items.Count - $Limit) more (see comparison.json)") }
    $Lines.Add('')
}

New-Item -Path $OutputDirectory -ItemType Directory -Force | Out-Null
$ledgerPath = Join-Path $OutputDirectory 'ledger.json'
$comparisonPath = Join-Path $OutputDirectory 'comparison.json'
$summaryPath = Join-Path $OutputDirectory 'summary.md'
$failureCsvPath = Join-Path $OutputDirectory 'failures.csv'
$shardCsvPath = Join-Path $OutputDirectory 'shards.csv'

$current = Read-TestLedger -Root $CurrentResultsPath -Sha $CurrentSha
Write-JsonFile $current $ledgerPath
$current.shards |
    Select-Object key, name, status, policyStatus, total, executed, notExecuted, aborted,
        failed, confirmedFailed, rerunPassedFailures, missingTrx, testStepOutcome |
    Export-Csv -LiteralPath $shardCsvPath -NoTypeInformation -Encoding utf8
$current.tests |
    Where-Object outcome -eq 'Failed' |
    Select-Object identity, fullyQualifiedName, displayName, shard, duration, message |
    Export-Csv -LiteralPath $failureCsvPath -NoTypeInformation -Encoding utf8

$currentFailures = @($current.tests | Where-Object outcome -eq 'Failed' | Group-Object identity | ForEach-Object { $_.Group[0] })
$currentPassIds = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::Ordinal)
foreach ($passed in @($current.tests | Where-Object outcome -eq 'Passed')) {
    [void] $currentPassIds.Add([string] $passed.identity)
}
$currentRerunPassedFailures = @($currentFailures | Where-Object {
    $currentPassIds.Contains([string] $_.identity)
})
$currentIncomplete = @($current.shards | Where-Object status -eq 'Incomplete')
$currentStats = Get-LedgerStatistics $current
$currentCategories = @(Get-FailureCategories $currentFailures)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# Test regression analysis')
$lines.Add('')

$baseline = $null
if ($BaselineLedgerPath) { $baseline = Read-LedgerFile $BaselineLedgerPath }
elseif ($BaselineResultsPath) { $baseline = Read-TestLedger -Root $BaselineResultsPath -Sha $BaselineSha }

if (-not $baseline) {
    $summary = [PSCustomObject]@{
        mode = 'inventory'
        currentSha = $CurrentSha
        counts = $currentStats
        failureCategories = $currentCategories
        policyPassed = $true
    }
    $lines.Add("Current master ledger: **$($currentStats.shardCount) shards**, **$($currentStats.passedShards) passed**, **$($currentStats.failedShards) failed**, **$($currentStats.incompleteShards) incomplete**.")
    $lines.Add('')
    $lines.Add("The TRX files report **$($currentStats.reportedFailureResults) failing results** representing **$($currentStats.distinctFailures) distinct failing tests**.")
    $lines.Add('')
    $lines.Add('This push establishes the TRX baseline artifact used by later pull requests.')
    $lines.Add('')
    $lines.Add('## Failure categories')
    $lines.Add('')
    $lines.Add('| Count | First error line |')
    $lines.Add('|---:|---|')
    foreach ($category in @($currentCategories | Select-Object -First 30)) {
        $lines.Add("| $($category.count) | $(ConvertTo-MarkdownCell $category.message) |")
    }
    if ($currentCategories.Count -eq 0) { $lines.Add('| 0 | _None_ |') }
    Write-JsonFile $summary $comparisonPath
} else {
    $baselineFailures = @($baseline.tests | Where-Object outcome -eq 'Failed' | Group-Object identity | ForEach-Object { $_.Group[0] })
    $baselineFailureIds = @{}
    foreach ($failure in $baselineFailures) { $baselineFailureIds[[string] $failure.identity] = $failure }
    $currentFailureIds = @{}
    foreach ($failure in $currentFailures) { $currentFailureIds[[string] $failure.identity] = $failure }

    $persistent = @($currentFailures | Where-Object { $baselineFailureIds.ContainsKey([string] $_.identity) })
    $newFailures = @($currentFailures | Where-Object { -not $baselineFailureIds.ContainsKey([string] $_.identity) })
    $rerunPassedNew = @($newFailures | Where-Object {
        $currentPassIds.Contains([string] $_.identity)
    })
    $confirmedNew = @($newFailures | Where-Object {
        -not $currentPassIds.Contains([string] $_.identity)
    })
    $fixed = @($baselineFailures | Where-Object {
        if ($currentFailureIds.ContainsKey([string] $_.identity)) { return $false }
        $id = [string] $_.identity
        return @($current.tests | Where-Object { $_.identity -eq $id -and $_.outcome -eq 'Passed' }).Count -gt 0
    })
    $notObserved = @($baselineFailures | Where-Object {
        -not $currentFailureIds.ContainsKey([string] $_.identity) -and
        -not (@($fixed | Where-Object identity -eq $_.identity).Count -gt 0)
    })

    $baselineShardMap = @{}
    foreach ($shard in $baseline.shards) { $baselineShardMap[[string] $shard.key] = $shard }
    $currentShardMap = @{}
    foreach ($shard in $current.shards) { $currentShardMap[[string] $shard.key] = $shard }

    # An absent artifact is an incomplete shard, even when that shard was already red on master.
    # Without this synthetic entry, dropping a red artifact also drops its failures and can make a
    # killed run look like an improvement. Baseline green shards are additionally caught below.
    $missingCurrentShards = @($baseline.shards | Where-Object {
        -not $currentShardMap.ContainsKey([string] $_.key)
    } | ForEach-Object {
        [PSCustomObject]@{
            key = [string] $_.key
            name = [string] $_.name
            status = 'Missing'
            policyStatus = 'Missing'
            total = [int] $_.total
            executed = 0
            notExecuted = 0
            aborted = 0
            failed = 0
            confirmedFailed = 0
            rerunPassedFailures = 0
            missingTrx = $true
            parseErrors = @('No current artifact was uploaded for this baseline shard.')
            testStepOutcome = 'missing'
        }
    })

    $greenToRed = New-Object System.Collections.Generic.List[object]
    foreach ($entry in $baselineShardMap.GetEnumerator()) {
        $beforeStatus = if ($entry.Value.PSObject.Properties['policyStatus']) {
            [string] $entry.Value.policyStatus
        } else { [string] $entry.Value.status }
        if ($beforeStatus -ne 'Passed') { continue }
        $now = $currentShardMap[$entry.Key]
        $nowStatus = if (-not $now) {
            'Missing'
        } elseif ($now.PSObject.Properties['policyStatus']) {
            [string] $now.policyStatus
        } else { [string] $now.status }
        if ($nowStatus -ne 'Passed') {
            $greenToRed.Add([PSCustomObject]@{
                key = $entry.Key
                name = [string] $entry.Value.name
                currentStatus = $nowStatus
            })
        }
    }

    $touchedTokens = @(Get-TouchedTokens -Repo $RepositoryPath -Base $BaselineSha -Head $CurrentSha)
    $touchedNew = @($confirmedNew | Where-Object { Test-MatchesToken $_ $touchedTokens })
    $baselineIncomplete = @($baseline.shards | Where-Object status -eq 'Incomplete')
    $effectiveCurrentIncomplete = @($currentIncomplete) + @($missingCurrentShards)
    $baselineStats = Get-LedgerStatistics $baseline
    $baselineCategories = @(Get-FailureCategories $baselineFailures)
    $greenToRedArray = @($greenToRed | ForEach-Object { $_ })
    # A missing result never earns credit as a fix. The verified balance shrinks only when explicit
    # current passes for baseline failures outnumber genuinely new failures.
    $netImproved = $fixed.Count -gt $confirmedNew.Count
    $incompleteNotIncreased = $effectiveCurrentIncomplete.Count -le $baselineIncomplete.Count
    $policyPassed = $netImproved -and $incompleteNotIncreased -and $greenToRed.Count -eq 0 -and $touchedNew.Count -eq 0

    $resolvedBaselineSha = [string] $baseline.sha
    if ($BaselineSha) { $resolvedBaselineSha = [string] $BaselineSha }
    $comparison = [PSCustomObject]@{
        mode = 'comparison'
        currentSha = $CurrentSha
        baselineSha = $resolvedBaselineSha
        policyPassed = $policyPassed
        criteria = [PSCustomObject]@{
            verifiedFailureBalanceShrank = $netImproved
            incompleteShardsDidNotIncrease = $incompleteNotIncreased
            noPreviouslyGreenShardRegressed = $greenToRed.Count -eq 0
            noTouchedSurfaceRegression = $touchedNew.Count -eq 0
        }
        counts = [PSCustomObject]@{
            baselineShards = $baselineStats.shardCount
            currentShards = $currentStats.shardCount
            baselinePassedShards = $baselineStats.passedShards
            currentPassedShards = $currentStats.passedShards
            baselineFailedShards = $baselineStats.failedShards
            currentFailedShards = $currentStats.failedShards
            baselinePolicyFailedShards = $baselineStats.policyFailedShards
            currentPolicyFailedShards = $currentStats.policyFailedShards
            baselineDistinctFailures = $baselineFailures.Count
            currentDistinctFailures = $currentFailures.Count
            baselineReportedFailureResults = $baselineStats.reportedFailureResults
            currentReportedFailureResults = $currentStats.reportedFailureResults
            netFailureDelta = $currentFailures.Count - $baselineFailures.Count
            persistentFailures = $persistent.Count
            fixedFailures = $fixed.Count
            newFailures = $newFailures.Count
            confirmedNewFailures = $confirmedNew.Count
            rerunPassedNewFailures = $rerunPassedNew.Count
            baselineFailuresNotObserved = $notObserved.Count
            baselineIncompleteShards = $baselineIncomplete.Count
            currentIncompleteShards = $effectiveCurrentIncomplete.Count
            missingCurrentShardArtifacts = $missingCurrentShards.Count
            greenToRedShards = $greenToRed.Count
            touchedNewFailures = $touchedNew.Count
        }
        greenToRedShards = $greenToRedArray
        newFailures = @($newFailures)
        confirmedNewFailures = @($confirmedNew)
        rerunPassedNewFailures = @($rerunPassedNew)
        touchedNewFailures = @($touchedNew)
        fixedFailures = @($fixed)
        persistentFailures = @($persistent)
        baselineFailuresNotObserved = @($notObserved)
        currentIncompleteShards = @($effectiveCurrentIncomplete)
        missingCurrentShardArtifacts = @($missingCurrentShards)
        baselineFailureCategories = $baselineCategories
        currentFailureCategories = $currentCategories
        touchedTokens = $touchedTokens
    }
    Write-JsonFile $comparison $comparisonPath

    $verdict = if ($policyPassed) { 'PASSED' } else { 'FAILED' }
    $lines.Add("## Policy $verdict")
    $lines.Add('')
    $lines.Add('| Metric | Baseline | Current | Delta |')
    $lines.Add('|---|---:|---:|---:|')
    $lines.Add("| Passed shards | $($baselineStats.passedShards) | $($currentStats.passedShards) | $($currentStats.passedShards - $baselineStats.passedShards) |")
    $lines.Add("| Failed shards | $($baselineStats.failedShards) | $($currentStats.failedShards) | $($currentStats.failedShards - $baselineStats.failedShards) |")
    $lines.Add("| Reproducibly failed shards after targeted retry | $($baselineStats.policyFailedShards) | $($currentStats.policyFailedShards) | $($currentStats.policyFailedShards - $baselineStats.policyFailedShards) |")
    $lines.Add("| Distinct failing tests | $($baselineFailures.Count) | $($currentFailures.Count) | $($currentFailures.Count - $baselineFailures.Count) |")
    $lines.Add("| Reported failing results | $($baselineStats.reportedFailureResults) | $($currentStats.reportedFailureResults) | $($currentStats.reportedFailureResults - $baselineStats.reportedFailureResults) |")
    $lines.Add("| Incomplete/missing shards | $($baselineIncomplete.Count) | $($effectiveCurrentIncomplete.Count) | $($effectiveCurrentIncomplete.Count - $baselineIncomplete.Count) |")
    $lines.Add('')
    $lines.Add('| Acceptance criterion | Result |')
    $lines.Add('|---|---|')
    $lines.Add("| Verified failure balance shrinks (explicit fixes > new) | $(if ($netImproved) { 'PASS' } else { 'FAIL' }) |")
    $lines.Add("| Incomplete shards do not increase | $(if ($incompleteNotIncreased) { 'PASS' } else { 'FAIL' }) |")
    $lines.Add("| Previously-green shards stay green | $(if ($greenToRed.Count -eq 0) { 'PASS' } else { 'FAIL' }) |")
    $lines.Add("| No new failure on a touched type/test | $(if ($touchedNew.Count -eq 0) { 'PASS' } else { 'FAIL' }) |")
    $lines.Add('')
    $lines.Add("Persistent: **$($persistent.Count)**; fixed and explicitly passing: **$($fixed.Count)**; new: **$($newFailures.Count)**; baseline failures not observed because their result/shard is missing: **$($notObserved.Count)**.")
    $lines.Add("Of the new failures, **$($confirmedNew.Count)** reproduced and **$($rerunPassedNew.Count)** explicitly passed the targeted retry.")
    $lines.Add('')

    $lines.Add('## Current failure categories')
    $lines.Add('')
    $lines.Add('| Count | First error line |')
    $lines.Add('|---:|---|')
    foreach ($category in @($currentCategories | Select-Object -First 30)) {
        $lines.Add("| $($category.count) | $(ConvertTo-MarkdownCell $category.message) |")
    }
    if ($currentCategories.Count -eq 0) { $lines.Add('| 0 | _None_ |') }
    $lines.Add('')

    Add-MarkdownList $lines 'Previously-green shards that regressed' @($greenToRedArray | ForEach-Object { "$($_.name) -> $($_.currentStatus)" })
    Add-MarkdownList $lines 'New failures on the touched surface' @($touchedNew | ForEach-Object { "$($_.identity) [$($_.shard)]" })
    Add-MarkdownList $lines 'Confirmed new failures' @($confirmedNew | ForEach-Object { "$($_.identity) [$($_.shard)]" })
    Add-MarkdownList $lines 'One-run new failures that passed targeted retry' @($rerunPassedNew | ForEach-Object { "$($_.identity) [$($_.shard)]" })
    Add-MarkdownList $lines 'Fixed failures (explicit pass required)' @($fixed | ForEach-Object { "$($_.identity) [$($_.shard)]" })
    Add-MarkdownList $lines 'Incomplete or missing current shards' @($effectiveCurrentIncomplete | ForEach-Object { "$($_.name): executed $($_.executed)/$($_.total), status $($_.testStepOutcome)" })
}

$lines | Set-Content -LiteralPath $summaryPath -Encoding utf8
Get-Content -LiteralPath $summaryPath | Write-Host
if ($env:GITHUB_STEP_SUMMARY) {
    try { Get-Content -LiteralPath $summaryPath | Add-Content -LiteralPath $env:GITHUB_STEP_SUMMARY }
    catch { Write-Warning "Could not write GitHub step summary: $($_.Exception.Message)" }
}

if ($FailOnPolicy -and $baseline) {
    $result = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
    if (-not $result.policyPassed) { exit 1 }
}
