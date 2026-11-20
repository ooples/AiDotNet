# Pure policy shared by the planner, receipt producer, reconciler and negative fixtures.
Set-StrictMode -Version Latest
. "$PSScriptRoot/../../.github/scripts/Get-ShardCoveragePolicy.ps1"

function ConvertTo-FailoverCanonicalValue {
    param([AllowNull()] $Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [System.Collections.IDictionary]) {
        $result = [ordered]@{}
        [string[]]$keys = @($Value.Keys)
        [Array]::Sort($keys, [StringComparer]::Ordinal)
        foreach ($key in $keys) {
            $result[[string]$key] = ConvertTo-FailoverCanonicalValue $Value[$key]
        }
        return $result
    }
    if ($Value -is [pscustomobject]) {
        $result = [ordered]@{}
        [string[]]$keys = @($Value.PSObject.Properties.Name)
        [Array]::Sort($keys, [StringComparer]::Ordinal)
        foreach ($key in $keys) {
            $result[$key] = ConvertTo-FailoverCanonicalValue $Value.$key
        }
        return $result
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        return ,@($Value | ForEach-Object { ConvertTo-FailoverCanonicalValue $_ })
    }
    return $Value
}

function Get-FailoverDefinitionHash {
    param([Parameter(Mandatory)] $Shard)
    $definition = [ordered]@{}
    foreach ($property in $Shard.PSObject.Properties) {
        if ($property.Name -notin @('runner', 'failoverOriginalJobId', 'failoverAttempt')) {
            $definition[$property.Name] = $property.Value
        }
    }
    $json = ConvertTo-Json -InputObject (ConvertTo-FailoverCanonicalValue $definition) -Depth 100 -Compress
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($json))).ToLowerInvariant()
}

function Get-FailoverSlug {
    param([Parameter(Mandatory)] [string] $Name)
    return $Name -replace '[\\/:*?"<>|\s-]+', '_'
}

function Get-FailoverProperty {
    param($Object, [string] $Name, $Default = $null)
    if ($null -ne $Object -and $Object.PSObject.Properties[$Name]) { return $Object.$Name }
    return $Default
}

function Test-IsSelfHostedFailoverJob {
    param([Parameter(Mandatory)] $Job)
    $labels = @(Get-FailoverProperty $Job 'labels' @())
    if ($labels -contains 'ubuntu-latest' -or $labels -contains 'ubuntu-24.04' -or $labels -contains 'ubuntu-22.04') {
        return $false
    }
    if (@($labels | Where-Object { $_ -match '(?i)windows|macos|arm64' }).Count -gt 0) {
        return $false
    }
    return ($labels -contains 'aidotnet-cpu' -or $labels -contains 'self-hosted')
}

function Test-FailoverInfrastructureFailure {
    param([Parameter(Mandatory)] $Job, [AllowEmptyCollection()] [object[]] $Annotations = @())
    if (Test-FailoverSuccessfulJob $Job) { return $false }
    $conclusion = [string](Get-FailoverProperty $Job 'conclusion' '')
    if ($conclusion -in @('success', 'skipped', 'neutral', '', 'unknown')) { return $false }
    $messages = @($Annotations | ForEach-Object { [string]$_.message })
    if (@($messages | Where-Object { $_ -match '(?i)manually cancel|supersed' }).Count -gt 0) { return $false }
    if (Test-IsSelfHostedFailoverJob $Job) {
        # Self-hosted runners are opportunistic capacity: any non-success outcome on a self-hosted
        # Linux/X64 CPU runner (disconnect, shutdown, offline/abandoned queue, setup/cache/restore
        # error, disk/environment drift, timeout, or step failure) authorizes one clean fallback
        # attempt on standard GitHub-hosted ubuntu-latest. Real product/test failures still fail
        # closed because Assert-FailoverResolvedShard requires ubuntu-latest itself to pass 100%.
        return $conclusion -in @('failure', 'timed_out', 'startup_failure', 'abandoned', 'stale', 'cancelled')
    }
    # Normal GitHub-hosted primary jobs only fail over on explicit server runner-loss annotations.
    if ($conclusion -ne 'failure') { return $false }
    $steps = @(Get-FailoverProperty $Job 'steps' @())
    if (@($steps | Where-Object { $_.conclusion -in @('failure', 'timed_out') }).Count -gt 0) { return $false }
    if (@($messages | Where-Object { $_ -match '(?i)out.of.memory|\bOOM\b|assert|compil|analy[sz]er|timed.out' }).Count -gt 0) { return $false }
    return @($Annotations | Where-Object {
        $_.annotation_level -eq 'failure' -and $_.path -eq '.github' -and $_.start_line -eq 1 -and
        ([string]$_.message -match '^The self-hosted runner lost communication with the server\.' -or
         [string]$_.message -match '^The runner has received a shutdown signal\.' -or
         [string]$_.message -match '^The hosted runner encountered an error running your job\.')
    }).Count -gt 0
}

function Get-FailoverJobPool {
    param([Parameter(Mandatory)] $Job, [AllowEmptyCollection()] [object[]] $Runners = @())
    $labels = @($Job.labels)
    if ($labels -contains 'ubuntu-latest' -or $labels -contains 'ubuntu-24.04' -or $labels -contains 'ubuntu-22.04') {
        if (@($labels | Where-Object { $_ -match '(?i)windows|macos|arm64|self-hosted|aidotnet-cpu' }).Count -gt 0) { throw 'Hosted runner labels contradict Linux/X64 hosted routing.' }
        return [pscustomobject]@{ pool = 'hosted'; os = 'Linux'; arch = 'X64' }
    }
    if (($labels -notcontains 'aidotnet-cpu' -and $labels -notcontains 'self-hosted') -or
        @($labels | Where-Object { $_ -match '(?i)windows|macos|arm64' }).Count -gt 0) {
        throw "Runner for job $($Job.id) is not a qualified Linux/X64 CPU runner."
    }
    $runnerId = [string](Get-FailoverProperty $Job 'runner_id' '')
    $runner = @(if ($runnerId -and $runnerId -ne '0') { $Runners | Where-Object { [string]$_.id -eq $runnerId } })
    if ($runner.Count -gt 1) { throw "Duplicate runner identity for job $($Job.id)." }
    if ($runner.Count -eq 1) {
        $runnerLabels = @($runner[0].labels | ForEach-Object { $_.name })
        if ($runnerLabels -notcontains 'Linux' -or $runnerLabels -notcontains 'X64' -or $runnerLabels -notcontains 'aidotnet-cpu') {
            throw "Runner for job $($Job.id) is not a qualified Linux/X64 CPU runner."
        }
    }
    return [pscustomobject]@{ pool = 'self-hosted'; os = 'Linux'; arch = 'X64' }
}

function Assert-FailoverTrx {
    param([Parameter(Mandatory)] [string] $Directory)
    $files = @(Get-ChildItem -LiteralPath $Directory -Recurse -File -Filter '*.trx')
    # A canonical shard has one complete initial run. Flake retries cannot erase a real failure.
    if ($files.Count -ne 1) { throw "Expected exactly one complete TRX in '$Directory'; found $($files.Count)." }
    $settings = [Xml.XmlReaderSettings]::new()
    $settings.DtdProcessing = [Xml.DtdProcessing]::Prohibit
    $settings.XmlResolver = $null
    $reader = [Xml.XmlReader]::Create($files[0].FullName, $settings)
    try { $xml = [xml]::new(); $xml.Load($reader) } finally { $reader.Dispose() }
    $summary = $xml.SelectSingleNode('//*[local-name()="ResultSummary"]')
    $counters = $xml.SelectSingleNode('//*[local-name()="ResultSummary"]/*[local-name()="Counters"]')
    $times = $xml.SelectSingleNode('/*[local-name()="TestRun"]/*[local-name()="Times"]')
    $results = @($xml.SelectNodes('//*[local-name()="Results"]/*[local-name()="UnitTestResult"]'))
    if ($null -eq $counters -or $null -eq $times -or -not $times.GetAttribute('finish') -or
        $null -eq $summary -or $summary.GetAttribute('outcome') -notin @('Completed', 'Passed')) { throw 'Missing TRX completion evidence.' }
    foreach ($field in @('failed', 'error', 'timeout', 'aborted', 'inconclusive')) {
        if (-not $counters.HasAttribute($field) -or [int]$counters.GetAttribute($field) -ne 0) { throw "TRX has a nonzero/missing '$field' counter." }
    }
    $passed = @($results | Where-Object { $_.GetAttribute('outcome') -eq 'Passed' })
    $ids = @($results | ForEach-Object { $_.GetAttribute('executionId') })
    if ($results.Count -ne [int]$counters.GetAttribute('total') -or $passed.Count -eq 0 -or
        $passed.Count -ne [int]$counters.GetAttribute('passed') -or
        $passed.Count -ne [int]$counters.GetAttribute('executed') -or
        @($results | Where-Object { $_.GetAttribute('outcome') -notin @('Passed', 'NotExecuted') }).Count -gt 0 -or
        @($ids | Select-Object -Unique).Count -ne $ids.Count -or @($ids | Where-Object { -not $_ }).Count -gt 0) {
        throw 'TRX is partial, duplicated, failed, or its counters disagree with its results.'
    }
    return [pscustomobject]@{ total = $results.Count; passed = $passed.Count; skipped = $results.Count - $passed.Count }
}

function Assert-FailoverReceipt {
    param($Receipt, $Shard, [string] $TestedSha, [string] $RunId, [string] $RunAttempt,
        [string] $ArtifactId, [string] $ArtifactDigest, [string] $Attempt, [string] $CoveragePolicyHash = '')
    $expected = @{
        schemaVersion = '1'; shard = [string]$Shard.name; definitionHash = Get-FailoverDefinitionHash $Shard
        testedSha = $TestedSha; runId = $RunId; runAttempt = $RunAttempt
        sourceArtifactId = $ArtifactId; sourceArtifactDigest = $ArtifactDigest
        attempt = $Attempt; runnerOs = 'Linux'; runnerArch = 'X64'; testOutcome = 'success'
        coveragePolicyHash = $CoveragePolicyHash
    }
    foreach ($key in $expected.Keys) {
        if ([string](Get-FailoverProperty $Receipt $key '') -cne [string]$expected[$key]) { throw "Receipt '$key' mismatch for '$($Shard.name)'." }
    }
}

function Invoke-FailoverApi {
    param([string] $Path, [switch] $Paginate, [string] $Token = '')
    $token = if ($Token) { $Token } elseif ($env:GH_TOKEN) { $env:GH_TOKEN } else { $env:GITHUB_TOKEN }
    if (-not $token) { throw 'A GitHub API token is required.' }
    $headers = @{ Authorization = "Bearer $token"; Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28' }
    if (-not (Get-Command Invoke-GitHubApiWithRetry -ErrorAction SilentlyContinue)) {
        . "$PSScriptRoot/Invoke-GitHubApiWithRetry.ps1"
    }
    $items = @(); $page = 1
    do {
        $separator = if ($Path.Contains('?')) { '&' } else { '?' }
        $uri = "https://api.github.com/$Path${separator}per_page=50&page=$page"
        $response = Invoke-GitHubApiWithRetry -Headers $headers -Uri $uri
        if (-not $Paginate) { return $response }
        $property = if ($null -ne $response -and $response -isnot [System.Collections.IList]) {
            @('jobs', 'artifacts', 'runners') | Where-Object { $response.PSObject.Properties[$_] } | Select-Object -First 1
        } else { $null }
        $batch = @(if ($property) { $response.$property } elseif ($null -ne $response) { $response })
        $items += $batch; $page++
    } while ($batch.Count -eq 50)
    return $items
}

function Assert-FailoverSourceArtifact {
    param($Artifact, [string] $ArtifactId, [string] $ArtifactDigest, [string] $TestedSha, [string] $RunId)
    if ($ArtifactId -notmatch '^[1-9][0-9]*$' -or $ArtifactDigest -notmatch '^sha256:[a-f0-9]{64}$' -or
        [string]$Artifact.id -cne $ArtifactId -or [string]$Artifact.digest -cne $ArtifactDigest -or
        [string]$Artifact.name -cne "build-output-$TestedSha" -or $Artifact.expired -ne $false -or
        [string]$Artifact.workflow_run.id -cne $RunId) { throw 'Source build artifact identity/digest/run does not match.' }
}

function Get-FailoverLogicalShards {
    param($Shard)
    $segments = @(Get-FailoverProperty $Shard 'segments' @())
    if ($segments.Count -gt 0) { return $segments }
    return $Shard
}

function Get-FailoverAnnotations {
    param($Job, [string] $Repository, [string] $HeadSha)
    $prefix = "https://api.github.com/repos/$Repository/check-runs/"
    $url = [string](Get-FailoverProperty $Job 'check_run_url' '')
    if (-not $url.StartsWith($prefix, [StringComparison]::Ordinal) -or $url.Substring($prefix.Length) -notmatch '^[1-9][0-9]*$') {
        throw 'Job does not identify a check run in the current repository.'
    }
    $path = $url.Substring('https://api.github.com/'.Length)
    $check = Invoke-FailoverApi $path
    if ([string]$check.head_sha -cne $HeadSha -or [string]$check.name -cne [string]$Job.name -or
        [string]$check.details_url -cne [string]$Job.html_url) { throw 'Check run does not bind to the expected job/head.' }
    return @(Invoke-FailoverApi "$path/annotations" -Paginate)
}

function Test-FailoverSuccessfulJob {
    param($Job)
    if ($null -eq $Job -or $Job.status -ne 'completed' -or $Job.conclusion -ne 'success') { return $false }
    $steps = @(Get-FailoverProperty $Job 'steps' @())
    return @($steps | Where-Object { $_.conclusion -in @('failure', 'cancelled', 'timed_out') }).Count -eq 0 -and
        @($steps | Where-Object { $_.name -eq 'Run tests (sharded) with coverage' -and $_.conclusion -eq 'success' }).Count -eq 1
}

function Assert-FailoverResolvedShard {
    param([Parameter(Mandatory)] $Shard, [Parameter(Mandatory)] $PrimaryJob,
        [AllowEmptyCollection()] [object[]] $PrimaryAnnotations = @(), $AlternateJob = $null,
        [Parameter(Mandatory)] $Receipt, [Parameter(Mandatory)] $AttemptReceipt,
        [Parameter(Mandatory)] $CanonicalArtifact, [Parameter(Mandatory)] $AttemptArtifact,
        [Parameter(Mandatory)] $CoverageArtifact,
        [Parameter(Mandatory)] [string] $CanonicalDirectory,
        [Parameter(Mandatory)] [string] $AttemptDirectory,
        [Parameter(Mandatory)] [string] $CoverageDirectory,
        [Parameter(Mandatory)] [string] $ExpectedAttemptArtifactName,
        [Parameter(Mandatory)] [string] $TestedSha, [Parameter(Mandatory)] [string] $RunId,
        [Parameter(Mandatory)] [string] $RunAttempt, [Parameter(Mandatory)] [string] $SourceArtifactId,
        [Parameter(Mandatory)] [string] $SourceArtifactDigest,
        [string] $CoveragePolicyHash = '', $CoveragePolicy = $null)
    $attempt = 'primary'; $job = $PrimaryJob
    if (-not (Test-FailoverSuccessfulJob $PrimaryJob)) {
        if (-not (Test-FailoverInfrastructureFailure $PrimaryJob $PrimaryAnnotations)) { throw 'Primary failure is not explicitly recoverable infrastructure.' }
        if (-not (Test-FailoverSuccessfulJob $AlternateJob)) { throw 'The single alternate attempt did not independently succeed.' }
        $attempt = 'alternate'; $job = $AlternateJob
    } elseif ($null -ne $AlternateJob) { throw 'A successful primary must never be overwritten by an alternate.' }
    Assert-FailoverReceipt $Receipt $Shard $TestedSha $RunId $RunAttempt $SourceArtifactId $SourceArtifactDigest $attempt $CoveragePolicyHash
    Assert-FailoverReceipt $AttemptReceipt $Shard $TestedSha $RunId $RunAttempt $SourceArtifactId $SourceArtifactDigest $attempt $CoveragePolicyHash
    $receiptJson = ConvertTo-Json -InputObject (ConvertTo-FailoverCanonicalValue $Receipt) -Depth 30 -Compress
    $attemptJson = ConvertTo-Json -InputObject (ConvertTo-FailoverCanonicalValue $AttemptReceipt) -Depth 30 -Compress
    if ($receiptJson -cne $attemptJson -or [string]$Receipt.runnerName -cne [string]$job.runner_name -or
        ($attempt -eq 'alternate' -and [string]$Receipt.originalJobId -cne [string]$PrimaryJob.id) -or
        ($attempt -eq 'primary' -and [string]$Receipt.originalJobId -ne '')) { throw 'Canonical/immutable attempt/job provenance disagrees.' }
    foreach ($artifact in @($CanonicalArtifact, $AttemptArtifact, $CoverageArtifact)) {
        if ([string]$artifact.id -notmatch '^[1-9][0-9]*$' -or [string]$artifact.digest -notmatch '^sha256:[a-f0-9]{64}$' -or
            $artifact.expired -ne $false -or [string]$artifact.workflow_run.id -cne $RunId) { throw 'Invalid canonical or attempt artifact identity/digest/run.' }
    }
    if (@(@($CanonicalArtifact.id, $AttemptArtifact.id, $CoverageArtifact.id) | Select-Object -Unique).Count -ne 3) { throw 'Canonical, coverage and immutable attempt artifacts must have distinct identities.' }
    if ([string]$PrimaryJob.name -notmatch '^Tests \([^)]+\) - (.+)$') { throw 'Primary job does not identify its physical shard.' }
    $expectedAttemptName = "shard-attempt-$attempt-$TestedSha-$(Get-FailoverSlug $Matches[1])-$RunAttempt"
    if ($ExpectedAttemptArtifactName -cne $expectedAttemptName -or
        [string]$CanonicalArtifact.name -cne "test-results-$TestedSha-$(Get-FailoverSlug $Shard.name)" -or
        [string]$CoverageArtifact.name -cne "coverage-$TestedSha-$(Get-FailoverSlug $Shard.name)" -or
        [string]$AttemptArtifact.name -cne $ExpectedAttemptArtifactName -or
        [string]$CanonicalArtifact.id -ceq [string]$AttemptArtifact.id) { throw 'Canonical and immutable attempt artifact names/IDs disagree.' }
    $counts = Assert-FailoverTrx $CanonicalDirectory
    $null = Assert-FailoverTrx $AttemptDirectory
    $null = Assert-FailoverTrx $CoverageDirectory
    $canonicalTrx = @(Get-ChildItem -LiteralPath $CanonicalDirectory -Filter '*.trx' -File -Recurse)[0]
    $attemptTrx = @(Get-ChildItem -LiteralPath $AttemptDirectory -Filter '*.trx' -File -Recurse)[0]
    if ((Get-FileHash $canonicalTrx.FullName -Algorithm SHA256).Hash -cne (Get-FileHash $attemptTrx.FullName -Algorithm SHA256).Hash) {
        throw 'Canonical TRX differs from the immutable original attempt.'
    }
    if ((Get-FailoverProperty $Receipt 'collectCoverage' $null) -isnot [bool]) { throw 'Actual coverage decision is missing or malformed.' }
    if ($null -eq $CoveragePolicy) { $CoveragePolicy = Get-FailoverCoveragePolicy }
    if ((Get-FailoverCoveragePolicyHash $CoveragePolicy) -cne $CoveragePolicyHash -or
        $Receipt.collectCoverage -ne (Get-FailoverExpectedCoverage $Shard $CoveragePolicy)) { throw 'Attempt omitted or changed the independently required coverage policy.' }
    foreach ($directory in @($CanonicalDirectory, $AttemptDirectory, $CoverageDirectory)) {
        $invocations = @(Get-ChildItem -LiteralPath $directory -Recurse -File -Filter coverage-invocation.json)
        if ($invocations.Count -ne 1) { throw 'Actual invocation coverage evidence is missing or duplicated.' }
        $invocation = Get-Content $invocations[0].FullName -Raw | ConvertFrom-Json
        if ((Get-FailoverProperty $invocation 'collectCoverage' $null) -isnot [bool] -or
            $invocation.collectCoverage -ne $Receipt.collectCoverage) { throw 'Receipt coverage differs from the actual test invocation.' }
    }
    $coverageTrx = @(Get-ChildItem $CoverageDirectory -Recurse -File -Filter '*.trx')[0]
    if ((Get-FileHash $coverageTrx.FullName).Hash -cne (Get-FileHash $canonicalTrx.FullName).Hash) { throw 'Coverage and diagnostic TRX disagree.' }
    $coverageReceipts = @(Get-ChildItem $CoverageDirectory -Recurse -File -Filter shard-attempt.json)
    if ($coverageReceipts.Count -ne 1 -or
        (ConvertTo-Json -InputObject (ConvertTo-FailoverCanonicalValue (Get-Content $coverageReceipts[0].FullName -Raw | ConvertFrom-Json)) -Depth 30 -Compress) -cne $receiptJson) { throw 'Coverage and diagnostic receipts disagree.' }
    if ($Receipt.collectCoverage -and @(Get-ChildItem $CoverageDirectory -Recurse -File -Filter coverage.opencover.xml).Count -eq 0) { throw 'Required coverage output is missing.' }
    return [pscustomobject]@{ shard = [string]$Shard.name; definitionHash = Get-FailoverDefinitionHash $Shard; attempt = $attempt
        primaryJobId = [string]$PrimaryJob.id; chosenJobId = [string]$job.id
        canonicalArtifactId = [string]$CanonicalArtifact.id; canonicalArtifactDigest = [string]$CanonicalArtifact.digest
        attemptArtifactId = [string]$AttemptArtifact.id; attemptArtifactDigest = [string]$AttemptArtifact.digest
        coverageArtifactId = [string]$CoverageArtifact.id; coverageArtifactDigest = [string]$CoverageArtifact.digest
        counts = $counts }
}

function Get-FailoverCoveragePolicyHash {
    param($Policy = $null)
    if ($null -eq $Policy) { $Policy = Get-FailoverCoveragePolicy }
    $json = ConvertTo-Json -InputObject (ConvertTo-FailoverCanonicalValue $Policy) -Depth 50 -Compress
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($json))).ToLowerInvariant()
}

function Get-FailoverCoveragePolicy {
    $policy = [ordered]@{ eventName = [string]$env:GITHUB_EVENT_NAME; forceCoverage = [string]$env:SHARD_FORCE_COVERAGE }
    foreach ($name in @('COVERAGE_CARRIED', 'COVERAGE_RUN_WITHOUT_INSTRUMENTATION', 'COVERAGE_ROLLING')) {
        $value = [Environment]::GetEnvironmentVariable($name)
        $policy[$name] = if ($value) { @($value | ConvertFrom-Json) } else { @() }
    }
    return [pscustomobject]$policy
}

function Get-FailoverExpectedCoverage {
    param($Shard, $Policy)
    $decision = Get-ShardCoveragePolicy -ShardName ([string]$Shard.name) -Heavy ([bool](Get-FailoverProperty $Shard 'heavy' $false)) `
        -EventName ([string]$Policy.eventName) -ForceCoverage ([string]$Policy.forceCoverage) `
        -CarriedJson (ConvertTo-Json -InputObject @($Policy.COVERAGE_CARRIED) -Compress) `
        -RunWithoutCoverageJson (ConvertTo-Json -InputObject @($Policy.COVERAGE_RUN_WITHOUT_INSTRUMENTATION) -Compress) `
        -RollingJson (ConvertTo-Json -InputObject @($Policy.COVERAGE_ROLLING) -Compress) -Quiet
    return [bool]$decision.CollectCoverage
}

function New-FailoverAlternateShard {
    param($Shard, $Job, [AllowEmptyCollection()] [object[]] $Runners = @(), [bool] $TrustedSameRepository, [bool] $AlreadyRetried = $false)
    if ($AlreadyRetried -or (Get-FailoverProperty $Shard 'failoverAttempt' '') -or
        (Get-FailoverProperty $Shard 'failoverOriginalJobId' '')) { throw 'Only one alternate attempt is permitted.' }
    $pool = Get-FailoverJobPool $Job $Runners
    $runner = 'ubuntu-latest'
    if ($pool.pool -eq 'hosted') {
        if (-not $TrustedSameRepository) { throw 'Untrusted forks cannot fail over to self-hosted machines.' }
        if ((Get-FailoverProperty $Shard 'heavy' $false) -or (Get-FailoverProperty $Shard 'timeout' 0) -or
            @(Get-FailoverProperty $Shard 'segments' @()).Count -gt 0) { throw 'Shard capabilities are not certified for the alternate CPU pool.' }
        $qualified = @($Runners | Where-Object {
            $labels = @($_.labels | ForEach-Object { $_.name })
            $_.status -eq 'online' -and -not $_.busy -and $labels -contains 'aidotnet-cpu' -and
            $labels -contains 'Linux' -and $labels -contains 'X64'
        })
        if ($qualified.Count -eq 0) { throw 'No idle, online Linux/X64 CPU alternate is available.' }
        $runner = @('self-hosted', 'Linux', 'X64', 'aidotnet-cpu')
    }
    $copy = $Shard.PSObject.Copy()
    $copy | Add-Member runner $runner -Force
    $copy | Add-Member failoverOriginalJobId ([string]$Job.id) -Force
    $copy | Add-Member failoverAttempt 'alternate' -Force
    return $copy
}

function Assert-FailoverLogicalInventory {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $RequiredShards,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $ResolvedEvidence)
    $required = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($shard in $RequiredShards) {
        if (-not $required.Add([string]$shard.name)) { throw 'Duplicate required logical shard.' }
    }
    if ($required.Count -eq 0) { throw 'Empty required logical shard inventory.' }
    $resolved = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($evidence in $ResolvedEvidence) {
        if (-not $required.Contains([string]$evidence.shard) -or -not $resolved.Add([string]$evidence.shard)) { throw 'Unexpected or duplicate resolved logical shard.' }
    }
    if (-not $required.SetEquals($resolved)) { throw 'Required logical shard receipt is missing.' }
}

function Assert-FailoverPublishedProof {
    param($Proof, $Certificate, [AllowEmptyCollection()] [object[]] $Artifacts = @())
    if ($Certificate.schemaVersion -ne 4 -or $Proof.schemaVersion -ne 1 -or $Proof.complete -isnot [bool] -or -not $Proof.complete -or
        @($Proof.errors).Count -ne 0 -or [string]$Proof.testedSha -cne [string]$Certificate.testedSha -or
        [string]$Proof.runId -cne [string]$Certificate.runId -or
        [string]$Proof.runAttempt -cne [string]$Certificate.runnerFailoverProof.runAttempt -or
        [string]$Proof.coveragePolicyHash -notmatch '^[a-f0-9]{64}$') { throw 'Published failover proof is incomplete or mismatched.' }
    if ($Certificate.requiresValidation -isnot [bool] -or $Certificate.requiresTests -isnot [bool] -or
        $Proof.requiresValidation -isnot [bool] -or $Proof.requiresTests -isnot [bool] -or
        $Certificate.requiresValidation -ne $Proof.requiresValidation -or $Certificate.requiresTests -ne $Proof.requiresTests -or
        ($Certificate.requiresTests -and -not $Certificate.requiresValidation)) { throw 'Proof validation/test requirements must exactly match explicit JSON booleans.' }
    if (-not $Certificate.requiresTests) {
        if ($Proof.requiresTests -or @($Proof.requiredShards).Count -ne 0 -or
            @($Proof.shards).Count -ne 0 -or [string]$Proof.sourceArtifactId -ne '' -or [string]$Proof.sourceArtifactDigest -ne '') { throw 'Non-runtime proof cannot claim omitted runtime requirements or artifacts.' }
        return
    }
    if (-not $Proof.requiresValidation -or -not $Proof.requiresTests) { throw 'Runtime certificate lacks required test proof.' }
    Assert-FailoverLogicalInventory @($Proof.requiredShards) @($Proof.shards)
    $source = @($Artifacts | Where-Object { [string]$_.id -ceq [string]$Proof.sourceArtifactId })
    if ($source.Count -ne 1) { throw 'Proof source artifact is missing or duplicated.' }
    Assert-FailoverSourceArtifact $source[0] ([string]$Proof.sourceArtifactId) ([string]$Proof.sourceArtifactDigest) ([string]$Proof.testedSha) ([string]$Proof.runId)
    $usedIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($shard in $Proof.shards) {
        $required = @($Proof.requiredShards | Where-Object { $_.name -ceq $shard.shard })
        if ($required.Count -ne 1 -or [string]$shard.definitionHash -cne (Get-FailoverDefinitionHash $required[0])) { throw 'Published definition hash is invalid.' }
        foreach ($kind in @('canonical', 'coverage', 'attempt')) {
            $id = [string]$shard."${kind}ArtifactId"; $digest = [string]$shard."${kind}ArtifactDigest"
            $artifact = @($Artifacts | Where-Object { [string]$_.id -ceq $id })
            if ($artifact.Count -ne 1 -or $artifact[0].expired -ne $false -or [string]$artifact[0].digest -cne $digest -or
                $digest -notmatch '^sha256:[a-f0-9]{64}$' -or [string]$artifact[0].workflow_run.id -cne [string]$Proof.runId) { throw 'Published canonical/attempt artifact was removed, replaced or tampered with.' }
            if ($kind -ne 'attempt' -and -not $usedIds.Add($id)) { throw 'Canonical artifact identity was reused across logical shards.' }
            if ($kind -ne 'attempt') {
                $prefix = if ($kind -eq 'canonical') { 'test-results' } else { 'coverage' }
                if ([string]$artifact[0].name -cne "$prefix-$($Proof.testedSha)-$(Get-FailoverSlug $shard.shard)") { throw 'Published canonical artifact name mismatch.' }
            }
        }
    }
}

function Get-FailoverDiagnosticContext {
    param([string] $VerifierResult, [AllowNull()] $Proof,
        [Parameter(Mandatory)] [string] $TestedSha, [Parameter(Mandatory)] [string] $RunId,
        [Parameter(Mandatory)] [string] $RunAttempt,
        [AllowEmptyCollection()] [object[]] $OriginalJobs = @(),
        [AllowEmptyCollection()] [object[]] $ResolvedJobs = @(), [switch] $RequireResolvedJobs)
    $reasons = [Collections.Generic.List[string]]::new()
    if ($VerifierResult -cne 'success') { $reasons.Add("Logical shard verifier concluded '$VerifierResult'.") }
    if ($null -eq $Proof) { $reasons.Add('Logical shard proof is missing or unreadable.') }
    else {
        if ((Get-FailoverProperty $Proof 'schemaVersion' 0) -ne 1 -or
            (Get-FailoverProperty $Proof 'complete' $null) -isnot [bool] -or
            (Get-FailoverProperty $Proof 'complete' $false) -ne $true) { $reasons.Add('Logical shard proof is incomplete.') }
        if ([string](Get-FailoverProperty $Proof 'testedSha' '') -cne $TestedSha -or
            [string](Get-FailoverProperty $Proof 'runId' '') -cne $RunId -or
            [string](Get-FailoverProperty $Proof 'runAttempt' '') -cne $RunAttempt) { $reasons.Add('Logical shard proof does not match this run, attempt and tested SHA.') }
        if (@(Get-FailoverProperty $Proof 'errors' @()).Count -ne 0) { $reasons.Add('Logical shard proof reports rejected evidence.') }
        $requiresTests = Get-FailoverProperty $Proof 'requiresTests' $null
        if ($requiresTests -isnot [bool]) { $reasons.Add('Logical shard proof omits its explicit test requirement.') }
        elseif ($requiresTests) {
            try { Assert-FailoverLogicalInventory @(Get-FailoverProperty $Proof 'requiredShards' @()) @(Get-FailoverProperty $Proof 'shards' @()) }
            catch { $reasons.Add($_.Exception.Message) }
        }
        if ($RequireResolvedJobs -and $requiresTests -eq $true) {
            $expected = @(Get-FailoverProperty $Proof 'requiredShards' @() | ForEach-Object { "Tests ($($_.framework)) - $($_.name)" })
            $actual = @($ResolvedJobs | ForEach-Object { [string]$_.name })
            if ($expected.Count -ne $actual.Count -or @($actual | Select-Object -Unique).Count -ne $actual.Count -or
                @($expected | Where-Object { $_ -cnotin $actual }).Count -ne 0) { $reasons.Add('Resolved job inventory is missing or does not match the proof.') }
        }
    }
    $verified = $reasons.Count -eq 0
    $jobs = @($OriginalJobs)
    if ($verified -and $RequireResolvedJobs) {
        $jobs = @($OriginalJobs | Where-Object { $_.name -notlike 'Tests (*) - *' -and $_.name -notlike 'Alternate attempt (*) - *' }) + @($ResolvedJobs)
    }
    return [pscustomobject]@{
        schemaVersion = 1; verified = $verified; diagnosticOnly = -not $verified
        testedSha = $TestedSha; runId = $RunId; runAttempt = $RunAttempt; verifierResult = $VerifierResult
        reasons = @($reasons); proofErrors = @(Get-FailoverProperty $Proof 'errors' @()); jobs = $jobs
        regressionArtifactPrefix = $(if ($verified) { 'test-outcome-ledger' } else { 'test-outcome-diagnostics' })
        aggregateArtifactPrefix = $(if ($verified) { 'ci-test-analysis' } else { 'ci-test-diagnostics' })
    }
}
