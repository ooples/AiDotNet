[CmdletBinding(DefaultParameterSetName = 'Json')]
param([Parameter(Mandatory, ParameterSetName = 'Json')] [string] $MatrixJson,
    [Parameter(Mandatory, ParameterSetName = 'Path')] [string] $MatrixJsonPath, [Parameter(Mandatory)] [string] $AttemptRoot,
    [Parameter(Mandatory)] [string] $CanonicalRoot, [Parameter(Mandatory)] [ValidateSet('coverage', 'test-results')] [string] $ArtifactPrefix,
    [Parameter(Mandatory)] [string] $TestedSha, [Parameter(Mandatory)] [string] $RunId,
    [Parameter(Mandatory)] [string] $RunAttempt, [Parameter(Mandatory)] [AllowEmptyString()] [string] $SourceArtifactId,
    [Parameter(Mandatory)] [AllowEmptyString()] [string] $SourceArtifactDigest, [Parameter(Mandatory)] [string] $OutputPath)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/RunnerFailover.ps1"
# CI passes the shard matrix as a file: it can outgrow Linux's 128 KiB limit on one environment string.
if ($PSCmdlet.ParameterSetName -ceq 'Path') { $MatrixJson = Get-Content -LiteralPath $MatrixJsonPath -Raw }
if ([string]::IsNullOrWhiteSpace($MatrixJson)) { throw 'The shard matrix is empty.' }
$matrix = @($MatrixJson | ConvertFrom-Json)
$records = [Collections.Generic.List[object]]::new()
New-Item -ItemType Directory -Path $CanonicalRoot -Force | Out-Null
foreach ($physical in $matrix) {
    $logical = @(Get-FailoverLogicalShards $physical)
    $container = Join-Path $AttemptRoot "shard-attempt-primary-$TestedSha-$(Get-FailoverSlug $physical.name)-$RunAttempt"
    $packed = @(Get-FailoverProperty $physical 'segments' @()).Count -gt 0
    for ($index = 0; $index -lt $logical.Count; $index++) {
        $shard = $logical[$index]; $slug = Get-FailoverSlug $shard.name
        $destination = Join-Path $CanonicalRoot "$ArtifactPrefix-$TestedSha-$slug"
        if (Test-Path -LiteralPath $destination -PathType Container) {
            $records.Add([pscustomobject]@{ shard = $shard.name; status = 'canonical'; excludedTrx = 0; reason = 'Existing canonical result retained without replacement.' })
            continue
        }
        # Match Invoke-Shard.ps1 and Write-ShardAttemptReceipt.ps1 exactly. A nested retry
        # directory with the same basename is never an initial result or a replacement receipt.
        $directory = if ($packed) { Join-Path $container "slot-$index" } else { $container }
        $initial = Join-Path $directory "$slug/test-results.trx"
        $status = 'rejected'; $reason = ''; $excluded = 0
        try {
            $receipt = Get-Content (Join-Path $directory 'shard-attempt.json') -Raw -ErrorAction Stop | ConvertFrom-Json
            $expected = @{
                schemaVersion = '1'; shard = [string]$shard.name; definitionHash = Get-FailoverDefinitionHash $shard
                testedSha = $TestedSha; runId = $RunId; runAttempt = $RunAttempt; attempt = 'primary'
                sourceArtifactId = $SourceArtifactId; sourceArtifactDigest = $SourceArtifactDigest
                coveragePolicyHash = Get-FailoverCoveragePolicyHash
            }
            foreach ($key in $expected.Keys) {
                if ([string](Get-FailoverProperty $receipt $key '') -cne [string]$expected[$key]) { throw "Primary diagnostic receipt '$key' mismatch." }
            }
            if ([string](Get-FailoverProperty $receipt 'testOutcome' '') -cnotin @('success', 'failure')) { throw 'Primary diagnostic receipt has no terminal test outcome.' }
            $excluded = @(Get-ChildItem -LiteralPath $directory -Recurse -Filter '*.trx' -File |
                Where-Object { $_.FullName -cne [IO.Path]::GetFullPath($initial) }).Count
            if (-not (Test-Path -LiteralPath $initial -PathType Leaf)) { throw 'Exact initial test-results.trx is missing; nested retries are not substitutes.' }
            [xml]$trx = Get-Content -LiteralPath $initial -Raw
            if ($null -eq $trx.SelectSingleNode('//*[local-name()="Counters"]')) { throw 'Initial TRX has no counters.' }
            New-Item -ItemType Directory -Path $destination -Force | Out-Null
            Copy-Item -LiteralPath $initial -Destination (Join-Path $destination 'test-results.trx')
            [ordered]@{ schemaVersion = 1; shard = $shard.name; slug = $slug; diagnosticOnly = $true
                testStepOutcome = $receipt.testOutcome; source = 'immutable-primary-initial-attempt' } |
                ConvertTo-Json | Set-Content (Join-Path $destination 'shard-metadata.json') -Encoding utf8
            $status = 'imported'; $reason = 'Initial primary TRX observed without certified canonical evidence; retries excluded.'
        }
        catch { $reason = $_.Exception.Message }
        $records.Add([pscustomobject]@{ shard = $shard.name; status = $status; excludedTrx = $excluded; reason = $reason })
    }
}
$report = [ordered]@{ schemaVersion = 1; diagnosticOnly = $true; testedSha = $TestedSha; runId = $RunId; runAttempt = $RunAttempt
    imported = @($records | Where-Object status -eq 'imported').Count
    rejected = @($records | Where-Object status -eq 'rejected').Count
    canonical = @($records | Where-Object status -eq 'canonical').Count; shards = @($records) }
New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null
$report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Host "Primary diagnostic intake: $($report.imported) initial TRX imported, $($report.rejected) missing/rejected, $($report.canonical) canonical retained."
