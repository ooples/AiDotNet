[CmdletBinding()]
param(
    [string] $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string] $DescriptorPath,
    # Fixture-only entry point: validates already downloaded evidence without credentials or network.
    [string] $EvidenceDirectory,
    [switch] $ValidateOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Assert-Feed([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}
function Get-Sha256([string] $Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Get-CaseDigest([string[]] $Cases) {
    $ordered = [string[]]@($Cases)
    [Array]::Sort($ordered, [StringComparer]::Ordinal)
    $bytes = [Text.Encoding]::UTF8.GetBytes([string]::Join("`n", $ordered))
    [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
}
[xml] $props = Get-Content -LiteralPath (Join-Path $RepositoryRoot 'Directory.Packages.props') -Raw
$versions = @($props.SelectNodes('//PackageVersion[@Include="AiDotNet.Tensors"]'))
Assert-Feed ($versions.Count -eq 1) 'Expected exactly one central AiDotNet.Tensors version.'
$version = $versions[0].Version
if ($version -notmatch '-coordination\.') {
    if ($env:GITHUB_OUTPUT) { Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value 'active=false' }
    return
}
if (-not $DescriptorPath) { $DescriptorPath = Join-Path $RepositoryRoot '.github/tensors-validation.json' }
$pin = Get-Content -LiteralPath $DescriptorPath -Raw | ConvertFrom-Json
Assert-Feed ($version -ceq $pin.packageVersion) 'Requested coordination package does not match the pinned descriptor.'
if (-not $ValidateOnly) {
    Assert-Feed ($env:GITHUB_REPOSITORY -ceq 'ooples/AiDotNet') 'Validation artifacts require the trusted consumer repository.'
    $trusted = $env:GITHUB_EVENT_NAME -in @('push', 'workflow_dispatch', 'schedule')
    if ($env:GITHUB_EVENT_NAME -eq 'pull_request') {
        $event = Get-Content -LiteralPath $env:GITHUB_EVENT_PATH -Raw | ConvertFrom-Json
        $trusted = $event.pull_request.head.repo.full_name -ceq $env:GITHUB_REPOSITORY
    }
    # Temporary candidates intentionally fail closed on merge_group and all other events.
    # The final merge uses a stable release; fork-derived queue code never receives the PAT.
    Assert-Feed $trusted 'Coordination package restore is unavailable to untrusted events and forks (including merge_group).'
    Assert-Feed (-not [string]::IsNullOrWhiteSpace($env:TENSORS_ARTIFACT_TOKEN)) 'A producer artifact read token is required.'
    . (Join-Path $RepositoryRoot 'tools/TestImpact/Invoke-GitHubApiWithRetry.ps1')
    $headers = @{ Authorization = "Bearer $env:TENSORS_ARTIFACT_TOKEN"; Accept = 'application/vnd.github+json'; 'X-GitHub-Api-Version' = '2022-11-28' }
    $api = "https://api.github.com/repos/$($pin.repository)/actions"
    $run = Invoke-GitHubApiWithRetry -Uri "$api/runs/$($pin.runId)" -Headers $headers
    Assert-Feed ($run.id -eq $pin.runId -and $run.head_sha -ceq $pin.sourceCommit -and $run.path -ceq $pin.workflowPath -and $run.run_attempt -eq $pin.runAttempt -and $run.status -ceq 'completed' -and $run.conclusion -ceq 'success') 'Producer workflow identity or successful completion did not match.'
    $artifact = Invoke-GitHubApiWithRetry -Uri "$api/artifacts/$($pin.artifactId)" -Headers $headers
    Assert-Feed ($artifact.id -eq $pin.artifactId -and $artifact.name -ceq $pin.artifactName -and $artifact.digest -ceq $pin.artifactDigest -and -not $artifact.expired -and $artifact.workflow_run.id -eq $pin.runId -and $artifact.workflow_run.head_sha -ceq $pin.sourceCommit) 'Producer artifact identity, digest, or retention did not match.'
    $EvidenceDirectory = Join-Path $env:RUNNER_TEMP "tensors-validation-$($pin.artifactId)-$([guid]::NewGuid().ToString('N'))"
    $originalToken = $env:GITHUB_TOKEN
    $originalGhToken = $env:GH_TOKEN
    try {
        $env:GITHUB_TOKEN = $env:TENSORS_ARTIFACT_TOKEN
        $env:GH_TOKEN = $env:TENSORS_ARTIFACT_TOKEN
        & (Join-Path $RepositoryRoot 'tools/TestImpact/Receive-RequiredArtifact.ps1') -Repository $pin.repository -ArtifactId $pin.artifactId -ExpectedDigest $pin.artifactDigest -Destination $EvidenceDirectory -RunId $pin.runId -JobIndex 0
        Assert-Feed ($LASTEXITCODE -eq 0) "Artifact download failed with exit code $LASTEXITCODE."
    } finally {
        $env:GITHUB_TOKEN = $originalToken
        $env:GH_TOKEN = $originalGhToken
    }
}
Assert-Feed (-not [string]::IsNullOrWhiteSpace($EvidenceDirectory)) 'An evidence directory is required.'
$manifest = Get-Content -LiteralPath (Join-Path $EvidenceDirectory 'manifest.json') -Raw | ConvertFrom-Json
Assert-Feed ($manifest.repository -ceq $pin.repository -and $manifest.sourceCommit -ceq $pin.sourceCommit -and $manifest.packageVersion -ceq $version -and $manifest.runAttempt -eq $pin.runAttempt -and $manifest.workflowRun -ceq "https://github.com/$($pin.repository)/actions/runs/$($pin.runId)") 'Manifest identity did not match.'
Assert-Feed ((@($manifest.targetFrameworks | Sort-Object) -join ',') -ceq (@($pin.targetFrameworks | Sort-Object) -join ',')) 'Manifest target frameworks did not match.'
$counters = $manifest.testCounters
Assert-Feed ([int]$counters.total -eq $pin.testCount -and [int]$counters.executed -eq $pin.testCount -and [int]$counters.passed -eq $pin.testCount) 'Manifest test count did not match.'
foreach ($counter in $counters.PSObject.Properties) {
    if ($counter.Name -notin @('total', 'executed', 'passed')) { Assert-Feed ([int]$counter.Value -eq 0) "Nonzero test outcome: $($counter.Name)" }
}
Assert-Feed ($null -ne $manifest.PSObject.Properties['retainedOutputTestCounters']) 'Missing retained-output test group.'
$retainedCounters = $manifest.retainedOutputTestCounters
Assert-Feed ([int]$retainedCounters.total -eq $pin.retainedOutputTestCount -and [int]$retainedCounters.executed -eq $pin.retainedOutputTestCount -and [int]$retainedCounters.passed -eq $pin.retainedOutputTestCount) 'Retained-output test count did not match.'
foreach ($outcome in @('failed', 'error', 'timeout', 'aborted', 'inconclusive', 'passedButRunAborted', 'notRunnable', 'notExecuted', 'disconnected', 'warning', 'completed', 'inProgress', 'pending')) {
    Assert-Feed ($null -ne $retainedCounters.PSObject.Properties[$outcome]) "Missing retained-output test outcome: $outcome"
}
foreach ($counter in $retainedCounters.PSObject.Properties) {
    if ($counter.Name -notin @('total', 'executed', 'passed')) { Assert-Feed ([int]$counter.Value -eq 0) "Nonzero retained-output test outcome: $($counter.Name)" }
}
Assert-Feed ($null -ne $manifest.PSObject.Properties['retainedOutputTestCases']) 'Missing retained-output test identities.'
$retainedCases = @($manifest.retainedOutputTestCases)
Assert-Feed ($retainedCases.Count -eq $pin.retainedOutputTestCount) 'Retained-output test identity count did not match.'
$uniqueCases = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($case in $retainedCases) {
    Assert-Feed (-not [string]::IsNullOrWhiteSpace($case) -and $uniqueCases.Add($case)) 'Retained-output test identities must be nonempty and unique.'
}
$newCpuGroups = @(
    @{ Prefix = 'heterogeneousReplay'; Class = 'AiDotNet.Tensors.Tests.Engines.Compilation.ForwardOnlyHeterogeneousTrainingTests' }
    @{ Prefix = 'noGrad'; Class = 'AiDotNet.Tensors.Tests.Engines.Compilation.NoGradTrainingTraceTests' }
    @{ Prefix = 'optimizerEnrollment'; Class = 'AiDotNet.Tensors.Tests.Engines.Compilation.CompiledOptimizerEnrollmentTests' }
    @{ Prefix = 'enrollmentCheckpoint'; Class = 'AiDotNet.Tensors.Tests.Engines.Compilation.CompiledEnrollmentCheckpointTests' }
)
$relativeNewTrx = 'TestResults/optimizer-enrollment-signature.trx'
$newTrxPath = Join-Path $EvidenceDirectory $relativeNewTrx
Assert-Feed ($null -ne $manifest.sha256.PSObject.Properties[$relativeNewTrx] -and
    $manifest.sha256.$relativeNewTrx -cmatch '^[0-9a-f]{64}$' -and (Test-Path -LiteralPath $newTrxPath) -and
    (Get-Sha256 $newTrxPath) -ceq $manifest.sha256.$relativeNewTrx) 'Mandatory CPU union TRX checksum did not match.'
[xml]$newTrx = Get-Content -LiteralPath $newTrxPath -Raw
$newResults = @($newTrx.SelectNodes('//*[local-name()="UnitTestResult"]'))
$newCounters = $newTrx.SelectSingleNode('//*[local-name()="ResultSummary"]/*[local-name()="Counters"]')
$newSummary = $newTrx.SelectSingleNode('//*[local-name()="ResultSummary"]')
Assert-Feed ($null -ne $newCounters -and $null -ne $newSummary -and $newSummary.GetAttribute('outcome') -ceq 'Completed') 'Mandatory CPU union TRX did not complete.'
$newNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($result in $newResults) {
    Assert-Feed ($result.GetAttribute('outcome') -ceq 'Passed' -and $newNames.Add($result.GetAttribute('testName'))) 'Mandatory CPU union TRX cases must be distinct and passed.'
}
$newExpectedTotal = 0
$newManifestTotal = 0
foreach ($group in $newCpuGroups) {
    $counterField = $group.Prefix + 'TestCounters'
    $caseField = $group.Prefix + 'TestCases'
    $countField = $group.Prefix + 'TestCount'
    $digestField = $group.Prefix + 'TestCasesSha256'
    Assert-Feed ($null -ne $pin.PSObject.Properties[$countField] -and $null -ne $pin.PSObject.Properties[$digestField]) "Missing pinned mandatory CPU group: $($group.Prefix)"
    $expectedCount = [int]$pin.$countField
    Assert-Feed ($expectedCount -gt 0 -and $pin.$digestField -cmatch '^[0-9a-f]{64}$') "Invalid pinned mandatory CPU group: $($group.Prefix)"
    Assert-Feed ($null -ne $manifest.PSObject.Properties[$counterField]) "Missing mandatory CPU test group: $($group.Prefix)"
    $groupCounters = $manifest.$counterField
    Assert-Feed ([int]$groupCounters.total -eq $expectedCount -and [int]$groupCounters.executed -eq $expectedCount -and [int]$groupCounters.passed -eq $expectedCount) "Mandatory CPU group count did not match: $($group.Prefix)"
    foreach ($outcome in @('failed', 'error', 'timeout', 'aborted', 'inconclusive', 'passedButRunAborted', 'notRunnable', 'notExecuted', 'disconnected', 'warning', 'completed', 'inProgress', 'pending')) {
        Assert-Feed ($null -ne $groupCounters.PSObject.Properties[$outcome]) "Missing mandatory CPU test outcome: $($group.Prefix).$outcome"
    }
    foreach ($counter in $groupCounters.PSObject.Properties) {
        if ($counter.Name -notin @('total', 'executed', 'passed')) { Assert-Feed ([int]$counter.Value -eq 0) "Nonzero mandatory CPU test outcome: $($group.Prefix).$($counter.Name)" }
    }
    Assert-Feed ($null -ne $manifest.PSObject.Properties[$caseField]) "Missing mandatory CPU test identities: $($group.Prefix)"
    $cases = [string[]]@($manifest.$caseField)
    Assert-Feed ($cases.Count -eq $expectedCount) "Mandatory CPU test identity count did not match: $($group.Prefix)"
    $identities = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($case in $cases) {
        Assert-Feed (-not [string]::IsNullOrWhiteSpace($case) -and $case.StartsWith($group.Class + '.', [StringComparison]::Ordinal) -and $identities.Add($case)) 'Mandatory CPU test identities must be nonempty and unique and belong to the pinned class.'
    }
    Assert-Feed ((Get-CaseDigest $cases) -ceq $pin.$digestField) "Mandatory CPU test identity digest did not match: $($group.Prefix)"
    $trxCases = [string[]]@($newResults | Where-Object { $_.GetAttribute('testName').StartsWith($group.Class + '.', [StringComparison]::Ordinal) } | ForEach-Object { $_.GetAttribute('testName') })
    Assert-Feed ($trxCases.Count -eq $expectedCount -and (Get-CaseDigest $trxCases) -ceq $pin.$digestField) "Mandatory CPU TRX identities did not match: $($group.Prefix)"
    $newExpectedTotal += $expectedCount
    $newManifestTotal += [int]$groupCounters.total
}
Assert-Feed ($newResults.Count -eq $newExpectedTotal -and [int]$newCounters.GetAttribute('total') -eq $newExpectedTotal -and [int]$newCounters.GetAttribute('executed') -eq $newExpectedTotal -and [int]$newCounters.GetAttribute('passed') -eq $newExpectedTotal) 'Mandatory CPU union TRX count did not match.'
foreach ($counter in $newCounters.Attributes) {
    if ($counter.Name -notin @('total', 'executed', 'passed')) { Assert-Feed ([int]$counter.Value -eq 0) "Nonzero mandatory CPU union TRX outcome: $($counter.Name)" }
}
Assert-Feed ($pin.mandatoryCpuTestCount -eq ($pin.testCount + $pin.retainedOutputTestCount + $newExpectedTotal) -and $manifest.mandatoryCpuTestCount -eq $pin.mandatoryCpuTestCount -and ([int]$counters.total + [int]$retainedCounters.total + $newManifestTotal) -eq $pin.mandatoryCpuTestCount) 'Mandatory CPU test total did not match.'
$relativePackage = "packages/AiDotNet.Tensors.$version.nupkg"
$package = Join-Path $EvidenceDirectory $relativePackage
Assert-Feed ($manifest.sha256.$relativePackage -ceq $pin.packageSha256 -and (Get-Sha256 $package) -ceq $pin.packageSha256) 'Package hash did not match.'
Assert-Feed ($manifest.testedAndPackagedNet10BinarySha256 -ceq $pin.testedAndPackagedNet10BinarySha256) 'Tested binary hash did not match.'
Add-Type -AssemblyName System.IO.Compression.ZipFile
$zip = [IO.Compression.ZipFile]::OpenRead($package)
try {
    $nuspecs = @($zip.Entries | Where-Object FullName -Like '*.nuspec')
    Assert-Feed ($nuspecs.Count -eq 1) 'Package must contain one nuspec.'
    $reader = [IO.StreamReader]::new($nuspecs[0].Open())
    try { [xml]$nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
    Assert-Feed ($nuspec.package.metadata.id -ceq 'AiDotNet.Tensors' -and $nuspec.package.metadata.version -ceq $version) 'Nuspec identity did not match.'
    foreach ($tfm in $pin.targetFrameworks) {
        Assert-Feed ($null -ne $zip.GetEntry("lib/$tfm/AiDotNet.Tensors.dll")) "Missing packaged framework: $tfm"
    }
    $stream = $zip.GetEntry('lib/net10.0/AiDotNet.Tensors.dll').Open()
    try { $binaryHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant() } finally { $stream.Dispose() }
    Assert-Feed ($binaryHash -ceq $pin.testedAndPackagedNet10BinarySha256) 'Packaged binary differs from tested binary.'
} finally { $zip.Dispose() }
if ($ValidateOnly) { Write-Host 'Pinned Tensors package evidence validated.'; return }
# Preserve nuget.org for baseline API validation and all other dependencies; only this local
# source is transient. Never mutate the repository or the runner global NuGet configuration.
$configPath = Join-Path $EvidenceDirectory 'NuGet.Config'
[xml] $config = Get-Content -LiteralPath (Join-Path $RepositoryRoot 'NuGet.config') -Raw
$source = $config.CreateElement('add')
$source.SetAttribute('key', 'tensors-validation')
$source.SetAttribute('value', (Join-Path $EvidenceDirectory 'packages'))
[void]$config.configuration.packageSources.AppendChild($source)
$config.Save($configPath)
Add-Content -LiteralPath $env:GITHUB_ENV -Value "RestoreConfigFile=$configPath"
Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value 'active=true'
