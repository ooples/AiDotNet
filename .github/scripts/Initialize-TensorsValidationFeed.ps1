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
