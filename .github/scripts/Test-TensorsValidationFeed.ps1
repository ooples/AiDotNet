[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'Initialize-TensorsValidationFeed.ps1'
$root = Join-Path ([IO.Path]::GetTempPath()) "tensors-feed-test-$([guid]::NewGuid().ToString('N'))"
[void](New-Item -ItemType Directory -Path "$root/.github", "$root/evidence/packages", "$root/lib")
try {
    $pin = Get-Content (Join-Path $PSScriptRoot '../tensors-validation.json') -Raw | ConvertFrom-Json
    $version = $pin.packageVersion
    $props = "<Project><ItemGroup><PackageVersion Include=`"AiDotNet.Tensors`" Version=`"$version`" /></ItemGroup></Project>"
    Set-Content "$root/Directory.Packages.props" $props
    Add-Type -AssemblyName System.IO.Compression.ZipFile
    $package = "$root/evidence/packages/AiDotNet.Tensors.$version.nupkg"
    $zip = [IO.Compression.ZipFile]::Open($package, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($tfm in $pin.targetFrameworks) {
            $entry = $zip.CreateEntry("lib/$tfm/AiDotNet.Tensors.dll")
            $writer = [IO.StreamWriter]::new($entry.Open())
            try { $writer.Write('fixture binary') } finally { $writer.Dispose() }
        }
        $writer = [IO.StreamWriter]::new($zip.CreateEntry('AiDotNet.Tensors.nuspec').Open())
        try { $writer.Write("<package><metadata><id>AiDotNet.Tensors</id><version>$version</version></metadata></package>") } finally { $writer.Dispose() }
    } finally { $zip.Dispose() }
    $pin.packageSha256 = (Get-FileHash $package -Algorithm SHA256).Hash.ToLowerInvariant()
    $pin.testedAndPackagedNet10BinarySha256 = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes('fixture binary'))).ToLowerInvariant()
    $pin | ConvertTo-Json -Depth 10 | Set-Content "$root/.github/tensors-validation.json"
    $manifest = @{
        repository = $pin.repository; sourceCommit = $pin.sourceCommit; packageVersion = $version
        runAttempt = $pin.runAttempt; workflowRun = "https://github.com/$($pin.repository)/actions/runs/$($pin.runId)"
        targetFrameworks = $pin.targetFrameworks
        testedAndPackagedNet10BinarySha256 = $pin.testedAndPackagedNet10BinarySha256
        testCounters = @{ total = $pin.testCount; executed = $pin.testCount; passed = $pin.testCount; failed = 0; notExecuted = 0 }
        retainedOutputTestCounters = @{ total = $pin.retainedOutputTestCount; executed = $pin.retainedOutputTestCount; passed = $pin.retainedOutputTestCount }
        retainedOutputTestCases = @(1..$pin.retainedOutputTestCount | ForEach-Object { "Fixture.RetainedOutput.Case$_" })
        mandatoryCpuTestCount = $pin.mandatoryCpuTestCount
        sha256 = @{ "packages/AiDotNet.Tensors.$version.nupkg" = $pin.packageSha256 }
    }
    foreach ($outcome in @('failed', 'error', 'timeout', 'aborted', 'inconclusive', 'passedButRunAborted', 'notRunnable', 'notExecuted', 'disconnected', 'warning', 'completed', 'inProgress', 'pending')) {
        $manifest.retainedOutputTestCounters[$outcome] = 0
    }
    function Test-Case([string] $Name, [scriptblock] $Mutate, [string] $ExpectedError) {
        $case = $manifest | ConvertTo-Json -Depth 10 | ConvertFrom-Json
        & $Mutate $case
        $case | ConvertTo-Json -Depth 10 | Set-Content "$root/evidence/manifest.json"
        $caught = $null
        try { & $script -RepositoryRoot $root -EvidenceDirectory "$root/evidence" -ValidateOnly } catch { $caught = $_.Exception.Message }
        if ($ExpectedError) {
            if (-not $caught -or $caught -notlike "*$ExpectedError*") { throw "${Name}: expected '$ExpectedError', got '$caught'" }
        } elseif ($caught) { throw "${Name}: $caught" }
        Write-Host "PASS: $Name"
    }
    Test-Case 'valid package evidence' {} ''
    Test-Case 'wrong source commit' { param($m) $m.sourceCommit = 'wrong' } 'Manifest identity'
    Test-Case 'skipped producer tests' { param($m) $m.testCounters.notExecuted = 1 } 'Nonzero test outcome'
    Test-Case 'missing retained-output group' { param($m) $m.PSObject.Properties.Remove('retainedOutputTestCounters') } 'Missing retained-output test group'
    Test-Case 'skipped retained-output tests' { param($m) $m.retainedOutputTestCounters.notExecuted = 1 } 'Nonzero retained-output test outcome'
    Test-Case 'duplicate retained-output identities' { param($m) $m.retainedOutputTestCases[1] = $m.retainedOutputTestCases[0] } 'must be nonempty and unique'
    Test-Case 'missing retained-output identity' { param($m) $m.retainedOutputTestCases = @($m.retainedOutputTestCases | Select-Object -Skip 1) } 'test identity count'
    Test-Case 'wrong mandatory CPU total' { param($m) $m.mandatoryCpuTestCount-- } 'Mandatory CPU test total'
    Test-Case 'missing target framework' { param($m) $m.targetFrameworks = @('net10.0') } 'target frameworks'
    Test-Case 'wrong package hash' { param($m) $m.sha256."packages/AiDotNet.Tensors.$version.nupkg" = 'wrong' } 'Package hash'
    Test-Case 'wrong tested binary' { param($m) $m.testedAndPackagedNet10BinarySha256 = 'wrong' } 'Tested binary hash'
    Set-Content "$root/Directory.Packages.props" ($props.Replace($version, '0.1.0-coordination.unknown'))
    Test-Case 'unrecognized coordination version' {} 'does not match the pinned descriptor'
    Set-Content "$root/Directory.Packages.props" $props
    $oldRepository = $env:GITHUB_REPOSITORY
    $oldEvent = $env:GITHUB_EVENT_NAME
    $oldEventPath = $env:GITHUB_EVENT_PATH
    try {
        $env:GITHUB_REPOSITORY = 'ooples/AiDotNet'
        $env:GITHUB_EVENT_NAME = 'pull_request'
        $env:GITHUB_EVENT_PATH = "$root/event.json"
        '{"pull_request":{"head":{"repo":{"full_name":"fork/AiDotNet"}}}}' | Set-Content $env:GITHUB_EVENT_PATH
        $caught = $null
        try { & $script -RepositoryRoot $root } catch { $caught = $_.Exception.Message }
        if ($caught -notlike '*untrusted events and forks*') { throw "Fork guard did not fail closed: $caught" }
        Write-Host 'PASS: fork fails before token or network access'
        $env:GITHUB_EVENT_NAME = 'merge_group'
        $caught = $null
        try { & $script -RepositoryRoot $root } catch { $caught = $_.Exception.Message }
        if ($caught -notlike '*including merge_group*') { throw "Merge queue guard did not fail closed: $caught" }
        Write-Host 'PASS: merge_group fails before token or network access'
    } finally {
        $env:GITHUB_REPOSITORY = $oldRepository
        $env:GITHUB_EVENT_NAME = $oldEvent
        $env:GITHUB_EVENT_PATH = $oldEventPath
    }
    # Mock only the existing read-only transport boundary; exercise the real bootstrap.
    [void](New-Item -ItemType Directory -Path "$root/tools/TestImpact", "$root/runner")
    @'
function Invoke-GitHubApiWithRetry {
    param($Uri, $Headers)
    $p = Get-Content "$env:TENSORS_FIXTURE_ROOT/.github/tensors-validation.json" -Raw | ConvertFrom-Json
    if ($Uri -like '*/runs/*') {
        return [pscustomobject]@{ id=$p.runId; head_sha=$p.sourceCommit; path=$p.workflowPath; run_attempt=$p.runAttempt; status='completed'; conclusion='success' }
    }
    return [pscustomobject]@{ id=$p.artifactId; name=$p.artifactName; digest=$p.artifactDigest; expired=$false; workflow_run=@{ id=$p.runId; head_sha=$p.sourceCommit } }
}
'@ | Set-Content "$root/tools/TestImpact/Invoke-GitHubApiWithRetry.ps1"
    @'
param($Repository, $ArtifactId, $ExpectedDigest, $Destination, $RunId, $JobIndex)
if ($env:GH_TOKEN -cne 'fixture-artifact' -or $env:GITHUB_TOKEN -cne 'fixture-artifact') { throw 'Download token override failed.' }
if ($env:TENSORS_FIXTURE_EXIT -eq '17') { exit 17 }
Copy-Item -LiteralPath "$env:TENSORS_FIXTURE_ROOT/evidence" -Destination $Destination -Recurse
exit 0
'@ | Set-Content "$root/tools/TestImpact/Receive-RequiredArtifact.ps1"
    '<configuration><packageSources><add key="nuget.org" value="https://api.nuget.org/v3/index.json" /></packageSources></configuration>' | Set-Content "$root/NuGet.config"
    # The mocked downloader deliberately sets a failing native exit code. Keep that
    # expected failure from leaking into the enclosing GitHub PowerShell step.
    $savedExitCode = if (Test-Path variable:global:LASTEXITCODE) { $global:LASTEXITCODE } else { 0 }
    $savedEnvironment = @{}
    $fixtureEnvironment = @{
        GITHUB_REPOSITORY='ooples/AiDotNet'; GITHUB_EVENT_NAME='push'; RUNNER_TEMP="$root/runner"
        GITHUB_TOKEN='inherited-github'; GH_TOKEN='inherited-gh'; TENSORS_ARTIFACT_TOKEN='fixture-artifact'
        TENSORS_FIXTURE_ROOT=$root; TENSORS_FIXTURE_EXIT='0'; GITHUB_ENV="$root/env"; GITHUB_OUTPUT="$root/output"
    }
    try {
        foreach ($name in $fixtureEnvironment.Keys) {
            $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
            [Environment]::SetEnvironmentVariable($name, $fixtureEnvironment[$name])
        }
        Set-Content "$root/Directory.Packages.props" $props
        $manifest | ConvertTo-Json -Depth 10 | Set-Content "$root/evidence/manifest.json"
        & $script -RepositoryRoot $root
        if ($env:GH_TOKEN -cne 'inherited-gh' -or $env:GITHUB_TOKEN -cne 'inherited-github') { throw 'Successful download did not restore both tokens.' }
        Write-Host 'PASS: inherited token priority overridden and restored after success'
        $env:TENSORS_FIXTURE_EXIT = '17'
        $caught = $null
        try { & $script -RepositoryRoot $root } catch { $caught = $_.Exception.Message }
        if ($caught -notlike '*download failed with exit code 17*') { throw "Download failure was not preserved: $caught" }
        if ($env:GH_TOKEN -cne 'inherited-gh' -or $env:GITHUB_TOKEN -cne 'inherited-github') { throw 'Failed download did not restore both tokens.' }
        Write-Host 'PASS: nonzero downloader exit fails before manifest read and restores both tokens'
    } finally {
        foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name]) }
        $global:LASTEXITCODE = $savedExitCode
    }
    Set-Content "$root/Directory.Packages.props" ($props.Replace($version, '0.151.1'))
    Move-Item -LiteralPath "$root/.github/tensors-validation.json" -Destination "$root/descriptor.saved.json"
    Test-Case 'stable version needs no evidence or credentials' {} ''
    Move-Item -LiteralPath "$root/descriptor.saved.json" -Destination "$root/.github/tensors-validation.json"
} finally {
    # Only this newly created, uniquely named fixture tree is removed.
    if (([IO.Path]::GetFullPath($root)).StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()), [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $root -Recurse -Force
    }
}
