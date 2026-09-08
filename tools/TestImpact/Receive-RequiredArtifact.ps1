<#
.SYNOPSIS
    Downloads one required GitHub Actions artifact by immutable ID.

.DESCRIPTION
    actions/download-artifact resolves an artifact name by listing every artifact in the workflow
    run. Large matrices make those concurrent ListArtifacts calls hit GitHub's secondary rate limit.
    This script consumes the artifact ID and digest emitted by actions/upload-artifact instead, uses
    the supported REST archive endpoint directly, and validates the downloaded archive before it is
    extracted.

    Transient responses use bounded exponential backoff with deterministic jitter. A permission or
    identity error fails immediately, and exhausting the retry budget remains a hard failure: callers
    must never report an untested shard as successful.
#>
[CmdletBinding(DefaultParameterSetName = 'Download')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Download')]
    [ValidatePattern('^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')]
    [string] $Repository,

    [Parameter(Mandatory, ParameterSetName = 'Download')]
    [ValidateRange(1, [long]::MaxValue)]
    [long] $ArtifactId,

    [Parameter(Mandatory, ParameterSetName = 'Download')]
    [ValidatePattern('^sha256:[0-9a-fA-F]{64}$')]
    [string] $ExpectedDigest,

    [Parameter(Mandatory, ParameterSetName = 'Download')]
    [string] $Destination,

    [Parameter(Mandatory, ParameterSetName = 'Download')]
    [ValidateRange(1, [long]::MaxValue)]
    [long] $RunId,

    [Parameter(Mandatory, ParameterSetName = 'Download')]
    [ValidateRange(0, [int]::MaxValue)]
    [int] $JobIndex,

    [Parameter(ParameterSetName = 'Download')]
    [ValidateRange(1, 5)]
    [int] $MaxAttempts = 3,

    [Parameter(Mandatory, ParameterSetName = 'SelfTest')]
    [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# curl's exit code is input to the retry policy, not a PowerShell exception. Pin this preference so
# future runner-image defaults cannot bypass classification before the response is inspected.
$PSNativeCommandUseErrorActionPreference = $false

enum ArtifactRequestDisposition {
    Success
    Retry
    Fail
}

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw $Message }
}

function Get-ArtifactRequestDisposition {
    param(
        [int] $CurlExitCode,
        [int] $HttpStatus,
        [AllowEmptyString()] [string] $ResponseBody,
        [ValidateRange(0, [int]::MaxValue)] [int] $RetryAfterSeconds = 0
    )

    if ($CurlExitCode -eq 0 -and $HttpStatus -eq 200) {
        return [ArtifactRequestDisposition]::Success
    }

    if ($HttpStatus -eq 403 -and ($RetryAfterSeconds -gt 0 -or
        $ResponseBody.Contains('secondary rate limit', [StringComparison]::OrdinalIgnoreCase))) {
        return [ArtifactRequestDisposition]::Retry
    }

    # A connection reset or partial transfer can occur after the archive endpoint has returned 200.
    # curl reports that through its exit code; treating the HTTP status alone as terminal would turn
    # a recoverable truncated download into a hard failure before digest validation can run.
    if ($CurlExitCode -ne 0 -and $HttpStatus -ge 200 -and $HttpStatus -le 399) {
        return [ArtifactRequestDisposition]::Retry
    }

    if ($HttpStatus -in @(0, 408, 425, 429) -or ($HttpStatus -ge 500 -and $HttpStatus -le 599)) {
        return [ArtifactRequestDisposition]::Retry
    }

    return [ArtifactRequestDisposition]::Fail
}

function Get-RetryAfterSeconds {
    param([AllowEmptyString()] [string] $Headers)

    $matches = [Regex]::Matches($Headers, '(?im)^retry-after:\s*(?<seconds>\d+)\s*\r?$')
    if ($matches.Count -eq 0) { return 0 }

    $seconds = 0
    if (-not [int]::TryParse($matches[$matches.Count - 1].Groups['seconds'].Value, [ref] $seconds)) {
        return 0
    }
    return $seconds
}

function Get-RetryDelaySeconds {
    param(
        [ValidateRange(1, 4)] [int] $FailedAttempt,
        [ValidateRange(1, [long]::MaxValue)] [long] $WorkflowRunId,
        [ValidateRange(0, [int]::MaxValue)] [int] $MatrixJobIndex,
        [ValidateRange(0, [int]::MaxValue)] [int] $RetryAfterSeconds
    )

    # GitHub requires at least one minute after a secondary limit. The run and matrix identities
    # spread simultaneous failures across a 31-second window so retries do not form another burst.
    $minimum = 60 * (1 -shl ($FailedAttempt - 1))
    $jitter = [int] ((($WorkflowRunId % 31) + (($MatrixJobIndex % 31) * 17)) % 31)
    return [math]::Max($minimum, $RetryAfterSeconds) + $jitter
}

function Test-ArtifactDigest {
    param([string] $Path, [string] $Expected)

    $actual = 'sha256:' + (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return $actual.Equals($Expected.ToLowerInvariant(), [StringComparison]::Ordinal)
}

function Get-SmallResponseBody {
    param([string] $Path)

    if (-not (Test-Path -LiteralPath $Path)) { return '' }
    $file = Get-Item -LiteralPath $Path
    if ($file.Length -gt 65536) { return '' }
    return [string] (Get-Content -LiteralPath $Path -Raw)
}

function Get-CurlApplicationPath {
    # Ubuntu exposes the same executable through /usr/bin and /bin. Property enumeration on the
    # unbounded Get-Command result joins both paths into one invalid invocation target.
    $command = Get-Command curl -CommandType Application -ErrorAction Stop |
        Select-Object -First 1
    return [string] $command.Source
}

if ($SelfTest) {
    $secondary = '{"message":"You have exceeded a secondary rate limit. Please wait."}'
    Assert-True ((Get-ArtifactRequestDisposition 22 403 $secondary) -eq
        [ArtifactRequestDisposition]::Retry) 'a secondary-limit 403 was not classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 22 403 '{"message":"Forbidden"}' 17) -eq
        [ArtifactRequestDisposition]::Retry) 'a Retry-After 403 was not classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 22 403 '' 17) -eq
        [ArtifactRequestDisposition]::Retry) 'a Retry-After 403 was not classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 22 403 '{"message":"Resource not accessible"}') -eq
        [ArtifactRequestDisposition]::Fail) 'a permission 403 was incorrectly classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 22 429 '') -eq
        [ArtifactRequestDisposition]::Retry) 'HTTP 429 was not classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 22 503 '') -eq
        [ArtifactRequestDisposition]::Retry) 'HTTP 503 was not classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 18 200 '') -eq
        [ArtifactRequestDisposition]::Retry) 'a partial HTTP 200 transfer was not classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 22 404 '') -eq
        [ArtifactRequestDisposition]::Fail) 'HTTP 404 was incorrectly classified as transient'
    Assert-True ((Get-ArtifactRequestDisposition 0 200 '') -eq
        [ArtifactRequestDisposition]::Success) 'HTTP 200 was not classified as success'

    $headers = "HTTP/2 403`r`nretry-after: 17`r`n`r`nHTTP/2 403`r`nRetry-After: 93`r`n"
    Assert-True ((Get-RetryAfterSeconds $headers) -eq 93) 'the final Retry-After header was not honored'
    Assert-True ((Get-RetryAfterSeconds '') -eq 0) 'missing Retry-After did not return zero'

    $first = Get-RetryDelaySeconds -FailedAttempt 1 -WorkflowRunId 100 -MatrixJobIndex 7 `
        -RetryAfterSeconds 0
    $second = Get-RetryDelaySeconds -FailedAttempt 2 -WorkflowRunId 100 -MatrixJobIndex 7 `
        -RetryAfterSeconds 0
    Assert-True ($first -ge 60 -and $first -le 90) 'first retry violated the one-minute minimum'
    Assert-True ($second -ge 120 -and $second -le 150) 'second retry is not exponential'
    Assert-True ((Get-RetryDelaySeconds -FailedAttempt 1 -WorkflowRunId 100 -MatrixJobIndex 7 `
        -RetryAfterSeconds 180) -ge 180) 'Retry-After was shortened'

    $curlApplicationPath = Get-CurlApplicationPath
    Assert-True (-not [string]::IsNullOrWhiteSpace($curlApplicationPath)) `
        'curl application resolution returned no executable path'
    Assert-True (Test-Path -LiteralPath $curlApplicationPath -PathType Leaf) `
        'curl application resolution did not return one executable'

    $digestFixture = Join-Path ([IO.Path]::GetTempPath()) `
        ("aidotnet-artifact-digest-" + [guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($digestFixture, 'abc', [Text.UTF8Encoding]::new($false))
        Assert-True (Test-ArtifactDigest $digestFixture `
            'sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad') `
            'artifact SHA-256 validation rejected known content'
        Assert-True (-not (Test-ArtifactDigest $digestFixture `
            'sha256:ca7816bf8f01cfea414140de5dae2223b00361a396177a9cbb410ff61f20015ad')) `
            'artifact SHA-256 validation accepted a mismatched digest'
    }
    finally {
        if (Test-Path -LiteralPath $digestFixture) {
            Remove-Item -LiteralPath $digestFixture -Force
        }
    }

    Write-Host 'Required artifact transport self-test passed.'
    exit 0
}

if ([string]::IsNullOrWhiteSpace($env:GITHUB_TOKEN)) {
    throw 'GITHUB_TOKEN is required to download a workflow artifact by ID'
}

$curlApplicationPath = Get-CurlApplicationPath
$destinationPath = [IO.Path]::GetFullPath($Destination)
[IO.Directory]::CreateDirectory($destinationPath) | Out-Null

$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) `
    ("aidotnet-required-artifact-" + [guid]::NewGuid().ToString('N'))
$archivePath = Join-Path $temporaryDirectory 'artifact.zip'
$headersPath = Join-Path $temporaryDirectory 'headers.txt'
[IO.Directory]::CreateDirectory($temporaryDirectory) | Out-Null

try {
    $url = "https://api.github.com/repos/$Repository/actions/artifacts/$ArtifactId/zip"

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        foreach ($path in @($archivePath, $headersPath)) {
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
        }

        Write-Host "Downloading required artifact $ArtifactId directly (attempt $attempt/$MaxAttempts)."
        $statusText = @(& $curlApplicationPath `
            --silent --show-error --location --proto-redir '=https' --fail-with-body `
            --connect-timeout 30 --max-time 300 `
            --output $archivePath --dump-header $headersPath --write-out '%{http_code}' `
            --header 'Accept: application/vnd.github+json' `
            --header "Authorization: Bearer $env:GITHUB_TOKEN" `
            --header 'X-GitHub-Api-Version: 2022-11-28' `
            --user-agent 'AiDotNet-CI-artifact-transport' `
            $url)
        $curlExitCode = $LASTEXITCODE
        $statusValue = ($statusText -join '').Trim()
        $httpStatus = 0
        [void] [int]::TryParse($statusValue, [ref] $httpStatus)
        $responseBody = Get-SmallResponseBody $archivePath
        $headerText = if (Test-Path -LiteralPath $headersPath) {
            [string] (Get-Content -LiteralPath $headersPath -Raw)
        }
        else { '' }
        $retryAfter = Get-RetryAfterSeconds $headerText
        $disposition = Get-ArtifactRequestDisposition -CurlExitCode $curlExitCode `
            -HttpStatus $httpStatus -ResponseBody $responseBody -RetryAfterSeconds $retryAfter

        if ($disposition -eq [ArtifactRequestDisposition]::Success) {
            if (-not (Test-ArtifactDigest -Path $archivePath -Expected $ExpectedDigest)) {
                throw "artifact $ArtifactId failed SHA-256 validation"
            }

            [IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $destinationPath, $true)
            Write-Host "Required artifact $ArtifactId passed digest validation and was extracted."
            exit 0
        }

        if ($disposition -eq [ArtifactRequestDisposition]::Fail -or $attempt -eq $MaxAttempts) {
            $detail = if ($responseBody) { $responseBody } else { 'no response body' }
            throw "required artifact $ArtifactId download failed (curl=$curlExitCode, HTTP=$httpStatus): $detail"
        }

        $delay = Get-RetryDelaySeconds -FailedAttempt $attempt -WorkflowRunId $RunId `
            -MatrixJobIndex $JobIndex -RetryAfterSeconds $retryAfter
        Write-Host "Transient artifact response (curl=$curlExitCode, HTTP=$httpStatus); retrying after $delay seconds."
        Start-Sleep -Seconds $delay
    }
}
finally {
    foreach ($path in @($archivePath, $headersPath)) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }
    if (Test-Path -LiteralPath $temporaryDirectory) {
        Remove-Item -LiteralPath $temporaryDirectory -Force
    }
}
