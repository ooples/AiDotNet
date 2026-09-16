<#
.SYNOPSIS
    Provides bounded retry handling for read-only GitHub API requests.

.DESCRIPTION
    Retries only transient HTTP responses and transport failures. Permission, identity, and other
    permanent errors fail immediately. Exhausting the retry budget rethrows the original request
    error so a diagnostic job cannot turn missing GitHub state into a successful result.
#>
[CmdletBinding()]
param(
    [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum GitHubApiRequestDisposition {
    Retry
    Fail
}

function Get-GitHubApiRequestDisposition {
    param(
        [ValidateRange(0, 599)]
        [int] $HttpStatus,
        [ValidateRange(0, [int]::MaxValue)]
        [int] $RetryAfterSeconds,
        [AllowEmptyString()]
        [string] $ResponseMessage,
        [bool] $IsTransportFailure
    )

    if ($HttpStatus -eq 403 -and ($RetryAfterSeconds -gt 0 -or
        ([string] $ResponseMessage).Contains('secondary rate limit', [StringComparison]::OrdinalIgnoreCase) -or
        ([string] $ResponseMessage).Contains('rate limit exceeded', [StringComparison]::OrdinalIgnoreCase))) {
        return [GitHubApiRequestDisposition]::Retry
    }

    if ($HttpStatus -in @(408, 425, 429) -or ($HttpStatus -ge 500 -and $HttpStatus -le 599)) {
        return [GitHubApiRequestDisposition]::Retry
    }

    if ($HttpStatus -eq 0 -and $IsTransportFailure) {
        return [GitHubApiRequestDisposition]::Retry
    }

    return [GitHubApiRequestDisposition]::Fail
}

function Get-GitHubApiRetryAfterSeconds {
    param([object] $Response)

    if ($null -eq $Response) { return 0 }
    $headersProperty = $Response.PSObject.Properties['Headers']
    if ($null -eq $headersProperty -or $null -eq $headersProperty.Value) { return 0 }
    $headers = $headersProperty.Value

    $retryAfterProperty = $headers.PSObject.Properties['RetryAfter']
    if ($null -ne $retryAfterProperty -and $null -ne $retryAfterProperty.Value) {
        $retryAfter = $retryAfterProperty.Value
        $deltaProperty = $retryAfter.PSObject.Properties['Delta']
        if ($null -ne $deltaProperty -and $null -ne $deltaProperty.Value) {
            return [math]::Max(0, [int] [math]::Ceiling($deltaProperty.Value.TotalSeconds))
        }
        $dateProperty = $retryAfter.PSObject.Properties['Date']
        if ($null -ne $dateProperty -and $null -ne $dateProperty.Value) {
            $remaining = $dateProperty.Value.UtcDateTime - [DateTime]::UtcNow
            return [math]::Max(0, [int] [math]::Ceiling($remaining.TotalSeconds))
        }
    }

    try {
        $values = @($headers.GetValues('Retry-After'))
        if ($values.Count -gt 0) {
            $seconds = 0
            if ([int]::TryParse([string] $values[$values.Count - 1], [ref] $seconds)) {
                return [math]::Max(0, $seconds)
            }
        }
    }
    catch {
        # The strongly typed RetryAfter property above is the normal HttpResponseHeaders path.
    }

    return 0
}

function Get-GitHubApiErrorDetails {
    param([Management.Automation.ErrorRecord] $ErrorRecord)

    $httpStatus = 0
    $response = $null
    $isTransportFailure = $false
    $exception = $ErrorRecord.Exception
    $current = $exception
    while ($null -ne $current) {
        if ($current -is [Net.Http.HttpRequestException] -or
            $current -is [Net.WebException] -or
            $current -is [TaskCanceledException]) {
            $isTransportFailure = $true
        }

        if ($httpStatus -eq 0) {
            $statusProperty = $current.PSObject.Properties['StatusCode']
            if ($null -ne $statusProperty -and $null -ne $statusProperty.Value) {
                $httpStatus = [int] $statusProperty.Value
            }
        }

        if ($null -eq $response) {
            $responseProperty = $current.PSObject.Properties['Response']
            if ($null -ne $responseProperty) { $response = $responseProperty.Value }
        }

        $current = $current.InnerException
    }

    if ($httpStatus -eq 0 -and $null -ne $response) {
        $statusProperty = $response.PSObject.Properties['StatusCode']
        if ($null -ne $statusProperty -and $null -ne $statusProperty.Value) {
            $httpStatus = [int] $statusProperty.Value
        }
    }

    $errorDetailsMessage = if ($null -eq $ErrorRecord.ErrorDetails) {
        ''
    }
    else {
        [string] $ErrorRecord.ErrorDetails.Message
    }
    $message = if ([string]::IsNullOrWhiteSpace($errorDetailsMessage)) {
        [string] $exception.Message
    }
    else {
        $errorDetailsMessage
    }

    return [pscustomobject] @{
        HttpStatus = $httpStatus
        RetryAfterSeconds = Get-GitHubApiRetryAfterSeconds -Response $response
        ResponseMessage = $message
        IsTransportFailure = $isTransportFailure
    }
}

function Get-GitHubApiRetryDelaySeconds {
    param(
        [ValidateRange(1, 4)]
        [int] $FailedAttempt,
        [ValidateRange(0, 599)]
        [int] $HttpStatus,
        [ValidateRange(0, [int]::MaxValue)]
        [int] $RetryAfterSeconds
    )

    # GitHub asks clients to wait at least one minute after rate limiting. Ordinary transport and
    # server errors use short exponential backoff so one transient 500 does not consume the job.
    $minimum = if ($HttpStatus -in @(403, 429)) {
        60 * (1 -shl ($FailedAttempt - 1))
    }
    else {
        2 * (1 -shl ($FailedAttempt - 1))
    }
    return [math]::Max($minimum, $RetryAfterSeconds)
}

function Invoke-GitHubApiWithRetry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ValidatePattern('^https://')]
        [string] $Uri,
        [Parameter(Mandatory)]
        [hashtable] $Headers,
        [string] $OutFile,
        [ValidateRange(1, 4)]
        [int] $MaxAttempts = 4,
        [Parameter(DontShow = $true)]
        [scriptblock] $RequestInvoker,
        [Parameter(DontShow = $true)]
        [scriptblock] $DelayInvoker
    )

    $hasOutputFile = $PSBoundParameters.ContainsKey('OutFile')
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            if ($null -ne $RequestInvoker) {
                return & $RequestInvoker $Uri $Headers $(if ($hasOutputFile) { $OutFile } else { $null })
            }
            if ($hasOutputFile) {
                return Invoke-WebRequest -Headers $Headers -Uri $Uri -OutFile $OutFile `
                    -ConnectionTimeoutSeconds 30 -OperationTimeoutSeconds 120
            }
            return Invoke-RestMethod -Headers $Headers -Uri $Uri `
                -ConnectionTimeoutSeconds 30 -OperationTimeoutSeconds 120
        }
        catch {
            $details = Get-GitHubApiErrorDetails -ErrorRecord $_
            $disposition = Get-GitHubApiRequestDisposition `
                -HttpStatus $details.HttpStatus `
                -RetryAfterSeconds $details.RetryAfterSeconds `
                -ResponseMessage $details.ResponseMessage `
                -IsTransportFailure $details.IsTransportFailure
            if ($disposition -eq [GitHubApiRequestDisposition]::Fail -or $attempt -eq $MaxAttempts) {
                throw
            }

            $delay = Get-GitHubApiRetryDelaySeconds -FailedAttempt $attempt `
                -HttpStatus $details.HttpStatus -RetryAfterSeconds $details.RetryAfterSeconds
            Write-Warning "GitHub API request returned transient status $($details.HttpStatus); retrying attempt $($attempt + 1)/$MaxAttempts after $delay second(s)."
            if ($null -ne $DelayInvoker) {
                & $DelayInvoker $delay
            }
            else {
                Start-Sleep -Seconds $delay
            }
        }
    }
}

function Assert-True {
    param([bool] $Condition, [string] $Message)
    if (-not $Condition) { throw $Message }
}

if ($SelfTest) {
    Assert-True ((Get-GitHubApiRequestDisposition 500 0 '' $false) -eq
        [GitHubApiRequestDisposition]::Retry) 'HTTP 500 was not classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 429 0 '' $false) -eq
        [GitHubApiRequestDisposition]::Retry) 'HTTP 429 was not classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 403 17 '' $false) -eq
        [GitHubApiRequestDisposition]::Retry) 'Retry-After 403 was not classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 403 0 'secondary rate limit' $false) -eq
        [GitHubApiRequestDisposition]::Retry) 'secondary-limit 403 was not classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 403 0 'Resource not accessible' $false) -eq
        [GitHubApiRequestDisposition]::Fail) 'permission 403 was incorrectly classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 404 0 '' $false) -eq
        [GitHubApiRequestDisposition]::Fail) 'HTTP 404 was incorrectly classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 0 0 '' $true) -eq
        [GitHubApiRequestDisposition]::Retry) 'transport failure was not classified as transient'
    Assert-True ((Get-GitHubApiRequestDisposition 0 0 '' $false) -eq
        [GitHubApiRequestDisposition]::Fail) 'non-transport exception was incorrectly classified as transient'
    Assert-True ((Get-GitHubApiRetryDelaySeconds 1 500 0) -eq 2) `
        'ordinary server-error backoff is not bounded to two seconds initially'
    Assert-True ((Get-GitHubApiRetryDelaySeconds 1 403 73) -eq 73) `
        'Retry-After was not honored for a rate-limited request'
    $retryResponse = [Net.Http.HttpResponseMessage]::new([Net.HttpStatusCode]::TooManyRequests)
    try {
        $retryResponse.Headers.RetryAfter = [Net.Http.Headers.RetryConditionHeaderValue]::new(
            [TimeSpan]::FromSeconds(37))
        Assert-True ((Get-GitHubApiRetryAfterSeconds $retryResponse) -eq 37) `
            'typed HTTP Retry-After response header was not parsed'
    }
    finally {
        $retryResponse.Dispose()
    }

    $script:successAttempts = 0
    $script:successDelays = [Collections.Generic.List[int]]::new()
    $script:successResponse = [Net.Http.HttpResponseMessage]::new(
        [Net.HttpStatusCode]::InternalServerError)
    try {
        # Invoke-RestMethod uses HttpResponseException for an HTTP 500. This exercises the same
        # Response.StatusCode extraction path as the failure observed in Actions.
        $eventualSuccess = Invoke-GitHubApiWithRetry -Uri 'https://api.github.test/eventual' `
            -Headers @{} -RequestInvoker {
                param($RequestUri, $RequestHeaders, $DownloadPath)
                $script:successAttempts++
                if ($script:successAttempts -eq 1) {
                    throw [Microsoft.PowerShell.Commands.HttpResponseException]::new(
                        'Server Error', $script:successResponse)
                }
                return 'success'
            } -DelayInvoker { param($Seconds) $script:successDelays.Add($Seconds) }
    }
    finally {
        $script:successResponse.Dispose()
    }
    Assert-True ($eventualSuccess -eq 'success') 'a transient failure did not return its eventual result'
    Assert-True ($script:successAttempts -eq 2) 'a single HTTP 500 did not cause exactly one retry'
    Assert-True ($script:successDelays.Count -eq 1 -and $script:successDelays[0] -eq 2) `
        'a single HTTP 500 did not use the expected first delay'

    $script:permanentAttempts = 0
    $script:permanentDelays = 0
    $permanentFailed = $false
    try {
        Invoke-GitHubApiWithRetry -Uri 'https://api.github.test/permanent' -Headers @{} `
            -RequestInvoker {
                param($RequestUri, $RequestHeaders, $DownloadPath)
                $script:permanentAttempts++
                throw [Net.Http.HttpRequestException]::new(
                    'Not Found', $null, [Net.HttpStatusCode]::NotFound)
            } -DelayInvoker { param($Seconds) $script:permanentDelays++ }
    }
    catch {
        $permanentFailed = $true
    }
    Assert-True $permanentFailed 'a permanent HTTP 404 was swallowed'
    Assert-True ($script:permanentAttempts -eq 1) 'a permanent HTTP 404 was retried'
    Assert-True ($script:permanentDelays -eq 0) 'a permanent HTTP 404 invoked backoff'

    $script:exhaustedAttempts = 0
    $script:exhaustedDelays = 0
    $exhaustionFailed = $false
    try {
        Invoke-GitHubApiWithRetry -Uri 'https://api.github.test/exhaustion' -Headers @{} `
            -MaxAttempts 3 -RequestInvoker {
                param($RequestUri, $RequestHeaders, $DownloadPath)
                $script:exhaustedAttempts++
                throw [Net.Http.HttpRequestException]::new(
                    'Server Error', $null, [Net.HttpStatusCode]::BadGateway)
            } -DelayInvoker { param($Seconds) $script:exhaustedDelays++ }
    }
    catch {
        $exhaustionFailed = $true
    }
    Assert-True $exhaustionFailed 'retry exhaustion was swallowed'
    Assert-True ($script:exhaustedAttempts -eq 3) 'retry exhaustion did not use the exact attempt budget'
    Assert-True ($script:exhaustedDelays -eq 2) 'retry exhaustion delayed after the final attempt'

    Write-Host 'GitHub API retry self-test passed.'
    exit 0
}
