<#
.SYNOPSIS
    Writes a typed, run-bound certificate for a successful CI validation stage.

.DESCRIPTION
    Validation and quality analysis are deliberately separate scopes. A Validation certificate
    proves that the build/test/regression stage passed and permits a landed tree to skip that
    expensive work while still rerunning CodeQL and Sonar. A Complete certificate is emitted only
    after the repository CI gate passes and permits the landed tree to reuse both stages.
#>
[CmdletBinding(DefaultParameterSetName = 'Write')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Write')]
    [ValidateSet('Validation', 'Complete')]
    [string] $Scope,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [long] $RunId,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [string] $TestedSha,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [string] $TestedTree,
    [Parameter(Mandatory, ParameterSetName = 'Write')]
    [ValidateSet('pull_request', 'merge_group')]
    [string] $EventName,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [bool] $RequiresValidation,
    [Parameter(Mandatory, ParameterSetName = 'Write')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum CiValidationCertificateScope {
    Validation
    Complete
}

function New-CiValidationCertificate {
    param(
        [Parameter(Mandatory)] [CiValidationCertificateScope] $CertificateScope,
        [Parameter(Mandatory)] [long] $CertificateRunId,
        [Parameter(Mandatory)] [string] $CertificateTestedSha,
        [Parameter(Mandatory)] [string] $CertificateTestedTree,
        [Parameter(Mandatory)] [string] $CertificateEvent,
        [Parameter(Mandatory)] [bool] $CertificateRequiresValidation
    )

    if ($CertificateRunId -lt 1) { throw 'run id must be a positive integer' }
    if ($CertificateTestedSha -cnotmatch '^[0-9a-f]{40}$') {
        throw 'tested SHA must be a lowercase 40-character Git SHA'
    }
    if ($CertificateTestedTree -cnotmatch '^[0-9a-f]{40}$') {
        throw 'tested tree must be a lowercase 40-character Git SHA'
    }
    if ($CertificateEvent -cne 'pull_request' -and $CertificateEvent -cne 'merge_group') {
        throw "unsupported certificate event '$CertificateEvent'"
    }

    return [pscustomobject] [ordered]@{
        schemaVersion = 3
        scope = $CertificateScope.ToString()
        runId = $CertificateRunId
        testedSha = $CertificateTestedSha
        testedTree = $CertificateTestedTree
        event = $CertificateEvent
        requiresValidation = $CertificateRequiresValidation
        gateConclusion = 'success'
    }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $Message)
        if (-not $Condition) { [void] $failures.Add($Message) }
    }
    function Assert-Rejected { param([scriptblock] $Action, [string] $Message)
        $rejected = $false
        try { & $Action | Out-Null } catch { $rejected = $true }
        Assert-True $rejected $Message
    }

    $sha = '0123456789abcdef0123456789abcdef01234567'
    $tree = '89abcdef0123456789abcdef0123456789abcdef'
    $validation = New-CiValidationCertificate -CertificateScope Validation `
        -CertificateRunId 12 -CertificateTestedSha $sha -CertificateTestedTree $tree `
        -CertificateEvent pull_request -CertificateRequiresValidation $true
    Assert-True ($validation.schemaVersion -eq 3) 'certificate schema is not version 3'
    Assert-True ($validation.scope -ceq 'Validation') 'validation scope was not preserved'
    Assert-True ($validation.requiresValidation -is [bool]) `
        'requiresValidation is not represented as a JSON boolean'

    $complete = New-CiValidationCertificate -CertificateScope Complete `
        -CertificateRunId 13 -CertificateTestedSha $sha -CertificateTestedTree $tree `
        -CertificateEvent merge_group -CertificateRequiresValidation $false
    Assert-True ($complete.scope -ceq 'Complete') 'complete scope was not preserved'
    Assert-True (-not $complete.requiresValidation) 'non-runtime scope was not preserved'

    Assert-Rejected {
        New-CiValidationCertificate -CertificateScope Validation -CertificateRunId 0 `
            -CertificateTestedSha $sha -CertificateTestedTree $tree `
            -CertificateEvent pull_request -CertificateRequiresValidation $true
    } 'a zero run id was accepted'
    Assert-Rejected {
        New-CiValidationCertificate -CertificateScope Validation -CertificateRunId 1 `
            -CertificateTestedSha 'not-a-sha' -CertificateTestedTree $tree `
            -CertificateEvent pull_request -CertificateRequiresValidation $true
    } 'an invalid tested SHA was accepted'
    Assert-Rejected {
        New-CiValidationCertificate -CertificateScope Validation -CertificateRunId 1 `
            -CertificateTestedSha $sha -CertificateTestedTree 'not-a-tree' `
            -CertificateEvent pull_request -CertificateRequiresValidation $true
    } 'an invalid tested tree was accepted'

    if ($failures.Count -gt 0) {
        Write-Host 'New-CiValidationCertificate self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'New-CiValidationCertificate self-test passed.'
    exit 0
}

$certificateScope = [CiValidationCertificateScope] $Scope
$certificate = New-CiValidationCertificate -CertificateScope $certificateScope `
    -CertificateRunId $RunId -CertificateTestedSha $TestedSha `
    -CertificateTestedTree $TestedTree -CertificateEvent $EventName `
    -CertificateRequiresValidation $RequiresValidation

if (Test-Path -LiteralPath $OutFile) { throw "refusing to overwrite $OutFile" }
$parent = Split-Path -Parent $OutFile
if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
$certificate | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $OutFile -Encoding utf8
exit 0
