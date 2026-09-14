<#
.SYNOPSIS
    Applies a shard's declared environment (the `env:` map of its .github/test-shards.yml entry)
    to the current process, so the test run and its targeted retry see the same configuration.
.DESCRIPTION
    A sweep or conformance shard is one test class parameterized by environment variables: which
    eighth of the parameter-count inventory to check, which five-model conformance window to probe.
    Without them the same filter runs the WHOLE inventory in one test host, which is exactly what
    those shards exist to avoid (the count contract crashes its host after the third row). So the
    first run and the retry must both apply it, and this is the one place that does.

    The map comes from a repository file, but it is still input to a job holding secrets, so names
    are restricted to the library's own configuration prefixes. A shard cannot use this to replace
    PATH, the runner's GITHUB_/RUNNER_/ACTIONS_ variables, or the license key.
.PARAMETER Json
    `toJSON(matrix.shard.env)` - a JSON object, or the text `null` when the entry declares none.
.PARAMETER SelfTest
    Runs the validation checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Apply')]
param(
    [Parameter(ParameterSetName = 'Apply')] [AllowEmptyString()] [string] $Json = '',
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ShardEnvironment {
    param([AllowEmptyString()] [string] $Text)

    $pairs = [System.Collections.Generic.List[object]]::new()
    if ([string]::IsNullOrWhiteSpace($Text) -or $Text.Trim() -eq 'null') { return , $pairs }

    try { $parsed = $Text | ConvertFrom-Json }
    catch { throw "shard env is not valid JSON: $($_.Exception.Message)" }
    if ($parsed -isnot [pscustomobject]) { throw 'shard env must be a JSON object of NAME: value pairs' }

    foreach ($property in $parsed.PSObject.Properties) {
        $name = [string] $property.Name
        if ($name -cnotmatch '^(AIDOTNET|ADNSHAPE)_[A-Z0-9_]+$') {
            throw "shard env name '$name' is not an AIDOTNET_ or ADNSHAPE_ configuration variable"
        }
        if ($name -ceq 'AIDOTNET_LICENSE_KEY') { throw 'shard env may not set AIDOTNET_LICENSE_KEY' }
        $value = $property.Value
        if ($null -ne $value -and $value -isnot [string] -and $value -isnot [ValueType]) {
            throw "shard env value for '$name' must be a scalar"
        }
        [void] $pairs.Add([pscustomobject]@{ Name = $name; Value = [string] $value })
    }
    return , $pairs
}

if ($SelfTest) {
    function Assert-Throws([scriptblock] $Block, [string] $What) {
        $threw = $false
        try { & $Block | Out-Null } catch { $threw = $true }
        if (-not $threw) { throw "self-test: expected a rejection for $What" }
    }

    if ((Get-ShardEnvironment -Text 'null').Count -ne 0) { throw 'self-test: null must mean no variables' }
    if ((Get-ShardEnvironment -Text '').Count -ne 0) { throw 'self-test: empty must mean no variables' }
    $ok = Get-ShardEnvironment -Text '{"ADNSHAPE_CONF_OFFSET":"10","AIDOTNET_PARAMETER_COUNT_SHARD":3}'
    if ($ok.Count -ne 2 -or $ok[0].Value -ne '10' -or $ok[1].Value -ne '3') { throw 'self-test: valid map misread' }
    Assert-Throws { Get-ShardEnvironment -Text '{"PATH":"/tmp"}' } 'PATH'
    Assert-Throws { Get-ShardEnvironment -Text '{"GITHUB_TOKEN":"x"}' } 'GITHUB_TOKEN'
    Assert-Throws { Get-ShardEnvironment -Text '{"AIDOTNET_LICENSE_KEY":"x"}' } 'the license key'
    Assert-Throws { Get-ShardEnvironment -Text '{"aidotnet_lower":"x"}' } 'a lower-case name'
    Assert-Throws { Get-ShardEnvironment -Text '{"AIDOTNET_X":{"nested":1}}' } 'a nested value'
    Assert-Throws { Get-ShardEnvironment -Text '["AIDOTNET_X"]' } 'an array'
    Assert-Throws { Get-ShardEnvironment -Text '{not json' } 'malformed JSON'
    Write-Host 'Set-ShardEnvironment self-test passed.'
    exit 0
}

foreach ($pair in (Get-ShardEnvironment -Text $Json)) {
    [Environment]::SetEnvironmentVariable($pair.Name, $pair.Value)
    Write-Host "shard env: $($pair.Name)=$($pair.Value)"
}
exit 0
