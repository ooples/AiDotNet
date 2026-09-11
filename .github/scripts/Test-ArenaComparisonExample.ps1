param([string] $DocumentPath = (Join-Path $PSScriptRoot '../../CI_SHARD_INVENTORY.md'))

$ErrorActionPreference = 'Stop'
$document = Microsoft.PowerShell.Management\Get-Content -LiteralPath $DocumentPath -Raw
$fence = ([string][char]96) * 3
$pattern = '(?s)' + [regex]::Escape($fence) + 'powershell\r?\n(?<code>.*?)' + [regex]::Escape($fence)
$blocks = [regex]::Matches($document, $pattern)
if ($blocks.Count -ne 2) { throw 'Expected exactly two PowerShell examples.' }
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput(
    $blocks[1].Groups['code'].Value, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
. ([scriptblock]::Create($blocks[1].Groups['code'].Value))

$checks = 0
foreach ($invalid in @('', ' ', 'FullyQualifiedName~<YourModelTests>')) {
    $rejected = $false
    try { Invoke-ArenaComparison -TestFilter $invalid -ErrorAction Stop }
    catch [System.Management.Automation.ParameterBindingException] { $rejected = $true }
    if (-not $rejected) { throw 'Invalid filter was accepted.' }
    $checks++
}

# Boundary doubles deliberately do not execute dotnet or touch a real model binary.
# This validates the documentation's control flow, not any model or arena behavior.
enum ArenaFixture { Executed; Empty; Skipped; Missing; ChangedTest; ChangedLibrary; ChangedTensor }
$script:fixture = [ArenaFixture]::Executed
$script:hashCalls = 0
function Get-FileHash {
    param($LiteralPath, $Algorithm)
    $script:hashCalls++
    $name = Split-Path $LiteralPath -Leaf
    $target = switch ($script:fixture) {
        ([ArenaFixture]::ChangedTest) { 'AiDotNetTests.dll' }
        ([ArenaFixture]::ChangedLibrary) { 'AiDotNet.dll' }
        ([ArenaFixture]::ChangedTensor) { 'AiDotNet.Tensors.dll' }
        default { '' }
    }
    [pscustomobject]@{ Hash = if ($name -eq $target -and $script:hashCalls -gt 3) { 'changed' } else { 'same' } }
}
function dotnet { $global:LASTEXITCODE = 0 }
function Test-Path { param($LiteralPath); $script:fixture -ne [ArenaFixture]::Missing }
function Get-Content {
    param($LiteralPath, [switch]$Raw)
    switch ($script:fixture) {
        ([ArenaFixture]::Empty) { '<TestRun><Results /></TestRun>' }
        ([ArenaFixture]::Skipped) { '<TestRun><Results><UnitTestResult outcome="NotExecuted" /></Results></TestRun>' }
        default { '<TestRun><Results><UnitTestResult outcome="Passed" /></Results></TestRun>' }
    }
}

$savedArena = [Environment]::GetEnvironmentVariable('AIDOTNET_INFERENCE_ARENA')
foreach ($case in [Enum]::GetValues([ArenaFixture])) {
    $script:fixture = $case
    $script:hashCalls = 0
    $rejected = $false
    try {
        $result = @(Invoke-ArenaComparison -TestFilter 'FullyQualifiedName~ActualFixture')
        if ($case -eq [ArenaFixture]::Executed -and
            ($result.Count -ne 2 -or $result[0].Arena -ne 1 -or $result[1].Arena -ne 0 -or
             $result[0].Executed -ne 1 -or $result[1].Executed -ne 1)) {
            throw 'Wrong comparison result.'
        }
    }
    catch {
        if ($case -eq [ArenaFixture]::Executed) { throw }
        $rejected = $true
    }
    if ($case -ne [ArenaFixture]::Executed -and -not $rejected) { throw "Unsafe fixture accepted: $case" }
    if ([Environment]::GetEnvironmentVariable('AIDOTNET_INFERENCE_ARENA') -cne $savedArena) {
        throw "Arena setting was not restored for $case."
    }
    $checks++
}
Write-Host "Inventory helper: $checks validation/outcome/restoration controls passed; no model-test pass claim."
