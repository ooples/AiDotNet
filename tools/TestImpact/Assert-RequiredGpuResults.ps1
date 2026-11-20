<# Strict acceptance for the required GPU lane. A green process without complete executed TRX is not proof. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Trx,
    [Parameter(Mandatory)][string] $BindingFile,
    [Parameter(Mandatory)][string] $ExpectedSha,
    [Parameter(Mandatory)][string] $ExpectedRunId,
    [Parameter(Mandatory)][string] $ExpectedRunAttempt,
    [Parameter(Mandatory)][string] $ExpectedArtifactId,
    [Parameter(Mandatory)][string] $ExpectedArtifactDigest,
    [Parameter(Mandatory)][string] $AssemblyDirectory
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$expected = @(
    'AiDotNet.Tests.Recovery.GpuRecoveryTests.AvailableBackend_ReportsCudaDeviceAndExecutes'
    'AiDotNet.Tests.Recovery.GpuRecoveryTests.DiagnosticReport_IdentifiesActiveBackendAndExecutes'
    'AiDotNet.Tests.Recovery.GpuRecoveryTests.InvalidMatrixDimensions_DoNotPoisonFollowingOperation'
    'AiDotNet.Tests.Recovery.GpuRecoveryTests.RepeatedOperations_KeepInputsAndResultsIndependent'
    'AiDotNet.Tests.Recovery.GpuRecoveryTests.Dispose_DoesNotDamageFreshEngine'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.VectorOperations_ConcurrentThreads_MatchIndependentReference'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.MatrixMultiply_ConcurrentThreads_MatchIndependentReference'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.MixedOperations_ConcurrentThreads_KeepResultsIndependent'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.Conv2D_ConcurrentThreads_MatchIndependentReference'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.Pooling_ConcurrentThreads_MatchIndependentReference'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.CapabilityReads_ConcurrentWithOperations_RemainConsistent'
    'AiDotNet.Tests.Concurrency.ThreadSafetyTests.RepeatedMixedLoad_CompletesEveryOperationWithoutDeadlock'
)
if ($ExpectedSha -cnotmatch '^[0-9a-f]{40}$') { throw 'Invalid expected source SHA.' }
if ($ExpectedArtifactDigest -cnotmatch '^sha256:[0-9a-f]{64}$') { throw 'Invalid expected artifact digest.' }
$binding = Get-Content -LiteralPath $BindingFile -Raw | ConvertFrom-Json
foreach ($pair in @(@('sha',$ExpectedSha),@('runId',$ExpectedRunId),@('runAttempt',$ExpectedRunAttempt),@('artifactId',$ExpectedArtifactId),@('artifactDigest',$ExpectedArtifactDigest))) {
    if ([string]$binding.($pair[0]) -cne [string]$pair[1]) { throw "GPU binding mismatch: $($pair[0])." }
}
$files = @('AiDotNet.dll','AiDotNet.Tensors.dll','AiDotNetTests.dll')
if (@($binding.assemblies).Count -ne $files.Count) { throw 'Incomplete assembly binding.' }
$hashes = @{}
foreach ($name in $files) {
    $record = @($binding.assemblies | Where-Object { $_.name -ceq $name })
    if ($record.Count -ne 1) { throw "Missing/duplicate binding for $name." }
    $file = Join-Path $AssemblyDirectory $name
    $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -cne $record[0].sha256) { throw "GPU assembly hash mismatch: $name." }
    $hashes[$name] = $hash
}
# XmlReader disables DTD/entity expansion; only the one primary report is accepted.
$settings = [System.Xml.XmlReaderSettings]::new()
$settings.DtdProcessing = [System.Xml.DtdProcessing]::Prohibit
$settings.XmlResolver = $null
$reader = [System.Xml.XmlReader]::Create((Resolve-Path -LiteralPath $Trx).Path, $settings)
$doc = [System.Xml.XmlDocument]::new()
$doc.XmlResolver = $null
try { $doc.Load($reader) } finally { $reader.Dispose() }
$ns = [System.Xml.XmlNamespaceManager]::new($doc.NameTable)
$ns.AddNamespace('t','http://microsoft.com/schemas/VisualStudio/TeamTest/2010')
if ($doc.SelectSingleNode('/t:TestRun/t:ResultSummary', $ns).GetAttribute('outcome') -cne 'Completed') { throw 'GPU run did not complete successfully.' }
$counters = $doc.SelectSingleNode('/t:TestRun/t:ResultSummary/t:Counters', $ns)
if ($null -eq $counters) { throw 'Missing GPU counters.' }
foreach ($name in @('total','executed','passed')) {
    if ($counters.GetAttribute($name) -cne [string]$expected.Count) { throw "GPU counter $name is incomplete." }
}
foreach ($name in @('failed','error','timeout','aborted','inconclusive','passedButRunAborted','notRunnable','notExecuted','disconnected','warning','completed','inProgress','pending')) {
    if ($counters.GetAttribute($name) -cne '0') { throw "GPU counter $name was not zero." }
}
$definitions = @($doc.SelectNodes('/t:TestRun/t:TestDefinitions/t:UnitTest', $ns))
$results = @($doc.SelectNodes('/t:TestRun/t:Results/t:UnitTestResult', $ns))
$entries = @($doc.SelectNodes('/t:TestRun/t:TestEntries/t:TestEntry', $ns))
if ($definitions.Count -ne $expected.Count -or $results.Count -ne $expected.Count -or $entries.Count -ne $expected.Count) { throw 'Incomplete or duplicate GPU test inventory.' }
$seenNames = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$seenExecutions = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($result in $results) {
    if ($result.GetAttribute('outcome') -cne 'Passed') { throw 'GPU result was not Passed.' }
    $testId = $result.GetAttribute('testId')
    $executionId = $result.GetAttribute('executionId')
    if ([string]::IsNullOrWhiteSpace($executionId) -or -not $seenExecutions.Add($executionId)) { throw 'Duplicate/missing GPU execution identity.' }
    $definition = @($definitions | Where-Object { $_.GetAttribute('id') -ceq $testId })
    if ($definition.Count -ne 1) { throw 'Missing/duplicate GPU test definition.' }
    $method = $definition[0].SelectSingleNode('t:TestMethod', $ns)
    $identity = $method.GetAttribute('className').Split(',')[0] + '.' + $method.GetAttribute('name')
    if ($expected -cnotcontains $identity -or -not $seenNames.Add($identity)) { throw "Unexpected/duplicate GPU test: $identity." }
    $entry = @($entries | Where-Object { $_.GetAttribute('testId') -ceq $testId -and $_.GetAttribute('executionId') -ceq $executionId })
    if ($entry.Count -ne 1) { throw 'Missing/duplicate GPU execution entry.' }
    $output = $result.SelectSingleNode('t:Output/t:StdOut', $ns)
    if ($null -eq $output) { throw "Missing actual CUDA execution evidence: $identity." }
    $text = $output.InnerText
    if ($text -cnotmatch 'RequiredGpuContract: backend=CUDA; device=\S') { throw "Missing CUDA capability evidence: $identity." }
    $launches = [regex]::Matches($text, 'GPU launches=([0-9]+); kernel misses=([0-9]+); fallbacks=([^\r\n]*)')
    if ($launches.Count -eq 0) { throw "Missing launch evidence: $identity." }
    foreach ($launch in $launches) {
        if ([long]$launch.Groups[1].Value -le 0 -or $launch.Groups[2].Value -cne '0' -or $launch.Groups[3].Value.Trim().Length -ne 0) { throw "GPU fallback/no-launch evidence: $identity." }
    }
    foreach ($name in $files) {
        if ($text.ToLowerInvariant() -notmatch [regex]::Escape('sha256=' + $hashes[$name])) { throw "Missing actual loaded assembly hash: $identity / $name." }
    }
}
Write-Host "Required GPU contract PASSED: $($expected.Count) exact identities executed on CUDA; no skips, misses or fallbacks. Source $ExpectedSha, run $ExpectedRunId/$ExpectedRunAttempt, artifact $ExpectedArtifactId."
