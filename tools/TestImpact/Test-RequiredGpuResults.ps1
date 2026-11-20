[CmdletBinding()]
param([string] $Verifier = (Join-Path $PSScriptRoot 'Assert-RequiredGpuResults.ps1'))
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('required-gpu-fixtures-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
$files = @('AiDotNet.dll','AiDotNet.Tensors.dll','AiDotNetTests.dll')
$sha = 'a' * 40
$digest = 'sha256:' + ('b' * 64)
$arguments = @{ Trx = "$temporary/results.trx"; BindingFile = "$temporary/binding.json"; ExpectedSha = $sha
    ExpectedRunId = '123'; ExpectedRunAttempt = '1'; ExpectedArtifactId = '456'; ExpectedArtifactDigest = $digest
    AssemblyDirectory = $temporary }
$testRoot = Join-Path $PSScriptRoot '../../tests/AiDotNet.Tests'
# Inventory comes independently from the actual case declarations, not from the acceptance script.
$identities = @()
foreach ($relative in @('Recovery/GpuRecoveryTests.cs','Concurrency/ThreadSafetyTests.cs')) {
    $source = Get-Content (Join-Path $testRoot $relative) -Raw
    $namespace = [regex]::Match($source, 'namespace ([A-Za-z.]+);').Groups[1].Value
    $class = [regex]::Match($source, 'public class ([A-Za-z]+)').Groups[1].Value
    foreach ($method in [regex]::Matches($source, '\[Fact(?:\([^\]]*\))?\]\s+public (?:async Task|void) ([A-Za-z0-9_]+)\(')) {
        $identities += "$namespace.$class.$($method.Groups[1].Value)"
    }
}
if ($identities.Count -ne 12) { throw 'Fixture source inventory changed; review GPU contract inventory.' }
$count = 0
function New-Fixture {
    foreach ($name in $files) { Set-Content -LiteralPath (Join-Path $temporary $name) -Value "fixture $name" }
    $records = @($files | ForEach-Object { @{ name = $_; sha256 = (Get-FileHash (Join-Path $temporary $_)).Hash.ToLowerInvariant() } })
    @{ sha = $sha; runId = '123'; runAttempt = '1'; artifactId = '456'; artifactDigest = $digest; assemblies = $records } |
        ConvertTo-Json -Depth 5 | Set-Content $arguments.BindingFile
    $doc = [xml]'<TestRun xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010"><Results/><TestDefinitions/><TestEntries/><ResultSummary outcome="Completed"><Counters/></ResultSummary></TestRun>'
    $namespace = $doc.DocumentElement.NamespaceURI
    $resultsNode = $doc.DocumentElement.ChildNodes[0]
    $definitionsNode = $doc.DocumentElement.ChildNodes[1]
    $entriesNode = $doc.DocumentElement.ChildNodes[2]
    $counters = $doc.DocumentElement.LastChild.FirstChild
    foreach ($key in @('total','executed','passed')) { $counters.SetAttribute($key, '12') }
    foreach ($key in @('failed','error','timeout','aborted','inconclusive','passedButRunAborted','notRunnable','notExecuted','disconnected','warning','completed','inProgress','pending')) { $counters.SetAttribute($key, '0') }
    foreach ($identity in $identities) {
        $id = [guid]::NewGuid().ToString(); $execution = [guid]::NewGuid().ToString()
        $definition = $doc.CreateElement('UnitTest',$namespace); $definition.SetAttribute('id',$id)
        $method = $doc.CreateElement('TestMethod',$namespace)
        $split = $identity.LastIndexOf('.')
        $method.SetAttribute('className',$identity.Substring(0,$split)); $method.SetAttribute('name',$identity.Substring($split+1))
        [void]$definition.AppendChild($method); [void]$definitionsNode.AppendChild($definition)
        $result = $doc.CreateElement('UnitTestResult',$namespace)
        $result.SetAttribute('testId',$id); $result.SetAttribute('executionId',$execution); $result.SetAttribute('outcome','Passed')
        $output = $doc.CreateElement('Output',$namespace); $stdout = $doc.CreateElement('StdOut',$namespace)
        $stdout.InnerText = "RequiredGpuContract: backend=CUDA; device=Fixture device`nGPU launches=8; kernel misses=0; fallbacks=`n" + (($records | ForEach-Object { "Assembly: $($_.name); SHA256=$($_.sha256)" }) -join "`n")
        [void]$output.AppendChild($stdout); [void]$result.AppendChild($output); [void]$resultsNode.AppendChild($result)
        $entry = $doc.CreateElement('TestEntry',$namespace); $entry.SetAttribute('testId',$id); $entry.SetAttribute('executionId',$execution)
        [void]$entriesNode.AppendChild($entry)
    }
    $doc.Save($arguments.Trx)
    return ,$doc
}
function Check-Fixture([string] $Name, [scriptblock] $Mutation, [bool] $ShouldPass = $false) {
    $doc = New-Fixture
    & $Mutation $doc
    $doc.Save($arguments.Trx)
    $accepted = $true
    try { & $Verifier @arguments | Out-Null } catch { $accepted = $false }
    if ($accepted -ne $ShouldPass) { throw "GPU fixture '$Name' had unexpected acceptance $accepted." }
    $script:count++
}
try {
    Check-Fixture complete {} $true
    Check-Fixture empty { param($d) $d.TestRun.Results.RemoveAll() }
    Check-Fixture missing { param($d) [void]$d.TestRun.Results.RemoveChild($d.TestRun.Results.FirstChild) }
    Check-Fixture failed { param($d) $d.TestRun.Results.FirstChild.SetAttribute('outcome','Failed') }
    Check-Fixture skipped { param($d) $d.TestRun.Results.FirstChild.SetAttribute('outcome','NotExecuted') }
    Check-Fixture duplicate { param($d) [void]$d.TestRun.Results.AppendChild($d.TestRun.Results.FirstChild.CloneNode($true)) }
    Check-Fixture duplicateIdentity { param($d) $first=$d.TestRun.TestDefinitions.FirstChild.TestMethod; $second=$d.TestRun.TestDefinitions.ChildNodes[1].TestMethod; $second.SetAttribute('className',$first.className); $second.SetAttribute('name',$first.name) }
    Check-Fixture wrongEntry { param($d) $d.TestRun.TestEntries.FirstChild.SetAttribute('executionId','wrong') }
    Check-Fixture missingEvidence { param($d) $d.TestRun.Results.FirstChild.Output.StdOut = '' }
    Check-Fixture cpu { param($d) $d.TestRun.Results.FirstChild.Output.StdOut = $d.TestRun.Results.FirstChild.Output.StdOut.Replace('backend=CUDA','backend=CPU') }
    Check-Fixture noLaunch { param($d) $d.TestRun.Results.FirstChild.Output.StdOut = $d.TestRun.Results.FirstChild.Output.StdOut.Replace('launches=8','launches=0') }
    Check-Fixture kernelMiss { param($d) $d.TestRun.Results.FirstChild.Output.StdOut = $d.TestRun.Results.FirstChild.Output.StdOut.Replace('misses=0','misses=1') }
    Check-Fixture fallback { param($d) $d.TestRun.Results.FirstChild.Output.StdOut = $d.TestRun.Results.FirstChild.Output.StdOut.Replace('fallbacks=','fallbacks=TensorAdd:driver failed') }
    Check-Fixture wrongLoadedHash { param($d) $d.TestRun.Results.FirstChild.Output.StdOut = $d.TestRun.Results.FirstChild.Output.StdOut.Replace('SHA256=','OTHER=') }
    foreach ($key in @('total','executed','passed','notExecuted','failed','inProgress')) {
        Check-Fixture "counter-$key" { param($d) $d.TestRun.ResultSummary.Counters.SetAttribute($key,'1') }.GetNewClosure()
    }
    Check-Fixture aborted { param($d) $d.TestRun.ResultSummary.SetAttribute('outcome','Aborted') }
    foreach ($key in @('sha','runId','runAttempt','artifactId','artifactDigest')) {
        Check-Fixture "binding-$key" { param($d) $binding=Get-Content $arguments.BindingFile -Raw | ConvertFrom-Json; $binding.$key='wrong'; $binding | ConvertTo-Json -Depth 5 | Set-Content $arguments.BindingFile }.GetNewClosure()
    }
    Check-Fixture changedDll { param($d) Add-Content (Join-Path $temporary 'AiDotNet.dll') 'different bytes' }
    Check-Fixture missingDll { param($d) Remove-Item -LiteralPath (Join-Path $temporary 'AiDotNet.Tensors.dll') }
    # The GPU requirement must survive every gate reuse mode and every non-success conclusion.
    $gate = Join-Path $PSScriptRoot 'Assert-CiGate.ps1'
    foreach ($reuse in @('None','Validation','Complete')) {
        foreach ($outcome in @('success','failure','skipped','cancelled','')) {
            & $gate -Stage Complete -ReuseScope $reuse -SourceResult success -RequiresValidation true `
                -ValidationGateResult success -PromotionResult success -RequiresGpu true -GpuResult $outcome | Out-Null
            $expectedExit = if ($outcome -ceq 'success') { 0 } else { 1 }
            if ($LASTEXITCODE -ne $expectedExit) { throw "GPU gate bypass: reuse=$reuse outcome=$outcome." }
            $count++
        }
    }
    # Exercise the workflow's actual obligation expressions: trust restricts runner access,
    # never whether a runtime run must obtain GPU proof before publishing a certificate.
    $workflow = Get-Content (Join-Path $PSScriptRoot '../../.github/workflows/sonarcloud.yml') -Raw
    $expressions = @([regex]::Matches($workflow, "-RequiresGpu '\$\{\{ (.*?) \}\}'") | ForEach-Object { $_.Groups[1].Value })
    $validationExpression = "needs.select-shards.outputs.requires_validation == 'true'"
    $completeExpression = "needs.validation-source.outputs.execute_validation == 'true' && needs.select-shards.outputs.requires_validation == 'true'"
    if ($expressions.Count -ne 2 -or $expressions[0] -cne $validationExpression -or $expressions[1] -cne $completeExpression) {
        throw 'GPU gate obligation expressions changed; trust must not waive the runtime obligation.'
    }
    foreach ($trusted in @($true,$false)) {
        foreach ($runtime in @($true,$false)) {
            # Convert the checked workflow expression, then use the real gate. The job is skipped
            # for an untrusted fork; required runtime evidence must consequently fail closed.
            $expression = $expressions[0].Replace("needs.select-shards.outputs.requires_validation", "'$($runtime.ToString().ToLowerInvariant())'").Replace(' == ', ' -ceq ')
            $owed = [bool](Invoke-Expression $expression)
            $gpuOutcome = if ($trusted) { 'success' } else { 'skipped' }
            & $gate -Stage Complete -ReuseScope None -SourceResult success -RequiresValidation ($runtime.ToString().ToLowerInvariant()) `
                -ValidationGateResult success -RequiresGpu ($owed.ToString().ToLowerInvariant()) -GpuResult $gpuOutcome | Out-Null
            $expectedExit = if ($runtime -and -not $trusted) { 1 } else { 0 }
            if ($LASTEXITCODE -ne $expectedExit) { throw "GPU fork trust gate bypass: trusted=$trusted runtime=$runtime." }
            $count++
        }
    }
    Write-Host "Required GPU acceptance fixtures passed: $count."
}
finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
# Negative native gate calls set LASTEXITCODE; do not leak their expected failure to Actions.
$global:LASTEXITCODE = 0
exit 0
