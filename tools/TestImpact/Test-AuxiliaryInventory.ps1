[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/AuxiliaryInventory.ps1"
$source = 'using System; namespace Models; public class Model<T> : Base<T> { public Model(int size) { } public int Predict() { return 1; } }'
$original = Get-AuxiliaryInventorySignature $source
if ($original -cne (Get-AuxiliaryInventorySignature ($source.Replace('return 1;', 'return 2;')))) {
    throw 'A method-body change invalidated the model inventory.'
}
if ($original -cne (Get-AuxiliaryInventorySignature ($source.Replace('return 1;', "if (true) { return 2; }`n return 1;")))) {
    throw 'A method-body control-flow edit changed the inventory.'
}
if ($original -cne (Get-AuxiliaryInventorySignature ($source.Replace('public int Predict()', '/* comment */ public  int Predict()')))) {
    throw 'Comments or formatting changed the inventory.'
}
foreach ($changed in @(
    $source.Replace('Model<T>', 'Renamed<T>'),
    $source.Replace('namespace Models;', 'namespace Other;'),
    $source.Replace('public class', 'public abstract class'),
    $source.Replace('Base<T>', 'Different<T>'),
    $source.Replace('int size', 'long size'),
    $source.Replace('int Predict()', 'int Predict(int count)'),
    $source.Replace('using System;', 'using System.Text;'),
    ($source + ' public class Added<T> { }')
)) {
    if ($original -ceq (Get-AuxiliaryInventorySignature $changed)) {
        throw 'A declaration change retained an unsafe ordinal inventory.'
    }
}
$conditional = "#if CUSTOM`n$source`n#endif"
if ((Get-AuxiliaryInventorySignature $conditional) -ceq
    (Get-AuxiliaryInventorySignature ($conditional.Replace('return 1;', 'return 2;')))) {
    throw 'Conditional compilation hid an unprovable inventory change.'
}
$rejected = $false
try { $null = Get-AuxiliaryInventorySignature 'public class {' }
catch { $rejected = $true }
if (-not $rejected) { throw 'Invalid syntax authorized a stable inventory.' }
if (-not (Test-AuxiliaryInventoryChange -MapSha 'invalid-map-source')) {
    throw 'Missing metadata authorized auxiliary omissions.'
}
# Name-keyed windows. FNV-1a('a') = 0xE40C292C, so its shard of 8 is 4; ParameterCountContractTests.ShardOf
# asserts the same value (StableModelShardingTests), which keeps the two implementations equal.
if ((Get-StableParameterCountShard 'a') -ne 4) { throw 'the PowerShell FNV-1a shard disagrees with the known answer' }
if ((Get-StableParameterCountShard 'AiDotNet.NeuralNetworks.ResNet`1') -ne (Get-StableParameterCountShard 'AiDotNet.NeuralNetworks.ResNet`1')) {
    throw 'the stable shard is not stable'
}
$model = 'namespace Models { public class Outer { public class Model<T> : Base<T> { public Model(int size) { } public int Predict() { return 1; } } } }'
$ids = Get-AuxiliaryModelIdentities $model
if (-not $ids.ContainsKey('Models.Outer+Model`1') -or -not $ids.ContainsKey('Models.Outer')) {
    throw "model identities are not spelled as Type.FullName: $(@($ids.Keys) -join ', ')"
}
if ($ids['Models.Outer+Model`1'] -cne (Get-AuxiliaryModelIdentities ($model.Replace('return 1;', 'return 2;')))['Models.Outer+Model`1']) {
    throw 'a method body moved a name-keyed window'
}
if ($ids['Models.Outer+Model`1'] -cne (Get-AuxiliaryModelIdentities ($model.Replace('int Predict()', 'int Predict(int n)')))['Models.Outer+Model`1']) {
    throw 'a non-constructor signature moved a name-keyed window'
}
foreach ($changed in @($model.Replace('int size', 'long size'), $model.Replace(': Base<T>', ': Other<T>'), $model.Replace('public class Model', 'public abstract class Model'))) {
    if ($ids['Models.Outer+Model`1'] -ceq (Get-AuxiliaryModelIdentities $changed)['Models.Outer+Model`1']) {
        throw 'a constructor or declaration change did not change the model identity'
    }
}
if (-not (Get-AuxiliaryModelIdentities ($model.Replace('namespace Models', 'namespace Other'))).ContainsKey('Other.Outer+Model`1')) {
    throw 'a namespace change did not rename the identity'
}
if ($null -ne (Get-AuxiliaryModelIdentities "#if X`n$model`n#endif")) { throw 'conditional compilation authorized a name-keyed reduction' }
$table = Get-ConformanceWindowAssignments "        [""A.B``1""] = 0,`n        [""A.C``1""] = 3,"
if ($table.Count -ne 2 -or $table['A.C`1'] -ne 3) { throw 'the conformance window table did not parse' }
$fallback = Get-AuxiliaryWindowInvalidation -MapSha 'invalid-map-source' -Windows @([pscustomobject]@{ name = 'W0'; env = [pscustomobject]@{ ADNSHAPE_CONF_WINDOW = '0' } })
if (-not $fallback.All -or @($fallback.Shards) -cne @('W0')) { throw 'missing metadata authorized window omissions' }
Write-Host 'Auxiliary inventory passed: body-only reduction, eight declaration changes, directives, invalid syntax and invalid-map fallback; name-keyed windows: FNV known answer, Type.FullName identities, constructor-only sensitivity, table parse and fallback.'
exit 0
