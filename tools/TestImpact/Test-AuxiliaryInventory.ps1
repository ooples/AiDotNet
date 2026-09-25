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
Write-Host 'Auxiliary inventory passed: body-only reduction, eight declaration changes, directives, invalid syntax and invalid-map fallback.'
exit 0
