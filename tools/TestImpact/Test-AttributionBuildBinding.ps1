[CmdletBinding()]
param([Parameter(Mandatory)][string] $EvidenceDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $root) { throw 'Evidence directory must be new.' }
[void](New-Item -ItemType Directory -Path $root)
$project = Join-Path $PSScriptRoot 'Attribution.Xunit/Attribution.Xunit.csproj'
function Read-Bindings {
    $result = [ordered]@{}
    foreach ($projectName in @('Attribution.Protocol', 'Attribution.Runtime', 'Attribution.Xunit')) {
        $assembly = if ($projectName -ceq 'Attribution.Runtime') { 'AttributionRuntime' } else { $projectName }
        foreach ($extension in @('dll', 'pdb')) {
            $path = Join-Path $PSScriptRoot "$projectName/bin/Release/net10.0/$assembly.$extension"
            $result["$assembly.$extension"] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        }
    }
    return $result
}
$initial = Read-Bindings
$runs = @()
foreach ($optIn in @('false', 'true')) {
    $revision = if ($optIn -ceq 'false') { 'a' * 40 } else { 'b' * 40 }
    $log = Join-Path $root "build-$optIn.log"
    # An up-to-date no-op cannot establish compiler output stability.
    & dotnet build $project -c Release --no-restore -t:Rebuild -m:2 -p:UseSharedCompilation=false `
        "-p:EnableTestAttribution=$optIn" "-p:SourceRevisionId=$revision" --nologo -v:quiet *> $log
    if ($LASTEXITCODE -ne 0) { throw "Build binding control failed; see $log" }
    $hashes = Read-Bindings
    foreach ($name in $initial.Keys) {
        if ($initial[$name] -cne $hashes[$name]) { throw "Unchanged $name differs for attribution=$optIn and revision=$revision." }
    }
    $runs += [pscustomobject]@{ AttributionOptIn = $optIn; SourceRevisionId = $revision; Hashes = $hashes }
}
[pscustomobject]@{ Schema = 1; IdenticalToolBinariesAndSymbols = $true; Runs = $runs } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'proof.json') -Encoding utf8
'Attribution build binding passed: unchanged DLL/PDB bytes across both revision/build-mode inputs.'
