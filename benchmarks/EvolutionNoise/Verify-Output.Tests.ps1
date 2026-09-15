param(
    [Parameter(Mandatory = $true)][string] $Report,
    [Parameter(Mandatory = $true)][string] $Assembly
)
$ErrorActionPreference = 'Stop'
$before = (Get-FileHash -LiteralPath $Report).Hash
& dotnet $Assembly --output $Report 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0 -or (Get-FileHash -LiteralPath $Report).Hash -ne $before) {
    throw 'An existing report must be refused without changing its contents.'
}
& dotnet $Assembly --unknown 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { throw 'Invalid arguments must be refused.' }
Write-Output 'PASS: existing report preserved and invalid arguments refused.'
# GitHub's pwsh wrapper propagates LASTEXITCODE; expected native failures must not leak out.
exit 0
