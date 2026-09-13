[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [AllowEmptyString()]
    [string] $MatrixJson
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($MatrixJson)) {
    throw 'Selected shard matrix is unavailable; refusing to publish an unverifiable test inventory.'
}

try {
    $document = [System.Text.Json.JsonDocument]::Parse($MatrixJson)
    if ($document.RootElement.ValueKind -ne [System.Text.Json.JsonValueKind]::Array) {
        throw 'the root value must be an array'
    }
    $matrix = @($MatrixJson | ConvertFrom-Json -ErrorAction Stop)
}
catch {
    throw "Could not parse the selected shard matrix: $($_.Exception.Message)"
}
finally {
    if ($null -ne $document) { $document.Dispose() }
}

if ($matrix.Count -eq 0) {
    throw 'Selected shard matrix contains no shards.'
}

$names = New-Object System.Collections.Generic.List[string]
$keys = New-Object System.Collections.Generic.HashSet[string]([StringComparer]::Ordinal)
foreach ($shard in $matrix) {
    $name = [string] $shard.name
    if ([string]::IsNullOrWhiteSpace($name)) {
        throw 'Selected shard matrix contains a shard with no name.'
    }

    $key = $name -replace '[\\/:*?"<>|\s-]+', '_'
    if (-not $keys.Add($key)) {
        throw "Selected shard names collide after normalization: '$name' maps to '$key'."
    }
    $names.Add($name)
}

# The success stream is the contract consumed by the workflow.
$names
