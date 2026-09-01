<#
.SYNOPSIS
    Merges per-shard coverage digests into the map that drives test selection.

.DESCRIPTION
    The complete shard manifest is authoritative. A valid digest with executed files makes a shard
    selectable; a missing digest or a valid digest with zero executed files makes it always-run.
    Unknown, duplicate or malformed digests abort map creation.

.PARAMETER DigestDirectory
    Directory holding *.digest.json files.

.PARAMETER AllShards
    The complete current shard manifest. Names are array elements and are never split on commas.

.PARAMETER Sha
    The exact commit the coverage run tested.

.PARAMETER OutFile
    Where to write shard-map.json.

.PARAMETER SelfTest
    Runs adversarial producer checks and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Build')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $DigestDirectory,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string[]] $AllShards,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $Sha,
    [Parameter(Mandatory, ParameterSetName = 'Build')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-ValidatedInteger {
    param(
        [Parameter(Mandatory)] $Value,
        [Parameter(Mandatory)] [string] $What,
        [int] $Minimum = 0,
        [int] $Maximum = [int]::MaxValue
    )

    $isInteger = $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    if (-not $isInteger) { throw "$What must be an integer." }
    $number = [long] $Value
    if ($number -lt $Minimum -or $number -gt $Maximum) {
        throw "$What must be between $Minimum and $Maximum."
    }
    return [int] $number
}

function Read-ValidatedDigest {
    param([Parameter(Mandatory)] [System.IO.FileInfo] $File)

    try { $digest = Get-Content -LiteralPath $File.FullName -Raw | ConvertFrom-Json }
    catch { throw "Digest $($File.Name) is not valid JSON: $($_.Exception.Message)" }

    foreach ($required in 'schemaVersion', 'shard', 'files') {
        if (-not $digest.PSObject.Properties[$required]) { throw "Digest $($File.Name) has no '$required' property." }
    }
    $version = ConvertTo-ValidatedInteger -Value $digest.schemaVersion -What "Digest $($File.Name) schemaVersion" -Minimum 1
    if ($version -ne 1) { throw "Digest $($File.Name) has unsupported schemaVersion $version." }

    $name = [string] $digest.shard
    if ([string]::IsNullOrWhiteSpace($name)) { throw "Digest $($File.Name) has no shard name." }
    if ($digest.files -isnot [pscustomobject]) { throw "Digest $($File.Name) files must be an object." }

    $fileProperties = @($digest.files.PSObject.Properties)
    if ($digest.PSObject.Properties['fileCount']) {
        $fileCount = ConvertTo-ValidatedInteger -Value $digest.fileCount -What "Digest $($File.Name) fileCount"
        if ($fileCount -ne $fileProperties.Count) {
            throw "Digest $($File.Name) fileCount $fileCount does not match $($fileProperties.Count) files."
        }
    }

    foreach ($property in $fileProperties) {
        if ([string]::IsNullOrWhiteSpace([string] $property.Name)) { throw "Digest $($File.Name) has an empty file path." }
        if ($property.Value -isnot [array]) { throw "Digest $($File.Name) '$($property.Name)' ranges must be an array." }
        $ranges = @($property.Value)
        if ($ranges.Count -eq 0 -or $ranges.Count % 2 -ne 0) {
            throw "Digest $($File.Name) '$($property.Name)' ranges must be nonempty start/end pairs."
        }
        for ($i = 0; $i -lt $ranges.Count; $i += 2) {
            $start = ConvertTo-ValidatedInteger -Value $ranges[$i] `
                -What "Digest $($File.Name) '$($property.Name)' range start" -Minimum 1
            $end = ConvertTo-ValidatedInteger -Value $ranges[$i + 1] `
                -What "Digest $($File.Name) '$($property.Name)' range end" -Minimum 1
            if ($start -gt $end) {
                throw "Digest $($File.Name) '$($property.Name)' range start $start exceeds end $end."
            }
        }
    }

    return $digest
}

function New-ShardMapObject {
    param(
        [Parameter(Mandatory)] [string] $DigestDirectory,
        [Parameter(Mandatory)] [string[]] $AllShards,
        [Parameter(Mandatory)] [string] $Sha
    )

    if ($Sha -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
        throw 'Sha must be a complete hexadecimal Git object id.'
    }
    if (-not (Test-Path -LiteralPath $DigestDirectory -PathType Container)) {
        throw "Digest directory not found: $DigestDirectory"
    }

    $manifest = [System.Collections.Generic.List[string]]::new()
    $manifestSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($value in @($AllShards)) {
        $name = [string] $value
        if ([string]::IsNullOrWhiteSpace($name)) { throw 'AllShards contains an empty shard name.' }
        if (-not $manifestSet.Add($name)) { throw "AllShards contains duplicate shard '$name'." }
        [void] $manifest.Add($name)
    }
    if ($manifest.Count -eq 0) { throw 'AllShards is empty.' }

    $digests = [System.Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
    foreach ($file in @(Get-ChildItem -LiteralPath $DigestDirectory -Filter '*.digest.json' -Recurse -File | Sort-Object FullName)) {
        $digest = Read-ValidatedDigest -File $file
        $name = [string] $digest.shard
        if (-not $manifestSet.Contains($name)) { throw "Digest $($file.Name) names unknown shard '$name'." }
        if ($digests.ContainsKey($name)) { throw "More than one digest names shard '$name'." }
        $digests.Add($name, $digest)
    }

    $known = [System.Collections.Generic.List[string]]::new()
    $always = [System.Collections.Generic.List[string]]::new()
    foreach ($name in $manifest) {
        if ($digests.ContainsKey($name) -and @($digests[$name].files.PSObject.Properties).Count -gt 0) {
            [void] $known.Add($name)
        }
        else {
            # Missing and valid-but-empty digests are both known-unsafe, so they always run.
            [void] $always.Add($name)
        }
    }
    if ($known.Count -eq 0) {
        throw 'No shard has a nonempty valid digest; refusing to publish a map with no index.'
    }

    $knownIndex = [System.Collections.Generic.Dictionary[string, int]]::new([StringComparer]::Ordinal)
    for ($i = 0; $i -lt $known.Count; $i++) { $knownIndex.Add($known[$i], $i) }

    $index = [System.Collections.Generic.Dictionary[string, System.Collections.Generic.List[object]]]::new([StringComparer]::Ordinal)
    foreach ($name in $known) {
        $digest = $digests[$name]
        foreach ($property in @($digest.files.PSObject.Properties | Sort-Object Name)) {
            $path = [string] $property.Name
            if (-not $index.ContainsKey($path)) {
                $index.Add($path, [System.Collections.Generic.List[object]]::new())
            }
            [void] $index[$path].Add([ordered]@{ s = $knownIndex[$name]; r = @($property.Value) })
        }
    }

    $files = [ordered]@{}
    foreach ($path in ($index.Keys | Sort-Object)) { $files[$path] = $index[$path] }
    return [pscustomobject] [ordered]@{
        schemaVersion = 1
        sha           = $Sha
        generatedUtc  = [DateTimeOffset]::UtcNow.ToString('o')
        knownShards   = @($known)
        alwaysRun     = @($always)
        fileCount     = $files.Count
        files         = $files
    }
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True {
        param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) }
    }
    function Assert-Throws {
        param([scriptblock] $Action, [string] $What)
        $threw = $false
        try { & $Action } catch { $threw = $true }
        Assert-True $threw $What
    }
    function Write-DigestFixture {
        param([string] $Path, [string] $Shard, $Files, [int] $SchemaVersion = 1)
        [ordered]@{
            schemaVersion = $SchemaVersion
            shard = $Shard
            fileCount = @($Files.PSObject.Properties).Count
            files = $Files
        } | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath $Path -Encoding utf8
    }

    $temp = Join-Path ([IO.Path]::GetTempPath()) ("AiDotNet-NewShardMap-$([Guid]::NewGuid().ToString('N'))")
    $fixtureSha = '0123456789abcdef0123456789abcdef01234567'
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        Write-DigestFixture -Path (Join-Path $temp 'alpha.digest.json') -Shard 'Alpha' `
            -Files ([pscustomobject] @{ 'src/A.cs' = @(1, 3) })
        Write-DigestFixture -Path (Join-Path $temp 'empty.digest.json') -Shard 'Empty' `
            -Files ([pscustomobject] @{})

        try {
            $map = New-ShardMapObject -DigestDirectory $temp -AllShards @('Alpha', 'Empty', 'Comma, Missing') -Sha $fixtureSha
            Assert-True (@($map.knownShards).Count -eq 1 -and $map.knownShards[0] -eq 'Alpha') `
                'only a shard with indexed files should be known'
            Assert-True ($map.alwaysRun -contains 'Empty') 'a valid empty digest must become always-run'
            Assert-True ($map.alwaysRun -contains 'Comma, Missing') 'a missing digest must become always-run without splitting commas'
            Assert-True (@($map.knownShards).Count + @($map.alwaysRun).Count -eq 3) `
                'every manifest shard must be classified exactly once'
        }
        catch { [void] $failures.Add("valid producer fixture failed: $_") }

        Copy-Item -LiteralPath (Join-Path $temp 'alpha.digest.json') -Destination (Join-Path $temp 'duplicate.digest.json')
        Assert-Throws { New-ShardMapObject -DigestDirectory $temp -AllShards @('Alpha', 'Empty') -Sha $fixtureSha } `
            'duplicate shard digests must be rejected'
        Remove-Item -LiteralPath (Join-Path $temp 'duplicate.digest.json')

        Write-DigestFixture -Path (Join-Path $temp 'schema.digest.json') -Shard 'Alpha' `
            -Files ([pscustomobject] @{ 'src/S.cs' = @(1, 1) }) -SchemaVersion 2
        Assert-Throws { New-ShardMapObject -DigestDirectory $temp -AllShards @('Alpha', 'Empty') -Sha $fixtureSha } `
            'unsupported digest schemas must be rejected before classification'
        Remove-Item -LiteralPath (Join-Path $temp 'schema.digest.json')

        [ordered]@{ schemaVersion = 1; shard = 'Alpha'; files = @(1, 2) } |
            ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $temp 'files.digest.json') -Encoding utf8
        Assert-Throws { New-ShardMapObject -DigestDirectory $temp -AllShards @('Alpha', 'Empty') -Sha $fixtureSha } `
            'a non-object digest files value must be rejected'
        Remove-Item -LiteralPath (Join-Path $temp 'files.digest.json')

        Write-DigestFixture -Path (Join-Path $temp 'bad.digest.json') -Shard 'Unknown' `
            -Files ([pscustomobject] @{ 'src/Bad.cs' = @(1) })
        Assert-Throws { New-ShardMapObject -DigestDirectory $temp -AllShards @('Alpha', 'Empty') -Sha $fixtureSha } `
            'malformed ranges must be rejected even when the shard is unknown'
        Remove-Item -LiteralPath (Join-Path $temp 'bad.digest.json')

        Write-DigestFixture -Path (Join-Path $temp 'unknown.digest.json') -Shard 'Unknown' `
            -Files ([pscustomobject] @{ 'src/U.cs' = @(1, 1) })
        Assert-Throws { New-ShardMapObject -DigestDirectory $temp -AllShards @('Alpha', 'Empty') -Sha $fixtureSha } `
            'unknown shard digests must be rejected'
    }
    finally {
        Remove-Item -LiteralPath $temp -Recurse -Force
    }

    if ($failures.Count -gt 0) {
        Write-Host 'New-ShardMap self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'New-ShardMap self-test passed.'
    exit 0
}

$map = New-ShardMapObject -DigestDirectory $DigestDirectory -AllShards $AllShards -Sha $Sha
$directory = Split-Path -Parent $OutFile
if ($directory -and -not (Test-Path -LiteralPath $directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$map | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 -NoNewline
Write-Host "map: $(@($map.knownShards).Count) indexed shard(s), $($map.fileCount) file(s), $(@($map.alwaysRun).Count) always-run -> $OutFile"
