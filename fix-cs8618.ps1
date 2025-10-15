# PowerShell script to fix CS8618 errors by adding = default! to non-nullable properties

$ErrorActionPreference = "Stop"

# Get all CS8618 errors from build
$buildOutput = & dotnet build 2>&1 | Out-String
$errors = $buildOutput -split "`n" | Where-Object { $_ -match "CS8618" }

# Group errors by file and line number
$errorsByFile = @{}
foreach ($err in $errors) {
    if ($err -match "^(.*?)\((\d+),\d+\).*property '(\w+)'") {
        $file = $matches[1]
        $line = [int]$matches[2]
        $property = $matches[3]

        if (-not $errorsByFile.ContainsKey($file)) {
            $errorsByFile[$file] = @{}
        }
        if (-not $errorsByFile[$file].ContainsKey($line)) {
            $errorsByFile[$file][$line] = $property
        }
    }
}

Write-Host "Found errors in $($errorsByFile.Count) files"

# Process each file
foreach ($file in $errorsByFile.Keys) {
    Write-Host "Processing: $file"
    $content = Get-Content $file -Raw
    $lines = Get-Content $file

    # Get unique line numbers sorted in descending order (to avoid line number shifts)
    $lineNumbers = $errorsByFile[$file].Keys | Sort-Object -Descending

    $modified = $false
    foreach ($lineNum in $lineNumbers) {
        $lineIndex = $lineNum - 1
        $line = $lines[$lineIndex]

        # Check if line matches property pattern without initialization
        if ($line -match '^(\s*public\s+\S+\s+\w+\s*\{\s*get;\s*set;\s*\})\s*$') {
            # Add = default! before the semicolon
            $lines[$lineIndex] = $line -replace '\{\s*get;\s*set;\s*\}', '{ get; set; } = default!'
            $modified = $true
            Write-Host "  Fixed line $lineNum"
        }
        elseif ($line -match '^(\s*)(private|protected|internal)?\s*(readonly\s+)?\S+\s+_\w+;') {
            # This is a field, add = default!
            $lines[$lineIndex] = $line -replace ';$', ' = default!;'
            $modified = $true
            Write-Host "  Fixed field line $lineNum"
        }
    }

    if ($modified) {
        $lines | Set-Content $file
        Write-Host "  Saved changes to $file"
    }
}

Write-Host "`nDone! Running build to check..."
& dotnet build 2>&1 | Select-String "CS8618" | Measure-Object | Select-Object -ExpandProperty Count | ForEach-Object {
    Write-Host "Remaining CS8618 errors: $_"
}
