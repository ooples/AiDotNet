# Fix CS8618 errors for fields initialized in helper methods

$ErrorActionPreference = "Stop"

# Get all CS8618 field errors
$buildOutput = & dotnet build 2>&1 | Out-String
$errors = $buildOutput -split "`n" | Where-Object { $_ -match "CS8618.*field" }

# Extract unique errors
$errorsByFile = @{}
foreach ($err in $errors) {
    if ($err -match "^(.*?)\((\d+),\d+\).*field '(\w+)'") {
        $file = $matches[1]
        $line = [int]$matches[2]
        $field = $matches[3]

        if (-not $errorsByFile.ContainsKey($file)) {
            $errorsByFile[$file] = @{}
        }
        if (-not $errorsByFile[$file].ContainsKey($line)) {
            $errorsByFile[$file][$line] = $field
        }
    }
}

Write-Host "Found field errors in $($errorsByFile.Count) files"

# Process each file
foreach ($file in $errorsByFile.Keys) {
    Write-Host "Processing: $file"
    $lines = Get-Content $file

    # Get unique line numbers sorted in descending order
    $lineNumbers = $errorsByFile[$file].Keys | Sort-Object -Descending

    $modified = $false
    foreach ($lineNum in $lineNumbers) {
        $lineIndex = $lineNum - 1
        $line = $lines[$lineIndex]

        # Fix fields without initialization
        if ($line -match '^(\s*)(private|protected|internal)?\s*(readonly\s+)?(\S+)\s+(_\w+);$') {
            $lines[$lineIndex] = $line -replace ';$', ' = default!;'
            $modified = $true
            Write-Host "  Fixed line $lineNum"
        }
    }

    if ($modified) {
        $lines | Set-Content $file
        Write-Host "  Saved changes to $file"
    }
}

Write-Host "`nDone! Running build to check..."
$remaining = & dotnet build 2>&1 | Select-String "CS8618" | Measure-Object | Select-Object -ExpandProperty Count
Write-Host "Remaining CS8618 errors: $remaining"
