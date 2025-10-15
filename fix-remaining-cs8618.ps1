# Fix remaining CS8618 errors for fields

$ErrorActionPreference = "Stop"

# Get all CS8618 errors from build (unique by file and line)
$buildOutput = & dotnet build 2>&1 | Out-String
$errors = $buildOutput -split "`n" | Where-Object { $_ -match "CS8618" }

# Extract unique errors by file and line
$errorsByFile = @{}
foreach ($err in $errors) {
    if ($err -match "^(.*?)\((\d+),\d+\).*'(\w+)'") {
        $file = $matches[1]
        $line = [int]$matches[2]
        $name = $matches[3]

        if (-not $errorsByFile.ContainsKey($file)) {
            $errorsByFile[$file] = @{}
        }
        if (-not $errorsByFile[$file].ContainsKey($line)) {
            $errorsByFile[$file][$line] = $name
        }
    }
}

Write-Host "Found errors in $($errorsByFile.Count) files"

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

        # Fix fields: private/protected/internal Type _field;
        if ($line -match '^(\s*)(private|protected|internal|public)?\s*(readonly\s+)?(\S+)\s+(_\w+);') {
            $lines[$lineIndex] = $line -replace ';$', ' = default!;'
            $modified = $true
            Write-Host "  Fixed field line $lineNum"
        }
        # Fix fields without semicolon at end
        elseif ($line -match '^(\s*)(private|protected|internal|public)?\s*(readonly\s+)?(\S+)\s+(_\w+)$') {
            $lines[$lineIndex] = "$line = default!;"
            $modified = $true
            Write-Host "  Fixed field line $lineNum (no semicolon)"
        }
        # Fix Dictionary/List properties without initialization
        elseif ($line -match '\{\s*get;\s*set;\s*\}\s*$' -and $line -notmatch '= default!') {
            $lines[$lineIndex] = $line -replace '\{\s*get;\s*set;\s*\}', '{ get; set; } = default!'
            $modified = $true
            Write-Host "  Fixed property line $lineNum"
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
