# Fix missing semicolons after = default!

$ErrorActionPreference = "Stop"

# Get all CS1002 errors from build
$buildOutput = & dotnet build 2>&1 | Out-String
$errors = $buildOutput -split "`n" | Where-Object { $_ -match "CS1002.*; expected" }

# Group errors by file and line number
$errorsByFile = @{}
foreach ($err in $errors) {
    if ($err -match "^(.*?)\((\d+),") {
        $file = $matches[1]
        $line = [int]$matches[2]

        if (-not $errorsByFile.ContainsKey($file)) {
            $errorsByFile[$file] = @()
        }
        if ($errorsByFile[$file] -notcontains $line) {
            $errorsByFile[$file] += $line
        }
    }
}

Write-Host "Found errors in $($errorsByFile.Count) files"

# Process each file
foreach ($file in $errorsByFile.Keys) {
    Write-Host "Processing: $file"
    $lines = Get-Content $file

    # Get unique line numbers sorted in descending order
    $lineNumbers = $errorsByFile[$file] | Sort-Object -Descending

    $modified = $false
    foreach ($lineNum in $lineNumbers) {
        $lineIndex = $lineNum - 1
        $line = $lines[$lineIndex]

        # Check if line ends with = default! but missing semicolon
        if ($line -match '= default!\s*$' -and $line -notmatch ';') {
            $lines[$lineIndex] = $line -replace '(= default!)\s*$', '$1;'
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
& dotnet build 2>&1 | Select-String "CS1002" | Measure-Object | Select-Object -ExpandProperty Count | ForEach-Object {
    Write-Host "Remaining CS1002 errors: $_"
}
