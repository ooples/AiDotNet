# Fix missing semicolons after = default! using a more direct approach

$ErrorActionPreference = "Stop"

# Get list of files that have CS1002 errors
$buildOutput = & dotnet build 2>&1 | Out-String
$errors = $buildOutput -split "`n" | Where-Object { $_ -match "CS1002" }

# Extract unique file paths
$files = @{}
foreach ($err in $errors) {
    if ($err -match "^(.*?)\(") {
        $file = $matches[1]
        if (-not $files.ContainsKey($file)) {
            $files[$file] = $true
        }
    }
}

Write-Host "Found $($files.Count) files with CS1002 errors"

# Process each file
foreach ($file in $files.Keys) {
    Write-Host "Processing: $file"
    $content = Get-Content $file -Raw

    # Replace = default! followed by end of line (not ;) with = default!;
    $newContent = $content -replace '= default!(\s*\r?\n)', '= default!;$1'

    if ($content -ne $newContent) {
        $newContent | Set-Content $file -NoNewline
        Write-Host "  Fixed $file"
    }
}

Write-Host "`nDone! Running build to check..."
$remaining = & dotnet build 2>&1 | Select-String "CS1002" | Measure-Object | Select-Object -ExpandProperty Count
Write-Host "Remaining CS1002 errors: $remaining"
