# Get all CS8618 errors
$buildOutput = dotnet build 2>&1 | Select-String "CS8618"

# Group errors by file
$errorsByFile = @{}
foreach ($line in $buildOutput) {
    if ($line -match "^(.*?)\((\d+),\d+\).*'(\w+)'") {
        $file = $matches[1]
        $lineNum = [int]$matches[2]
        $fieldName = $matches[3]
        
        if (-not $errorsByFile.ContainsKey($file)) {
            $errorsByFile[$file] = @()
        }
        $errorsByFile[$file] += @{Line = $lineNum; Field = $fieldName}
    }
}

Write-Host "Processing $($errorsByFile.Count) files..."

foreach ($file in $errorsByFile.Keys) {
    $content = Get-Content $file -Raw
    $lines = $content -split "`r?`n"
    $modified = $false
    
    foreach ($err in $errorsByFile[$file]) {
        $lineIndex = $err.Line - 1
        $fieldName = $err.Field
        
        if ($lineIndex -ge 0 -and $lineIndex -lt $lines.Count) {
            $line = $lines[$lineIndex]
            
            # Fix non-initialized fields: private Type _field;
            if ($line -match '^(\s*)(private|protected|internal|readonly|public)\s+.*\s+' + [regex]::Escape($fieldName) + '\s*;') {
                $lines[$lineIndex] = $line -replace ';$', ' = default!;'
                $modified = $true
                Write-Host "Fixed field $fieldName in $file"
            }
            # Fix properties without initialization
            elseif ($line -match '\{\s*get;\s*set;\s*\}\s*$' -and $line -notmatch '= default!') {
                $lines[$lineIndex] = $line -replace '\{\s*get;\s*set;\s*\}', '{ get; set; } = default!;'
                $modified = $true
                Write-Host "Fixed property $fieldName in $file"
            }
        }
    }
    
    if ($modified) {
        $newContent = $lines -join "`r`n"
        Set-Content -Path $file -Value $newContent -NoNewline
        Write-Host "Modified: $file"
    }
}

Write-Host "Done!"
