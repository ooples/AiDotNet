<#
.SYNOPSIS
    Proves Select-AffectedTests leaves out of TypeImpact's input the paths the shard selector classifies as unable to
    affect a test, and keeps every other path.

.DESCRIPTION
    TypeImpact cannot map a non-C# file, so a single README, .gitignore or unreferenced tool file in a change used to
    pass the whole test-level decision through (CI artifacts from late September 2026: 31 of 77 unresolved entries
    were such files). This runs the real script, with the real Select-Shards.ps1 classifier copied into a throwaway
    repository, on a change to README.md, .editorconfig and a C# source, and asserts that the change list handed to
    TypeImpact holds only the C# source. A second run without the classifier proves a classification failure removes
    nothing.
#>
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script = (Resolve-Path (Join-Path $PSScriptRoot 'Select-AffectedTests.ps1')).Path
$classifier = (Resolve-Path (Join-Path $PSScriptRoot 'Select-Shards.ps1')).Path
$saved = @{}
$names = 'GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY', 'GITHUB_EVENT_NAME', 'REQUIRES_SHARDS', 'DELTA_MODE',
    'COLLECT_COVERAGE_EVERYWHERE', 'SELECTED_MATRIX', 'SELECTED_LEDGER_MATRIX', 'SELECTED_SKIPPED', 'TMP', 'TEMP', 'TMPDIR'
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name) }

function Invoke-Fixture([bool] $WithClassifier) {
    $root = Join-Path ([IO.Path]::GetTempPath()) ("select-affected-inert-" + [guid]::NewGuid().ToString('N'))
    $repo = Join-Path $root 'repo'
    $tmp = Join-Path $root 'tmp'
    New-Item -ItemType Directory -Path $repo, $tmp -Force | Out-Null
    try {
        Push-Location $repo
        try {
            & git init -q
            & git config user.email 'fixture@example.invalid'
            & git config user.name 'fixture'
            New-Item -ItemType Directory -Path '.github', 'tools/TestImpact/TypeImpact', 'src' -Force | Out-Null
            Set-Content -LiteralPath '.github/test-shards.yml' -Value 'shard: []'
            # The self-test fails, so the script passes through right after writing TypeImpact's inputs.
            Set-Content -LiteralPath 'tools/TestImpact/TypeImpact/Test-TypeImpact.ps1' -Value 'exit 3'
            if ($WithClassifier) { Copy-Item -LiteralPath $classifier -Destination 'tools/TestImpact/Select-Shards.ps1' }
            Set-Content -LiteralPath 'README.md' -Value 'one'
            Set-Content -LiteralPath '.editorconfig' -Value 'root = true'
            Set-Content -LiteralPath 'src/Model.cs' -Value 'class Model { }'
            & git add -A; & git commit -q -m base
            Set-Content -LiteralPath 'README.md' -Value 'two'
            Set-Content -LiteralPath '.editorconfig' -Value "root = true`nindent_size = 4"
            Set-Content -LiteralPath 'src/Model.cs' -Value 'class Model { int x; }'
            & git commit -q -am change
            if ($LASTEXITCODE -ne 0) { throw 'fixture repository setup failed' }
        }
        finally { Pop-Location }

        $env:GITHUB_OUTPUT = Join-Path $root 'github-output.txt'
        $env:GITHUB_STEP_SUMMARY = Join-Path $root 'summary.md'
        $env:GITHUB_EVENT_NAME = 'push'
        $env:REQUIRES_SHARDS = 'true'
        $env:DELTA_MODE = ''
        $env:COLLECT_COVERAGE_EVERYWHERE = ''
        $env:SELECTED_MATRIX = '{"include":[]}'
        $env:SELECTED_LEDGER_MATRIX = '{"include":[]}'
        $env:SELECTED_SKIPPED = '[]'
        $env:TMP = $tmp; $env:TEMP = $tmp; $env:TMPDIR = $tmp

        $command = "function yq { '[{""name"":""x""}]' }; Set-Location '$repo'; & '$script' -Repository '$repo'; " +
            'if ((Test-Path -LiteralPath variable:\LASTEXITCODE)) { exit $LASTEXITCODE }'
        & pwsh -NoProfile -Command $command | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "the step failed (exit $LASTEXITCODE)" }
        $changes = Join-Path $tmp 'type-impact-changes.txt'
        if (-not (Test-Path -LiteralPath $changes)) { throw 'type-impact-changes.txt was not written' }
        return @(Get-Content -LiteralPath $changes | ForEach-Object { ([string] $_).Split("`t")[-1] })
    }
    finally {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
    }
}

try {
    $paths = Invoke-Fixture -WithClassifier $true
    if (($paths -join ',') -cne 'src/Model.cs') {
        throw "TypeImpact's input should hold only src/Model.cs once README.md and .editorconfig are classified; got: $($paths -join ', ')"
    }
    $unclassified = Invoke-Fixture -WithClassifier $false
    if ((@($unclassified | Sort-Object) -join ',') -cne '.editorconfig,README.md,src/Model.cs') {
        throw "without the classifier every changed path must stay a TypeImpact input; got: $($unclassified -join ', ')"
    }
    Write-Host 'Select-AffectedTests leaves out paths that cannot affect a test, and only when classification succeeds.'
}
finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name]) }
}
