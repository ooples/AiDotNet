<#
.SYNOPSIS
    Puts a lone pattern-matched artifact back into its own folder after actions/download-artifact.

.DESCRIPTION
    With `pattern:` and `merge-multiple: false`, actions/download-artifact gives each artifact its
    own folder under `path`, except when the pattern matches exactly one artifact. Then it extracts
    that artifact into `path` itself. Every consumer here looks for per-artifact folders named
    '<prefix>-<sha>-<slug>', so a run with one shard, one primary attempt or one paper slice recorded
    its only result as Missing (run 45309453002). Imports added later beside the flat files made the
    layout mixed, and the flat files were then ignored.

    When the root holds content but no folder that the pattern matches, this script asks the GitHub
    API which artifacts of the run match the pattern. If exactly one does, the root's content moves
    into a folder with that artifact's exact name. It never invents an identity: zero or several
    matching artifacts with flat content is an error, and a root that is empty, absent, or already
    laid out per artifact is left as it is. Consumers still bind each folder to a planned shard and
    keep their own completeness checks.
#>
[CmdletBinding(DefaultParameterSetName = 'Restore')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Restore')] [string] $Root,
    [Parameter(Mandatory, ParameterSetName = 'Restore')] [string] $Pattern,
    [Parameter(Mandatory, ParameterSetName = 'Restore')] [string] $Repository,
    [Parameter(Mandatory, ParameterSetName = 'Restore')] [string] $RunId,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-ArtifactPatternRegex {
    param([Parameter(Mandatory)] [string] $Pattern)
    # The workflow's artifact patterns use only '*' and '?'.
    return '^' + ([Regex]::Escape($Pattern) -replace '\\\*', '.*' -replace '\\\?', '.') + '$'
}

function Test-FlattenedArtifactRoot {
    param([Parameter(Mandatory)] [string] $Root, [Parameter(Mandatory)] [string] $Pattern)
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { return $false }
    $entries = @(Get-ChildItem -LiteralPath $Root -Force)
    if ($entries.Count -eq 0) { return $false }
    $regex = ConvertTo-ArtifactPatternRegex $Pattern
    return @($entries | Where-Object { $_.PSIsContainer -and $_.Name -cmatch $regex }).Count -eq 0
}

function Restore-FlattenedArtifactFolder {
    param(
        [Parameter(Mandatory)] [string] $Root,
        [Parameter(Mandatory)] [string] $Pattern,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $RunArtifactNames
    )
    if (-not (Test-FlattenedArtifactRoot -Root $Root -Pattern $Pattern)) { return $null }
    $regex = ConvertTo-ArtifactPatternRegex $Pattern
    $matched = @($RunArtifactNames | Where-Object { $_ -cmatch $regex } | Sort-Object -Unique -CaseSensitive)
    if ($matched.Count -ne 1) {
        throw "'$Root' holds one artifact's files with no artifact folder, but pattern '$Pattern' matches $($matched.Count) artifact(s) of the run: $($matched -join ', '). Refusing to guess its identity."
    }
    $name = [string] $matched[0]
    if ($name -match '[\\/]' -or $name -in @('.', '..')) { throw "artifact name '$name' is not a single folder name" }
    $staging = Join-Path $Root ('.restore-' + [guid]::NewGuid().ToString('N'))
    $entries = @(Get-ChildItem -LiteralPath $Root -Force)
    New-Item -ItemType Directory -Path $staging | Out-Null
    foreach ($entry in $entries) { Move-Item -LiteralPath $entry.FullName -Destination $staging }
    Rename-Item -LiteralPath $staging -NewName $name
    return $name
}

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True { param([bool] $Condition, [string] $What) if (-not $Condition) { [void] $failures.Add($What) } }
    function Put { param([string] $Path) New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null; 'x' | Set-Content -LiteralPath $Path }
    $sha = '0123456789abcdef0123456789abcdef01234567'
    $pattern = "coverage-$sha-*"
    $names = @("coverage-$sha-Unit_10_RL", "test-results-$sha-Unit_10_RL", "coverage-$('f' * 40)-Other")
    $scratch = Join-Path ([IO.Path]::GetTempPath()) ('restore-artifact-folders-' + [guid]::NewGuid().ToString('N'))
    try {
        # A lone match extracted into the root moves into a folder named after that artifact.
        $flat = Join-Path $scratch 'flat'
        Put (Join-Path $flat 'shard-metadata.json'); Put (Join-Path $flat 'Unit_10_RL/test-results.trx'); Put (Join-Path $flat '.hidden')
        $restored = Restore-FlattenedArtifactFolder -Root $flat -Pattern $pattern -RunArtifactNames $names
        Assert-True ($restored -ceq "coverage-$sha-Unit_10_RL") 'a lone flattened artifact was not bound to its exact name'
        Assert-True ((Test-Path (Join-Path $flat "coverage-$sha-Unit_10_RL/shard-metadata.json")) -and
            (Test-Path (Join-Path $flat "coverage-$sha-Unit_10_RL/Unit_10_RL/test-results.trx")) -and
            (Test-Path (Join-Path $flat "coverage-$sha-Unit_10_RL/.hidden")) -and
            @(Get-ChildItem -LiteralPath $flat -Force).Count -eq 1) 'the flattened artifact was not moved whole into its folder'
        Assert-True ($null -eq (Restore-FlattenedArtifactFolder -Root $flat -Pattern $pattern -RunArtifactNames $names)) 'a restored root was restored twice'

        # Per-artifact layouts, empty roots and absent roots are left alone.
        $nested = Join-Path $scratch 'nested'
        Put (Join-Path $nested "coverage-$sha-A/shard-metadata.json"); Put (Join-Path $nested "coverage-$sha-B/shard-metadata.json")
        Assert-True ($null -eq (Restore-FlattenedArtifactFolder -Root $nested -Pattern $pattern -RunArtifactNames @())) 'a per-artifact layout was rewritten'
        Assert-True (@(Get-ChildItem -LiteralPath $nested).Count -eq 2) 'a per-artifact layout lost a folder'
        $empty = Join-Path $scratch 'empty'; New-Item -ItemType Directory -Path $empty | Out-Null
        Assert-True ($null -eq (Restore-FlattenedArtifactFolder -Root $empty -Pattern $pattern -RunArtifactNames $names)) 'an empty root was restored'
        Assert-True ($null -eq (Restore-FlattenedArtifactFolder -Root (Join-Path $scratch 'absent') -Pattern $pattern -RunArtifactNames $names)) 'an absent root was restored'

        # Flat content with zero or several matching artifacts has no provable identity.
        foreach ($case in @(@{ Label = 'zero'; Names = @("coverage-$('f' * 40)-Other") },
                @{ Label = 'several'; Names = @("coverage-$sha-A", "coverage-$sha-B") })) {
            $root = Join-Path $scratch "ambiguous-$($case.Label)"; Put (Join-Path $root 'shard-metadata.json')
            $threw = $false
            try { Restore-FlattenedArtifactFolder -Root $root -Pattern $pattern -RunArtifactNames $case.Names | Out-Null } catch { $threw = $true }
            Assert-True $threw "flat content with $($case.Label) matching artifacts was given an identity"
            Assert-True (Test-Path (Join-Path $root 'shard-metadata.json')) "flat content with $($case.Label) matching artifacts was moved"
        }

        # The pattern's literal parts must match exactly: '-1' must not match attempt '-11'.
        $attemptPattern = "shard-attempt-primary-$sha-*-1"
        Assert-True ("shard-attempt-primary-$sha-Unit-11" -cnotmatch (ConvertTo-ArtifactPatternRegex $attemptPattern)) 'the pattern matched past its literal suffix'
        Assert-True ("shard-attempt-primary-$sha-Unit_A-1" -cmatch (ConvertTo-ArtifactPatternRegex $attemptPattern)) 'the pattern rejected a matching artifact name'
    }
    finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
    if ($failures.Count -gt 0) {
        Write-Host 'Restore-ArtifactFolders self-test FAILED:'
        foreach ($f in $failures) { Write-Host "  - $f" }
        exit 1
    }
    Write-Host 'Restore-ArtifactFolders self-test passed.'
    exit 0
}

if (-not (Test-FlattenedArtifactRoot -Root $Root -Pattern $Pattern)) {
    Write-Host "'$Root' is empty, absent or laid out one folder per artifact; nothing to restore."
    exit 0
}
. "$PSScriptRoot/RunnerFailover.ps1"
$artifacts = @(Invoke-FailoverApi -Path "repos/$Repository/actions/runs/$RunId/artifacts" -Paginate)
$names = @($artifacts | Where-Object { -not $_.expired } | ForEach-Object { [string] $_.name })
$restored = Restore-FlattenedArtifactFolder -Root $Root -Pattern $Pattern -RunArtifactNames $names
Write-Host "Moved the lone artifact downloaded into '$Root' into its own folder '$restored'."
exit 0
