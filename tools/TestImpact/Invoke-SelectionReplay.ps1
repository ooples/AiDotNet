<#
.SYNOPSIS
    Replays shard selection for past and open pull requests, so a selector change can be judged on real changes
    before it ships.
.DESCRIPTION
    For each pull request the merge CI would have validated is rebuilt in a scratch worktree: the pull request head
    merged onto its base (the base branch now for an open pull request; the first parent of the commit GitHub
    recorded as its merge commit, for a merged one). Pull requests into another branch, and closed ones, are
    skipped. Select-Shards.ps1 then runs exactly as the Select shards job runs it - against the
    given map, with the manifest converted from that merge's .github/test-shards.yml - using the selection tools
    from -ToolsRef. Comparing two runs that differ only in -ToolsRef shows what a selector change does.

    A pull request whose head no longer merges cleanly onto its base is reported as such and skipped; the replay
    never guesses a resolution.
.PARAMETER PullRequest
    Pull request numbers to replay. Comma-separated values are accepted too, because pwsh -File passes an array
    argument as one string.
.PARAMETER MapFile
    The shard map to select against (shard-map.json, for example from Resolve-CertifiedShardMap.ps1).
.PARAMETER ToolsRef
    The commit whose tools/TestImpact is used. Defaults to HEAD of the repository the script runs from.
.PARAMETER WorkTree
    A scratch directory for the replay worktree. It is created if missing and reused across calls; it must not be
    a worktree you use for anything else, because each replay checks out a detached merge there.
.PARAMETER OutDirectory
    Receives selection-<pr>.json per pull request and replay-summary.json.
.PARAMETER Remote
    The remote holding refs/pull/*. Defaults to origin.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string[]] $PullRequest,
    [Parameter(Mandatory)] [string] $MapFile,
    [string] $ToolsRef = 'HEAD',
    [Parameter(Mandatory)] [string] $WorkTree,
    [Parameter(Mandatory)] [string] $OutDirectory,
    [string] $Remote = 'origin',
    [string] $Repository = 'ooples/AiDotNet',
    [string] $BaseBranch = 'master'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Git {
    param([Parameter(Mandatory)] [string[]] $Arguments, [string] $In)
    $output = if ($In) { & git -C $In @Arguments 2>&1 } else { & git @Arguments 2>&1 }
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed: $($output -join "`n")" }
    return @($output | ForEach-Object { "$_" })
}

# The selector reads the shard manifest with yq, as the workflow does.
if (-not (Get-Command yq -ErrorAction SilentlyContinue)) { throw 'yq (https://github.com/mikefarah/yq) must be on PATH' }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw 'gh (the GitHub CLI) must be on PATH and signed in' }
$numbers = @($PullRequest | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ } | ForEach-Object { [int] $_ })
$MapFile = (Resolve-Path -LiteralPath $MapFile).Path
New-Item -ItemType Directory -Force -Path $OutDirectory | Out-Null
$OutDirectory = (Resolve-Path -LiteralPath $OutDirectory).Path
$toolsSha = @(Invoke-Git @('rev-parse', "$ToolsRef^{commit}"))[0]

# The selection tools come from ToolsRef, extracted beside the worktree so a replayed merge's own tools/TestImpact
# (whatever that pull request changed) never decides its own selection.
$toolsDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "selection-replay-tools-$($toolsSha.Substring(0, 12))"
if (-not (Test-Path -LiteralPath (Join-Path $toolsDirectory 'tools/TestImpact/Select-Shards.ps1'))) {
    New-Item -ItemType Directory -Force -Path $toolsDirectory | Out-Null
    $archive = Join-Path $toolsDirectory 'tools.zip'
    Invoke-Git @('archive', '--format=zip', "--output=$archive", $toolsSha, 'tools/TestImpact') | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $toolsDirectory -Force
    Remove-Item -LiteralPath $archive
}
$selector = Join-Path $toolsDirectory 'tools/TestImpact/Select-Shards.ps1'

if (-not (Test-Path -LiteralPath (Join-Path $WorkTree '.git'))) {
    Invoke-Git @('worktree', 'add', '--detach', $WorkTree, 'HEAD') | Out-Null
}
$WorkTree = (Resolve-Path -LiteralPath $WorkTree).Path

Invoke-Git @('fetch', '--quiet', $Remote, "+refs/heads/${BaseBranch}:refs/replay/base") | Out-Null

$summary = [System.Collections.Generic.List[object]]::new()
foreach ($number in $numbers) {
    $record = [ordered]@{ pullRequest = $number; status = $null; base = $null; head = $null; merge = $null }
    try {
        Invoke-Git @('fetch', '--quiet', $Remote, "+refs/pull/$number/head:refs/replay/pr/$number") | Out-Null
        $head = @(Invoke-Git @('rev-parse', "refs/replay/pr/$number"))[0]
        $record.head = $head

        # GitHub says whether the pull request merged, into which branch, and as which commit. A pull request into
        # another branch is not part of this base branch's history and is skipped rather than replayed against it.
        $described = @(& gh pr view $number --repo $Repository --json state,baseRefName,mergeCommit 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "gh could not describe pull request #${number}: $($described -join ' ')" }
        $info = ($described -join "`n") | ConvertFrom-Json
        if ($info.baseRefName -cne $BaseBranch) {
            $record.status = 'other-base'
            $record.error = "targets $($info.baseRefName), not $BaseBranch"
            $summary.Add([pscustomobject] $record)
            Write-Host "#${number}: $($record.error); skipped"
            continue
        }
        if ($info.state -eq 'MERGED') {
            # The commit the pull request became on the base branch: a merge commit or a squash. Its first parent is
            # the base the merge was validated against.
            $landed = [string] $info.mergeCommit.oid
            Invoke-Git @('fetch', '--quiet', $Remote, $landed) | Out-Null
            $base = @(Invoke-Git @('rev-parse', "$landed^1"))[0]
            $record.state = 'merged'
        }
        elseif ($info.state -eq 'OPEN') {
            $base = @(Invoke-Git @('rev-parse', 'refs/replay/base'))[0]
            $record.state = 'open'
        }
        else {
            $record.status = 'closed'
            $summary.Add([pscustomobject] $record)
            Write-Host "#${number}: closed without merging; skipped"
            continue
        }
        $record.base = $base

        Invoke-Git @('checkout', '--quiet', '--force', '--detach', $base) -In $WorkTree | Out-Null
        $merge = & git -C $WorkTree -c user.name=replay -c user.email=replay@localhost merge --no-ff --no-edit --quiet $head 2>&1
        if ($LASTEXITCODE -ne 0) {
            & git -C $WorkTree merge --abort 2>$null
            $record.status = 'conflict'
            $summary.Add([pscustomobject] $record)
            Write-Host "#${number}: head does not merge cleanly onto $($base.Substring(0, 8)); skipped"
            continue
        }
        $record.merge = @(Invoke-Git @('rev-parse', 'HEAD') -In $WorkTree)[0]

        $manifestFile = Join-Path $OutDirectory "manifest-$number.json"
        $shardsYaml = Join-Path $WorkTree '.github/test-shards.yml'
        # The same conversion the Select shards job makes.
        $all = @(& yq -o=json -I=0 '.shard' $shardsYaml | ConvertFrom-Json)
        if ($LASTEXITCODE -ne 0 -or $all.Count -eq 0) { throw "could not convert $shardsYaml" }
        ConvertTo-Json -InputObject @($all) -Depth 5 | Set-Content -LiteralPath $manifestFile -Encoding utf8
        $names = @($all | ForEach-Object { $_.name })

        $selection = Join-Path $OutDirectory "selection-$number.json"
        Push-Location $WorkTree
        try {
            # A script block, not -File: -File flattens the shard-name array into one string, and names hold spaces.
            & pwsh -NoProfile -Command {
                param($Selector, $Map, $Head, $Manifest, $Names, $Out)
                & $Selector -MapFile $Map -PullRequestHeadSha $Head -ShardManifestFile $Manifest -ExpectedShards $Names -OutFile $Out
                exit $LASTEXITCODE
            } -args $selector, $MapFile, $head, $manifestFile, $names, $selection *> (Join-Path $OutDirectory "selection-$number.log")
            $exit = $LASTEXITCODE
        }
        finally { Pop-Location }
        if ($exit -ne 0) { throw "selector exited $exit (see selection-$number.log)" }

        $result = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
        $wouldSelect = $result.PSObject.Properties['wouldSelect']
        $record.status = 'selected'
        $record.escalate = [bool] $result.escalate
        $record.shards = @($result.shards).Count
        $record.manifestShards = $names.Count
        $record.wouldSelect = if ($wouldSelect -and $null -ne $wouldSelect.Value) { @($wouldSelect.Value).Count } else { $null }
        $record.reasons = @($result.reasons | Where-Object { $_ -match 'escalat|full matrix' } | Select-Object -First 20)
        Write-Host ("#{0} ({1}): {2} of {3} shards{4}" -f $number, $record.state, $record.shards, $names.Count,
            $(if ($record.escalate) { " - ESCALATED (would select $($record.wouldSelect))" } else { '' }))
    }
    catch {
        $record.status = 'error'
        $record.error = $_.Exception.Message
        Write-Host "#${number}: $($_.Exception.Message)"
    }
    $summary.Add([pscustomobject] $record)
}

[ordered]@{ toolsRef = $toolsSha; map = $MapFile; pullRequests = $summary } |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutDirectory 'replay-summary.json') -Encoding utf8
