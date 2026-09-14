<#
.SYNOPSIS
    Resumes only explicitly deferred push validation, on its original SHA.
.DESCRIPTION
    Runs from trusted default-branch code, never PR artifacts or code. Both PR completion
    and deferred-push completion trigger reconciliation, closing the completion-before-deferral
    race. The schedule is a recovery path for missed events, not a blanket retry of failures.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, ParameterSetName = 'Resume')] [string] $Repository,
    [Parameter(ParameterSetName = 'Resume')] [long] $RunId = 0,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-CiApi {
    param([string[]] $Arguments)
    $response = @(& gh api @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "GitHub API failed: $($Arguments -join ' ')" }
    if ($response.Count -gt 0) { return ($response -join "`n" | ConvertFrom-Json) }
}

function Invoke-DeferredReconciliation {
    param([object[]] $Candidates)
    foreach ($candidate in $Candidates) {
        $id = [long] $candidate.id
        # Re-read: a completion event or scheduled sweep may have raced another reconciler.
        $run = Invoke-CiApi @("repos/$Repository/actions/runs/$id")
        if ($run.name -cne 'Build & SonarCloud' -or $run.event -cne 'push' -or
            $run.status -cne 'completed' -or $run.conclusion -cne 'failure') { continue }
        $branch = [string] $run.head_branch
        $sha = [string] $run.head_sha
        if (($branch -cne 'master' -and $branch -cne 'main' -and
            -not $branch.StartsWith('ci-proof/', [StringComparison]::Ordinal)) -or
            $sha -cnotmatch '^[0-9a-f]{40}$') { continue }
        # Never resurrect superseded master runs and cancel the newer run through concurrency.
        $ref = Invoke-CiApi @("repos/$Repository/git/ref/heads/$([uri]::EscapeDataString($branch))")
        if ([string] $ref.object.sha -cne $sha) { continue }
        $pages = @(Invoke-CiApi @('--paginate', '--slurp', "repos/$Repository/actions/runs/$id/jobs?filter=latest&per_page=100"))
        $deferred = @($pages | ForEach-Object { $_.jobs } | Where-Object {
            $_.name -ceq 'Resolve certified PR validation' -and $_.conclusion -ceq 'failure' -and
            @($_.steps | Where-Object {
                $_.name -ceq 'Defer until PR validation completes' -and $_.conclusion -ceq 'failure'
            }).Count -eq 1
        })
        if ($deferred.Count -ne 1) { continue }
        $pulls = @(Invoke-CiApi @("repos/$Repository/commits/$sha/pulls"))
        $merged = @($pulls | Where-Object {
            $_.merged_at -and [string] $_.merge_commit_sha -ceq $sha -and
            [string] $_.base.ref -ceq $branch
        })
        if ($merged.Count -ne 1) { continue }
        $head = [string] $merged[0].head.sha
        if ($head -cnotmatch '^[0-9a-f]{40}$') { throw 'Associated PR head is invalid.' }
        $prPages = @(Invoke-CiApi @('--paginate', '--slurp',
            "repos/$Repository/actions/workflows/sonarcloud.yml/runs?event=pull_request&head_sha=$head&per_page=100"))
        $prRuns = @($prPages | ForEach-Object { $_.workflow_runs } | Where-Object {
            $_.event -ceq 'pull_request' -and $_.head_sha -ceq $head
        })
        if ($prRuns.Count -eq 0 -or @($prRuns | Where-Object { $_.status -cne 'completed' }).Count -gt 0) {
            continue
        }
        # Failed/cancelled PRs resume too: the resolver must explicitly reject missing evidence
        # and validate, not leave the landed commit parked forever. No evidence is minted here.
        $jobId = [long] $deferred[0].id
        Invoke-CiApi @('--method', 'POST', "repos/$Repository/actions/jobs/$jobId/rerun") | Out-Null
        Write-Host "Resumed deferred run $id ($sha), resolver job $jobId and its dependents."
    }
}

if ($SelfTest) {
    $Repository = 'fixture/repo'
    $sha = '0123456789abcdef0123456789abcdef01234567'
    $head = '1123456789abcdef0123456789abcdef01234567'
    $cases = @('ready', 'pending', 'missing-pr-run', 'superseded', 'ordinary-failure',
        'cancelled-push', 'already-running', 'not-merged', 'wrong-branch', 'wrong-event',
        'failed-pr', 'cancelled-pr', 'wrong-pr-head', 'duplicate-marker', 'unrelated-pending')
    foreach ($case in $cases) {
        $script:posts = [System.Collections.Generic.List[string]]::new()
        function Invoke-CiApi {
            param([string[]] $Arguments)
            $path = $Arguments[-1]
            if ($Arguments -contains 'POST') { $script:posts.Add($path); return }
            if ($path -like '*/actions/runs/42') {
                return [pscustomobject]@{ name = 'Build & SonarCloud'; event = $(if ($case -eq 'wrong-event') { 'pull_request' } else { 'push' })
                    status = $(if ($case -eq 'already-running') { 'in_progress' } else { 'completed' })
                    conclusion = $(if ($case -eq 'cancelled-push') { 'cancelled' } else { 'failure' })
                    head_branch = $(if ($case -eq 'wrong-branch') { 'feature' } else { 'master' }); head_sha = $sha }
            }
            if ($path -like '*/git/ref/*') {
                return [pscustomobject]@{ object = @{ sha = $(if ($case -eq 'superseded') { $head } else { $sha }) } }
            }
            if ($path -like '*/jobs?*') {
                $step = @{ name = $(if ($case -eq 'ordinary-failure') { 'Checkout' } else { 'Defer until PR validation completes' }); conclusion = 'failure' }
                $steps = @($step)
                if ($case -eq 'duplicate-marker') { $steps += $step }
                return [pscustomobject]@{ jobs = @(@{ id = 99; name = 'Resolve certified PR validation'; conclusion = 'failure'; steps = $steps }) }
            }
            if ($path -like '*/pulls') {
                return [pscustomobject]@{ merged_at = $(if ($case -eq 'not-merged') { $null } else { '2026-09-14T00:00:00Z' })
                    merge_commit_sha = $sha; base = @{ ref = 'master' }; head = @{ sha = $head } }
            }
            if ($path -like '*/runs?*') {
                $runs = @(@{ event = 'pull_request'; head_sha = $(if ($case -eq 'wrong-pr-head') { $sha } else { $head })
                    status = $(if ($case -eq 'pending') { 'queued' } else { 'completed' })
                    conclusion = $(if ($case -eq 'failed-pr') { 'failure' } elseif ($case -eq 'cancelled-pr') { 'cancelled' } else { 'success' }) })
                if ($case -eq 'missing-pr-run') { $runs = @() }
                if ($case -eq 'unrelated-pending') { $runs += @{ event = 'pull_request'; head_sha = $sha; status = 'queued' } }
                return [pscustomobject]@{ workflow_runs = $runs }
            }
            throw "Unexpected fixture API: $path"
        }
        Invoke-DeferredReconciliation @(@{ id = 42 })
        $expected = if ($case -in @('ready', 'failed-pr', 'cancelled-pr', 'unrelated-pending')) { 1 } else { 0 }
        if ($script:posts.Count -ne $expected -or ($expected -eq 1 -and
            $script:posts[0] -cne 'repos/fixture/repo/actions/jobs/99/rerun')) { throw "Resume case failed: $case" }
    }
    Write-Host "Deferred CI reconciliation: $($cases.Count) cases passed."
    exit 0
}

if ($Repository -cnotmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$' -or $RunId -lt 0) {
    throw 'Invalid repository or run ID.'
}
if ($RunId -gt 0) {
    $candidates = @([pscustomobject]@{ id = $RunId })
} else {
    $since = [DateTimeOffset]::UtcNow.AddDays(-7).ToString('yyyy-MM-dd')
    $pages = @(Invoke-CiApi @('--paginate', '--slurp',
        "repos/$Repository/actions/workflows/sonarcloud.yml/runs?event=push&status=failure&created=%3E%3D$since&per_page=100"))
    $candidates = @($pages | ForEach-Object { $_.workflow_runs })
}
Invoke-DeferredReconciliation $candidates
exit 0
