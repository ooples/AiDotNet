<#
.SYNOPSIS
Renders the shard-selection decision as a readable report: the job summary and a JSON artifact.

.DESCRIPTION
Select-Shards.ps1 decides which test shards a change needs and writes selection.json. This turns that decision
into something a person can act on: why the full matrix ran (every blocking file and rule), what coverage alone
would have selected, every changed file's classification and the shards it reached, and every selected shard
with all of its reasons. With -TypeImpactPlanFile (the Build job's type-impact-plan.json) it adds the test-level
narrowing: which test classes each shard runs, and why the type-level plan could not resolve when it did not.

It never influences the matrix: it only reads the decision files. Missing or malformed input is reported, not fatal.

.PARAMETER SelectionFile      selection.json written by Select-Shards.ps1.
.PARAMETER TotalShards        Shards in the manifest (the full-matrix size).
.PARAMETER FinalShardCount    Shards the workflow actually runs (it can override the selector, e.g. no map).
.PARAMETER FinalEscalated     Whether the workflow ran the full matrix.
.PARAMETER WorkflowReason     Why the workflow overrode or skipped the selector, when it did.
.PARAMETER TypeImpactPlanFile type-impact-plan.json from Select-AffectedTests.ps1 (Build job).
.PARAMETER EffectiveFile      type-impact-effective.json: what actually ran after Select-AffectedTests post-processing.
.PARAMETER SummaryFile        Markdown destination; defaults to GITHUB_STEP_SUMMARY.
.PARAMETER ReportFile         JSON report destination.
#>
[CmdletBinding(DefaultParameterSetName = 'Report')]
param(
    [Parameter(ParameterSetName = 'Report')] [string] $SelectionFile = '',
    [Parameter(ParameterSetName = 'Report')] [int] $TotalShards = 0,
    [Parameter(ParameterSetName = 'Report')] [int] $FinalShardCount = -1,
    [Parameter(ParameterSetName = 'Report')] [switch] $FinalEscalated,
    [Parameter(ParameterSetName = 'Report')] [string] $WorkflowReason = '',
    [Parameter(ParameterSetName = 'Report')] [string] $TypeImpactPlanFile = '',
    [Parameter(ParameterSetName = 'Report')] [string] $EffectiveFile = '',
    [Parameter(ParameterSetName = 'Report')] [string] $SummaryFile = $env:GITHUB_STEP_SUMMARY,
    [Parameter(ParameterSetName = 'Report')] [string] $ReportFile = '',
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)
$ErrorActionPreference = 'Stop'

# GitHub caps one step's summary at 1 MiB; tables are truncated well before that, with the artifact as the full record.
$MaxRows = 200
$MaxSummaryChars = 700000

function Read-JsonFile([string] $Path) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-Prop($Object, [string] $Name) {
    if ($null -eq $Object) { return $null }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $null }
    return $p.Value
}

function Format-Cell([string] $Text) {
    if ($null -eq $Text) { return '' }
    return ($Text -replace '\|', '\|' -replace "`r?`n", ' ')
}

# The rule a reason string came from, so the summary can group 40 "abstract class" lines into one row.
function Get-ReasonRule([string] $Reason) {
    switch -Regex ($Reason) {
        '^current validation-selection control change' { return 'Selection tooling changed (SelectionControl)' }
        '^validation infrastructure or unknown GitHub configuration' { return 'Validation infrastructure (FullValidation)' }
        'declares an abstract class' { return 'Abstract test base: generated subclasses unresolved' }
        'which a source gen' { return 'Test helper consumed by a source generator' }
        '^no shard filter selects any test' { return 'Test file no shard filter selects' }
        '^not executed by any mapped shard' { return 'File not executed by any mapped shard' }
        '^changed file has no trustworthy line hunks' { return 'No trustworthy line hunks' }
        "identical to the map's copy" { return "Identical to the map's copy (no map line numbers)" }
        'auxiliary inventory changed' { return 'Auxiliary inventory changed; delta imports unsafe' }
        'retired shard' { return 'Reaches a retired shard' }
        '^selection was empty' { return 'Coverage selected nothing' }
        default { return 'Other' }
    }
}

function Split-Route([string] $Route) {
    $i = $Route.IndexOf(' <= ', [StringComparison]::Ordinal)
    if ($i -lt 0) { return $null }
    return [pscustomobject]@{ Shard = $Route.Substring(0, $i); Why = $Route.Substring($i + 4) }
}

# Structural routes are added whatever the change touches; they are reported apart from the change-driven ones.
function Test-StructuralRoute([string] $Why) {
    return $Why -match '^(is always run|is not in the coverage map yet|changed its manifest definition|reflection inventory changed|its manifest or execution policy changed)'
}

function New-SelectionReport($Selection, [int] $Total, [int] $FinalCount, [bool] $Escalated, [string] $Reason, $Plan, $Effective = $null) {
    $reasons = @(Get-Prop $Selection 'reasons' | Where-Object { $_ })
    # Evidence, not absence: a missing or partial selection.json must not read as "coverage selected nothing".
    $wouldProperty = if ($null -ne $Selection) { $Selection.PSObject.Properties['wouldSelect'] } else { $null }
    $shardsProperty = if ($null -ne $Selection) { $Selection.PSObject.Properties['shards'] } else { $null }
    $selectionAvailable = $null -ne $wouldProperty -or $null -ne $shardsProperty
    $would = @(if ($null -ne $wouldProperty) { @($wouldProperty.Value | Where-Object { $_ }) }
               elseif ($null -ne $shardsProperty) { @($shardsProperty.Value | Where-Object { $_ }) })
    $routes = @(Get-Prop $Selection 'routes' | Where-Object { $_ } | ForEach-Object { Split-Route ([string] $_) } | Where-Object { $_ })
    $files = @(Get-Prop $Selection 'files' | Where-Object { $_ })

    $shards = foreach ($group in ($routes | Group-Object Shard | Sort-Object Name)) {
        $why = @($group.Group.Why)
        $structural = @($why | Where-Object { Test-StructuralRoute $_ })
        [pscustomobject]@{
            name = $group.Name
            structuralOnly = ($structural.Count -eq $why.Count)
            routes = $why
        }
    }
    $escalations = foreach ($g in ($reasons | Group-Object { Get-ReasonRule ([string] $_) } | Sort-Object Count -Descending)) {
        [pscustomobject]@{ rule = $g.Name; count = $g.Count; reasons = @($g.Group) }
    }
    $structuralCounts = foreach ($g in ($routes | Where-Object { Test-StructuralRoute $_.Why } |
            Group-Object { ($_.Why -split ',')[0] } | Sort-Object Count -Descending)) {
        [pscustomobject]@{ rule = $g.Name; shards = @($g.Group.Shard | Sort-Object -Unique).Count }
    }

    $tests = $null
    if ($null -ne $Plan) {
        $planShards = @(Get-Prop $Plan 'shards' | Where-Object { $_ })
        $unresolved = @(Get-Prop $Plan 'unresolved' | Where-Object { $_ })
        $tests = [pscustomobject]@{
            resolved = [bool] (Get-Prop $Plan 'resolved')
            selectedTestClasses = Get-Prop $Plan 'selectedTestClasses'
            totalTestClasses = Get-Prop $Plan 'totalTestClasses'
            unresolved = @(foreach ($g in ($unresolved | Group-Object { [string] (Get-Prop $_ 'why') + [string] (Get-Prop $_ 'reason') } | Sort-Object Count -Descending)) {
                [pscustomobject]@{ reason = $g.Name; count = $g.Count; paths = @($g.Group | ForEach-Object { Get-Prop $_ 'path' } | Where-Object { $_ }) }
            })
            effective = $(if ($null -ne $Effective) { [pscustomobject]@{ mode = [string] (Get-Prop $Effective 'mode'); reason = [string] (Get-Prop $Effective 'reason'); shards = @(Get-Prop $Effective 'shards' | Where-Object { $_ }) } } else { $null })
            shards = @(foreach ($s in $planShards) {
                [pscustomobject]@{
                    name = Get-Prop $s 'name'; run = [bool] (Get-Prop $s 'run'); narrowed = [bool] (Get-Prop $s 'narrowed')
                    classes = Get-Prop $s 'classes'; testClasses = @(Get-Prop $s 'testClasses' | Where-Object { $_ })
                }
            })
        }
    }

    return [pscustomobject]@{
        schemaVersion = 1
        verdict = [pscustomobject]@{
            escalated = $Escalated; finalShards = $FinalCount; totalShards = $Total
            wouldSelect = $(if ($selectionAvailable) { $would.Count } else { $null })
            selectionAvailable = $selectionAvailable; workflowReason = $Reason
        }
        map = Get-Prop $Selection 'map'
        escalations = @($escalations)
        files = $files
        shards = @($shards)
        structural = @($structuralCounts)
        wouldSelect = $would
        tests = $tests
    }
}

function ConvertTo-SelectionMarkdown($Report) {
    $sb = [System.Text.StringBuilder]::new()
    $v = $Report.verdict
    [void] $sb.AppendLine('### Shard selection')
    [void] $sb.AppendLine()
    if ($v.escalated) {
        $alone = if ($v.selectionAvailable) { "Coverage alone would have selected **$($v.wouldSelect)** of $($v.totalShards)." }
                 else { 'Coverage selection unavailable: the selector left no evidence of what it would have selected.' }
        [void] $sb.AppendLine("**Full matrix: $($v.finalShards) of $($v.totalShards) shards.** $alone")
    }
    else {
        [void] $sb.AppendLine("**Selected $($v.finalShards) of $($v.totalShards) shards.**")
    }
    if ($v.workflowReason) { [void] $sb.AppendLine(); [void] $sb.AppendLine("Workflow override: $(Format-Cell $v.workflowReason)") }
    if ($Report.map) {
        [void] $sb.AppendLine()
        [void] $sb.AppendLine("Map ``$([string]$Report.map.sha)``: $($Report.map.changedSinceMap) file(s) changed since it was built; $($Report.map.changedByThisChange) changed by this change.")
    }

    if (@($Report.escalations).Count -gt 0) {
        [void] $sb.AppendLine(); [void] $sb.AppendLine('#### What forced the full matrix'); [void] $sb.AppendLine()
        [void] $sb.AppendLine('| Rule | Files | Examples |'); [void] $sb.AppendLine('| --- | ---: | --- |')
        foreach ($e in $Report.escalations) {
            $examples = (@($e.reasons | Select-Object -First 3 | ForEach-Object { Format-Cell $_ }) -join '<br>')
            [void] $sb.AppendLine("| $(Format-Cell $e.rule) | $($e.count) | $examples |")
        }
    }

    $files = @($Report.files)
    if ($files.Count -gt 0) {
        [void] $sb.AppendLine(); [void] $sb.AppendLine("#### Changed files ($($files.Count))"); [void] $sb.AppendLine()
        [void] $sb.AppendLine('| File | Category | Outcome | Shards | Why |'); [void] $sb.AppendLine('| --- | --- | --- | ---: | --- |')
        $order = @{ escalated = 0; routed = 1; reviewed = 2; skipped = 3 }
        $sorted = $files | Sort-Object @{ Expression = { $o = $order[[string] $_.outcome]; if ($null -eq $o) { 9 } else { $o } } }, @{ Expression = { -@($_.shards).Count } }, path
        foreach ($f in @($sorted | Select-Object -First $MaxRows)) {
            [void] $sb.AppendLine("| ``$(Format-Cell $f.path)`` | $($f.category) | $($f.outcome) | $(@($f.shards).Count) | $(Format-Cell $f.why) |")
        }
        if ($files.Count -gt $MaxRows) { [void] $sb.AppendLine("| ... $($files.Count - $MaxRows) more in the selection-report artifact | | | | |") }
    }

    if (@($Report.structural).Count -gt 0) {
        [void] $sb.AppendLine(); [void] $sb.AppendLine('#### Shards added whatever the change touched'); [void] $sb.AppendLine()
        [void] $sb.AppendLine('| Rule | Shards |'); [void] $sb.AppendLine('| --- | ---: |')
        foreach ($s in $Report.structural) { [void] $sb.AppendLine("| $(Format-Cell $s.rule) | $($s.shards) |") }
    }

    $shards = @($Report.shards)
    if ($shards.Count -gt 0) {
        $changeDriven = @($shards | Where-Object { -not $_.structuralOnly })
        [void] $sb.AppendLine(); [void] $sb.AppendLine("#### Shards the change reached ($($changeDriven.Count))"); [void] $sb.AppendLine()
        foreach ($s in @($changeDriven | Select-Object -First $MaxRows)) {
            $why = @($s.routes | Where-Object { -not (Test-StructuralRoute $_) })
            [void] $sb.AppendLine("<details><summary><b>$(Format-Cell $s.name)</b> - $($why.Count) reason(s)</summary>"); [void] $sb.AppendLine()
            foreach ($w in @($why | Select-Object -First 50)) { [void] $sb.AppendLine("- $(Format-Cell $w)") }
            if ($why.Count -gt 50) { [void] $sb.AppendLine("- ... $($why.Count - 50) more in the artifact") }
            [void] $sb.AppendLine(); [void] $sb.AppendLine('</details>')
        }
    }

    if ($null -ne $Report.tests) {
        $t = $Report.tests
        [void] $sb.AppendLine(); [void] $sb.AppendLine('### Test-level narrowing'); [void] $sb.AppendLine()
        if ($t.resolved) {
            [void] $sb.AppendLine("Resolved: **$($t.selectedTestClasses)** of $($t.totalTestClasses) test classes reachable from the change.")
        }
        else {
            [void] $sb.AppendLine("**Unresolved** - the shard selection above runs unnarrowed. Causes:"); [void] $sb.AppendLine()
            [void] $sb.AppendLine('| Cause | Files | Examples |'); [void] $sb.AppendLine('| --- | ---: | --- |')
            foreach ($u in @($t.unresolved)) {
                [void] $sb.AppendLine("| $(Format-Cell $u.reason) | $($u.count) | $((@($u.paths | Select-Object -First 3 | ForEach-Object { '`' + (Format-Cell $_) + '`' })) -join '<br>') |")
            }
        }
        # What actually runs: the effective matrix when Select-AffectedTests recorded it. The plan alone lists
        # candidates for every manifest shard, before the intersection with the chosen matrix, passthrough, and
        # any narrowing undone to fit the output limit.
        if ($null -ne $t.effective) {
            [void] $sb.AppendLine(); [void] $sb.AppendLine("Applied: **$($t.effective.mode)**$(if ($t.effective.reason) { " ($(Format-Cell $t.effective.reason))" })")
            $rows = @($t.effective.shards)
            if ($rows.Count -gt 0) {
                [void] $sb.AppendLine(); [void] $sb.AppendLine('| Shard | Runs | Test classes |'); [void] $sb.AppendLine('| --- | --- | ---: |')
                foreach ($s in @($rows | Sort-Object { if ($_.narrowed) { -[int] $_.classes } else { 1 } } | Select-Object -First $MaxRows)) {
                    $runs = if ($s.narrowed) { 'narrowed' } else { 'whole shard' }
                    $count = if ($s.narrowed) { "$($s.classes)" } else { 'all' }
                    [void] $sb.AppendLine("| $(Format-Cell $s.name) | $runs | $count |")
                }
            }
        }
        else {
            [void] $sb.AppendLine(); [void] $sb.AppendLine('_The effective matrix was not recorded; the plan below lists candidates, not what ran._')
            $ran = @($t.shards | Where-Object { $_.run })
            if ($ran.Count -gt 0) {
                [void] $sb.AppendLine(); [void] $sb.AppendLine('| Shard (candidate) | Narrowed | Test classes |'); [void] $sb.AppendLine('| --- | --- | ---: |')
                foreach ($s in @($ran | Sort-Object { -[int] $_.classes } | Select-Object -First $MaxRows)) {
                    [void] $sb.AppendLine("| $(Format-Cell $s.name) | $($s.narrowed) | $($s.classes) |")
                }
            }
        }
    }

    $text = $sb.ToString()
    if ($text.Length -gt $MaxSummaryChars) {
        $text = $text.Substring(0, $MaxSummaryChars) + "`n`n_Truncated; the selection-report artifact has the full record._`n"
    }
    return $text
}

if ($SelfTest) {
    $fixture = [pscustomobject]@{
        escalate = $true
        reasons = @(
            'test source tests/X/ABase.cs declares an abstract class, which build-time generated test classes may derive from',
            'test source tests/X/BBase.cs declares an abstract class, which build-time generated test classes may derive from',
            'validation infrastructure or unknown GitHub configuration: src/AiDotNet.Generators/G.cs')
        shards = @('Alpha', 'Always', 'Beta')
        wouldSelect = @('Alpha', 'Always', 'Beta')
        routes = @('Always <= is always run', 'Alpha <= executes changed lines 3-4 of src/A.cs', 'Beta <= executes src/A.cs, whose changed lines 9-9 no shard executes')
        files = @(
            [pscustomobject]@{ path = 'src/A.cs'; category = 'MapCandidate'; outcome = 'routed'; shards = @('Alpha', 'Beta'); why = '1 shard(s) execute its changed lines' },
            [pscustomobject]@{ path = 'src/AiDotNet.Generators/G.cs'; category = 'FullValidation'; outcome = 'escalated'; shards = @(); why = 'validation infrastructure' },
            [pscustomobject]@{ path = 'docs/x.md'; category = 'NonRuntime'; outcome = 'skipped'; shards = @(); why = 'cannot affect any test' })
        map = [pscustomobject]@{ sha = 'abc'; changedSinceMap = 10; changedByThisChange = 3 }
    }
    $plan = [pscustomobject]@{
        resolved = $false; selectedTestClasses = 0; totalTestClasses = 100
        unresolved = @([pscustomobject]@{ path = 'src/G.cs'; why = 'source generator input' }, [pscustomobject]@{ path = 'src/H.cs'; why = 'source generator input' })
        shards = @([pscustomobject]@{ name = 'Alpha'; run = $true; narrowed = $false; classes = 7; testClasses = @('T1') })
    }
    $r = New-SelectionReport $fixture 5 5 $true '' $plan
    $md = ConvertTo-SelectionMarkdown $r
    function Check([string] $Name, [bool] $Ok) { [pscustomobject]@{ Name = $Name; Ok = $Ok } }
    $empty = ConvertTo-SelectionMarkdown (New-SelectionReport $null 164 164 $true 'no certified shard map' $null)
    # The plan claims Alpha narrowed and lists a candidate Gamma; the effective record says the plan was unresolved,
    # so the chosen matrix (Alpha only) ran whole. The report must follow the effective record.
    $planWithCandidate = [pscustomobject]@{ resolved = $false; unresolved = @(); shards = @(
        [pscustomobject]@{ name = 'Alpha'; run = $true; narrowed = $true; classes = 7; testClasses = @('T1') },
        [pscustomobject]@{ name = 'Gamma'; run = $true; narrowed = $true; classes = 3; testClasses = @('T2') }) }
    $effectiveRecord = [pscustomobject]@{ mode = 'passthrough'; reason = 'a changed file cannot be mapped to types'; shards = @(
        [pscustomobject]@{ name = 'Alpha'; narrowed = $false; classes = $null }) }
    $eff = ConvertTo-SelectionMarkdown (New-SelectionReport $null 5 5 $false '' $planWithCandidate $effectiveRecord)
    $checks = @(
        (Check 'verdict names the full matrix and the would-select count' ($md -match 'Full matrix: 5 of 5 shards\.\*\* Coverage alone would have selected \*\*3\*\* of 5')),
        (Check 'two abstract-base reasons collapse into one rule row with count 2' ($md -match '\| Abstract test base: generated subclasses unresolved \| 2 \|')),
        (Check 'generator escalation is its own rule' ($md -match 'Validation infrastructure \(FullValidation\) \| 1 \|')),
        (Check 'escalated files sort before routed, routed before skipped' (($md.IndexOf('`src/AiDotNet.Generators/G.cs`') -lt $md.IndexOf('`src/A.cs`')) -and ($md.IndexOf('`src/A.cs`') -lt $md.IndexOf('`docs/x.md`')))),
        (Check 'always-run is structural, not change-driven' (($md -match '\| is always run \| 1 \|') -and ($md -notmatch '<b>Always</b>'))),
        (Check 'change-driven shards list their reasons' (($md -match '<b>Alpha</b> - 1 reason') -and ($md -match '<b>Beta</b> - 1 reason'))),
        (Check 'unresolved type-impact causes are grouped' ($md -match '\| source generator input \| 2 \|')),
        (Check 'test-level table lists the shard' ($md -match '\| Alpha \| False \| 7 \|')),
        (Check 'report JSON carries wouldSelect' (@($r.wouldSelect).Count -eq 3)),
        (Check 'a missing selection still renders, naming the workflow reason' ($empty -match 'Workflow override: no certified shard map')),
        (Check 'a missing selection says coverage is unavailable, not that it selected 0' (($empty -match 'Coverage selection unavailable') -and ($empty -notmatch 'would have selected \*\*0\*\*'))),
        (Check 'with the effective record, a plan-narrowed shard that passed through shows as a whole shard' (($eff -match '\| Alpha \| whole shard \| all \|') -and ($eff -match 'Applied: \*\*passthrough\*\*'))),
        (Check 'a candidate the effective matrix did not run is not listed' ($eff -notmatch '\| Gamma \|'))
    )
    foreach ($c in $checks) { Write-Host ("  [{0}] {1}" -f $(if ($c.Ok) { 'OK' } else { 'FAIL' }), $c.Name) }
    $failed = @($checks | Where-Object { -not $_.Ok })
    if ($failed.Count -gt 0) { Write-Host "Write-SelectionReport self-test FAILED ($($failed.Count))"; exit 1 }
    Write-Host 'Write-SelectionReport self-test passed.'
    exit 0
}

try {
    $selection = Read-JsonFile $SelectionFile
    $plan = Read-JsonFile $TypeImpactPlanFile
    $finalCount = if ($FinalShardCount -ge 0) { $FinalShardCount } else { @(Get-Prop $selection 'shards').Count }
    $effective = Read-JsonFile $EffectiveFile
    $report = New-SelectionReport $selection $TotalShards $finalCount ([bool] $FinalEscalated) $WorkflowReason $plan $effective
    if ($ReportFile) { $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReportFile -Encoding utf8 }
    if ($SummaryFile) {
        # With a plan, the Build job appends only the test-level section; the selection part is Select shards' own.
        $markdown = ConvertTo-SelectionMarkdown $report
        if ($null -ne $plan -and $null -eq $selection) { $markdown = $markdown.Substring($markdown.IndexOf('### Test-level narrowing')) }
        Add-Content -LiteralPath $SummaryFile -Value $markdown -Encoding utf8
    }
}
catch {
    # Never swallowed: exit nonzero so the caller sees the failure. The Select step then writes its own fallback
    # verdict, and the Build job runs this step with continue-on-error, so the matrix is never gated on the report.
    Write-Host "::warning::selection report could not be written: $($_.Exception.Message)"
    exit 1
}
exit 0
