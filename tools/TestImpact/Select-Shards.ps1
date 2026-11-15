<#
.SYNOPSIS
    Chooses which CI shards a pull request needs, from the coverage-derived shard map.

.DESCRIPTION
    Intersects changed line ranges against the map's file -> shard index. Selection is deliberately
    fail-safe: every changed file and every changed hunk must be accounted for, or the caller is told
    to run the full matrix.

.PARAMETER MapFile
    shard-map.json from New-ShardMap.ps1.

.PARAMETER ExpectedShards
    The complete current shard manifest. A map naming a shard the manifest no longer has is never
    trusted. A manifest shard the map has not seen yet - one added since the map was built - is
    run on every selection until a map includes it, exactly as if it were an always-run shard.

.PARAMETER AuditUnchangedMap
    Allows the nightly selection-miss audit to evaluate an unchanged map tree by selecting only
    always-run shards. Ordinary PR selection must not pass this switch: an unexpectedly empty PR
    diff remains a fail-closed full-matrix decision.

.PARAMETER PullRequestHeadSha
    The pull request's head commit. The checkout must be GitHub's merge of that head onto the base
    branch; its first parent is then the exact base this run validates against, and only the
    paths the merge changes relative to it are this pull request's change. Mutually exclusive
    with BaseSha.

.PARAMETER MapLineageRef
    The branch the map was measured on, when that is not the pull request's base branch: a stacked
    pull request selecting with master's map because its own base branch has none. Only valid with
    PullRequestHeadSha. The change selected for is then everything the checkout adds to that
    branch (the diff from their merge base), not only the diff from the base branch - the base
    branch's own commits were never measured by, or validated against, this map's lineage. The
    checkout must contain the map's commit; otherwise selection escalates.

.PARAMETER BaseSha
    An explicit base for the current change. The nightly audit passes the audited commit itself,
    so every map-to-HEAD path is replayed as one change and only selection-control edits are
    treated as historical. Pull requests must use PullRequestHeadSha instead: the event's
    base.sha is stale whenever the pull request is behind its base branch.

.PARAMETER DeltaFromTree
    A Git tree that was already validated. Selection is scoped to the paths that differ between
    it and HEAD - the change a landed commit adds on top of what a pull request run tested. Used
    by post-merge reuse to decide which of the pull request's shards master's own later commits
    could have affected. Mutually exclusive with PullRequestHeadSha and BaseSha.

.PARAMETER ShardManifestFile
    JSON array of { name, project, filter } for every shard (the converted test-shards.yml).
    Test sources are never in the coverage map, so without this every test-file change escalates;
    with it they are routed to the shards whose filters select their tests.

.PARAMETER SelfTest
    Runs the built-in adversarial checks and exits.

.NOTES
    Environment: TEST_IMPACT_PRECISE_FALLBACKS. A changed range no shard executes is selected by what
    it changed (nothing for comments and spacing, the enclosing member for a body, the enclosing type
    for a declaration) instead of by every shard that executes the file. Setting the variable to 0,
    false, off, no or disabled restores the file-wide rule; unset, or any other value, leaves the
    precise rules on. The Select shards job passes the repository variable of the same name.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $MapFile,
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string[]] $ExpectedShards,
    [Parameter(ParameterSetName = 'Select')] [switch] $AuditUnchangedMap,
    [Parameter(ParameterSetName = 'Select')] [string] $ShardManifestFile,
    [Parameter(ParameterSetName = 'Select')] [string] $DeltaFromTree,
    [Parameter(ParameterSetName = 'Select')] [string] $MapLineageRef,
    [Parameter(ParameterSetName = 'Select')]
    [Parameter(ParameterSetName = 'Classify')] [string] $BaseSha,
    [Parameter(ParameterSetName = 'Select')]
    [Parameter(ParameterSetName = 'Classify')] [string] $PullRequestHeadSha,
    [Parameter(ParameterSetName = 'Select')]
    [Parameter(ParameterSetName = 'Classify')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'Classify')] [switch] $ClassifyOnly,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum ChangedPathImpact {
    NonRuntime
    BuildOnly
    MapCandidate
    SelectionControl
    FullValidation
}

$script:SharedInfrastructureFiles = @(
    'Directory.Build.props', 'Directory.Build.targets', 'Directory.Packages.props',
    'global.json', 'nuget.config', 'NuGet.config'
)
# Compile the change, but do not run the test matrix for it. An .editorconfig edit can raise an
# analyzer to error and fail the BUILD, which is why it cannot be NonRuntime - the build jobs and
# the shard matrix share one gate, so calling it non-runtime would skip the compile that catches it.
# It cannot change runtime behaviour: nothing under src/AiDotNet.Generators reads
# AnalyzerConfigOptions (verified by grep), so no generated test text depends on it, and the
# remaining keys are formatting and diagnostic severities. Escalating 164 shards - 34,421 tests -
# to prove an indent rule is the single largest unjustified escalation measured on real PRs.
$script:BuildOnlyFiles = @('.editorconfig')
# Analyzer and pack inputs of src/AiDotNet.csproj, by exact path. None is compiled into an assembly or
# read by a test (verified by grep over tests/), and each is checked by a job that a build-only
# change still runs:
#   - PublicAPI.Shipped/Unshipped.txt: AdditionalFiles of Microsoft.CodeAnalysis.PublicApiAnalyzers
#     (RS0016/RS0017), reported by the Build job's analyzer-enabled compile of src/AiDotNet.csproj.
#   - ElementWiseBaseline.txt: an AdditionalFile read only by ElementWiseAnalyzer, a
#     DiagnosticAnalyzer (ADNELEM001-003). No source generator reads AdditionalFiles, so no
#     generated code depends on it.
#   - CompatibilitySuppressions.xml: package validation, run by the compat build's `dotnet pack`.
# They are regenerated with almost every product change, so until they were listed here an
# otherwise mapped pull request escalated on "not executed by any mapped shard" for a text file
# (18 of 42 pull-request runs measured in November 2026).
$script:BuildOnlyPaths = @(
    'src/PublicAPI.Shipped.txt',
    'src/PublicAPI.Unshipped.txt',
    'src/ElementWiseBaseline.txt',
    'src/CompatibilitySuppressions.xml'
)
# The per-framework public API baselines: src/PublicAPI/<tfm>/PublicAPI.(Shipped|Unshipped).txt.
$script:BuildOnlyPublicApiDirectory = 'src/PublicAPI/'
$script:BuildOnlyPublicApiNames = @('PublicAPI.Shipped.txt', 'PublicAPI.Unshipped.txt')
# Benchmark projects. The solution build compiles them (which is why a build-only change still
# builds), but no test shard executes them: no test or product project references them and
# .github/test-shards.yml names none. Test-BuildOnlyDirectoryConsumed re-checks that against the
# tree on every run, so the day a test project references one of these the selector stops trusting
# it. Build files inside them (Directory.Build.props, global.json) stay full validation: the
# shared-infrastructure rule is applied first.
$script:BuildOnlyDirectories = @('AiDotNetBenchmarkTests/', 'benchmarks/')
# Per tool directory: whether this workflow's build or tests reference it (Test-ToolDirectoryReferenced).
$script:ToolReferenceCache = @{}
$script:FullValidationPaths = @(
    '.github/test-shards.yml',
    '.github/test-shard-changes.json'
)
$script:SelectionControlPaths = @(
    '.github/workflows/sonarcloud.yml',
    '.github/workflows/test-impact-map.yml',
    '.github/workflows/ci-shard-closure-policy.yml'
)
# Workflows that take part in the validation pipeline without choosing shards: full validation.
#   - resume-deferred-ci.yml re-runs the deferred jobs of a Build & SonarCloud run once the pull request run it
#     waits for completes, so it decides when post-merge validation executes.
# Every file under .github/workflows is in exactly one of SelectionControlPaths, this list and
# NonRuntimeWorkflowPaths; Assert-WorkflowsClassified fails the self-test for a file in none of them.
$script:PipelineWorkflowPaths = @(
    '.github/workflows/resume-deferred-ci.yml'
)
# Build-time code: the source generators the test project loads as an analyzer. It runs inside the
# compiler, so runtime coverage never records it, yet one edit can rewrite thousands of generated
# test classes across every ModelFamily shard. No coverage-derived routing can bound that.
$script:BuildTimeDirectories = @('src/AiDotNet.Generators/')
$script:FullValidationDirectories = @('.github/actions/', '.github/scripts/') + $script:BuildTimeDirectories
$script:SelectionControlDirectories = @('tools/TestImpact/')
# Trees that contain no compilable product code, so no change inside them can alter a C# test.
#
# MEASURED, NOT ASSUMED. website/ holds the documentation site: on master it contains zero .cs and
# zero .csproj files and appears nowhere in AiDotNet.sln, and its pipelines (ci-website.yml,
# deploy-website.yml) are already listed above as independent. Until this entry existed a change to
# website/package-lock.json fell through to MapCandidate, found no coverage entry, and escalated --
# PR #2223 changed that one JavaScript lockfile and ran all 164 shards.
#
# AN ALLOWLIST, and deliberately short. A directory earns a place here only by demonstrably holding
# nothing the solution compiles; Assert-NonRuntimeDirectories re-checks that on every run, so the
# day someone adds a project under one of these the selector stops trusting it.
#
# tools/reference-data/ holds the Python scripts that generate reference fixtures from the original
# implementations (369 .py, a README and a requirements file; no .cs, no project). Tests read the
# generated fixtures under tests/, never this directory (verified by grep over tests/), and
# tts-paper-contracts.yml triggers on it by itself. It fell through to MapCandidate only because
# comments in src/ name its scripts, which Test-ToolDirectoryReferenced counts as a reference.
$script:NonRuntimeDirectories = @('website/', 'tools/reference-data/')
# Exact root-level files nothing in this workflow's build or tests reads. vercel.json configures the
# Vercel deployment of api/ (scripts/vercel-ignore-api.sh is its only reader); a root-level file has
# no directory for Test-ToolDirectoryReferenced to scope by, so it used to escalate as unmapped.
$script:NonRuntimePaths = @('vercel.json')
# These helpers cannot choose shards or certify validation. They are exercised by the
# mandatory tooling checks before selection, including real HTTP transfer regressions.
# Keep this exact: unknown helpers and selection/certificate policy remain fail-closed.
$script:IndependentToolPaths = @(
    'tools/TestImpact/Receive-RequiredArtifact.ps1',
    'tools/TestImpact/Test-RequiredArtifactResume.ps1',
    'tools/TestImpact/Test-CiImpactWorkflow.ps1'
)
# Workflows independent of the test pipeline: not sonarcloud.yml or the map workflow, not called by them
# (workflow_call), defining nothing the test jobs run, and producing neither the coverage map nor a selection
# input. Each has its own triggers, so changing one cannot change what a shard executes. Read before listing:
#   - deploys, release, website and docs: azure-functions-deploy, deploy-serving, deploy-website, docs, docs-wiki,
#     release-please, samples, ci-website, website-alert, website-claims, website-e2e, website-health, and the three
#     *.yml.disabled release files (GitHub does not load them at all; re-enabling one is a rename to a new path,
#     which is unclassified until listed).
#   - linting and review: codacy, commitlint, commitlint-fix, pr-title-lint, dotnet-format-autofix,
#     copilot-review-gate.
#   - code scanning: codeql.yml. It is not in this tree yet: #2585 moves both CodeQL jobs out of sonarcloud.yml
#     into it (push to master, nightly, manual dispatch; no pull-request trigger, no workflow_call, only
#     third-party actions). Listed ahead of time, as read on that branch, so the two changes can merge in either
#     order; a listed path with no file is allowed.
#   - cancel and report helpers: cancel-on-pr-close and cancel-superseded-merged-runs only cancel runs;
#     cancellation-report runs after a cancelled run and explains it. None can make a shard pass or be skipped.
#   - separate checks with their own jobs and triggers: layer-factory-architecture, recurrent-foundations,
#     tts-paper-contracts, test-attribution-canary, self-hosted-routing-check (manual dispatch).
#   - nightlies: cross-isa-agreement-nightly, heavy-timeout-nightly, paper-fidelity-nightly,
#     model-performance-census.
# The scripts they run stay classified by their own paths (.github/scripts, .github/actions, tools/TestImpact).
$script:NonRuntimeWorkflowPaths = @(
    '.github/workflows/automated-release.yml.disabled',
    '.github/workflows/azure-functions-deploy.yml',
    '.github/workflows/cancel-on-pr-close.yml',
    '.github/workflows/cancel-superseded-merged-runs.yml',
    '.github/workflows/cancellation-report.yml',
    '.github/workflows/ci-website.yml',
    '.github/workflows/codacy.yml',
    '.github/workflows/codeql.yml',
    '.github/workflows/commitlint-fix.yml',
    '.github/workflows/commitlint.yml',
    '.github/workflows/copilot-review-gate.yml',
    '.github/workflows/cross-isa-agreement-nightly.yml',
    '.github/workflows/deploy-serving.yml',
    '.github/workflows/deploy-website.yml',
    '.github/workflows/docs-wiki.yml',
    '.github/workflows/docs.yml',
    '.github/workflows/dotnet-format-autofix.yml',
    '.github/workflows/heavy-timeout-nightly.yml',
    '.github/workflows/layer-factory-architecture.yml',
    '.github/workflows/model-performance-census.yml',
    '.github/workflows/paper-fidelity-nightly.yml',
    '.github/workflows/pr-title-lint.yml',
    '.github/workflows/publish.yml.disabled',
    '.github/workflows/recurrent-foundations.yml',
    '.github/workflows/release-please.yml',
    '.github/workflows/release.yml.disabled',
    '.github/workflows/samples.yml',
    '.github/workflows/self-hosted-routing-check.yml',
    '.github/workflows/test-attribution-canary.yml',
    '.github/workflows/tts-paper-contracts.yml',
    '.github/workflows/website-alert.yml',
    '.github/workflows/website-claims.yml',
    '.github/workflows/website-e2e.yml',
    '.github/workflows/website-health.yml'
)

function Test-SelectionControl {
    param([string] $Path)
    $normalized = $Path.Replace('\', '/')
    foreach ($entry in $script:SelectionControlPaths) {
        if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    foreach ($entry in $script:SelectionControlDirectories) {
        if ($normalized.StartsWith($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Test-SharedInfrastructure {
    param([string] $Path)
    $normalized = $Path.Replace('\', '/')
    $name = [System.IO.Path]::GetFileName($normalized)
    foreach ($entry in $script:SharedInfrastructureFiles) {
        # MSBuild and NuGet apply these names per-directory, so a nested file is just as capable of
        # changing compilation as the root one. Exact basename matching keeps lookalike suffixes out.
        if ($name -ieq $entry) { return $true }
    }
    foreach ($entry in $script:FullValidationPaths) {
        if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    foreach ($entry in $script:FullValidationDirectories) {
        if ($normalized.StartsWith($entry, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Get-ChangedPathImpact {
    param([Parameter(Mandatory)] [string] $Path)

    $normalized = $Path.Replace('\', '/')
    foreach ($entry in $script:IndependentToolPaths) {
        if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) {
            return [ChangedPathImpact]::NonRuntime
        }
    }
    if (Test-SelectionControl -Path $normalized) {
        return [ChangedPathImpact]::SelectionControl
    }
    if (Test-ScriptUsedOnlyByIndependentWorkflows -Path $normalized) {
        return [ChangedPathImpact]::NonRuntime
    }
    # A version bump of packages that provably reach no product code (Resolve-PackagePropsVerdict). Without a
    # verdict for this run, and for every other outcome, the shared-infrastructure rule below applies.
    if ($normalized -ceq $script:PackagePropsPath -and $null -ne $script:PackagePropsVerdict) {
        if ($script:PackagePropsVerdict.Impact -ceq 'BuildOnly') { return [ChangedPathImpact]::BuildOnly }
        if ($script:PackagePropsVerdict.Impact -ceq 'TestRoute') { return [ChangedPathImpact]::MapCandidate }
    }
    if (Test-SharedInfrastructure -Path $normalized) {
        return [ChangedPathImpact]::FullValidation
    }
    # Checked after SelectionControl and SharedInfrastructure so a build-only name can never
    # downgrade a path those already claimed.
    $buildOnlyName = [System.IO.Path]::GetFileName($normalized)
    foreach ($entry in $script:BuildOnlyFiles) {
        if ($buildOnlyName -ieq $entry) { return [ChangedPathImpact]::BuildOnly }
    }
    # Exact and case-sensitive: a path that differs only in case is another file on the runners,
    # not the reviewed analyzer or pack input.
    if ($normalized -cin $script:BuildOnlyPaths) { return [ChangedPathImpact]::BuildOnly }
    if ($normalized.StartsWith($script:BuildOnlyPublicApiDirectory, [StringComparison]::Ordinal) -and
        $buildOnlyName -cin $script:BuildOnlyPublicApiNames) {
        return [ChangedPathImpact]::BuildOnly
    }
    foreach ($entry in $script:BuildOnlyDirectories) {
        if ($normalized.StartsWith($entry, [StringComparison]::Ordinal) -and
            -not (Test-BuildOnlyDirectoryConsumed -Directory $entry)) {
            return [ChangedPathImpact]::BuildOnly
        }
    }
    if ($normalized -cin $script:NonRuntimePaths) { return [ChangedPathImpact]::NonRuntime }

    # AFTER shared infrastructure, so a build file keeps its meaning wherever it sits, and before the
    # markdown and map-candidate rules, so a non-product tree is spared whatever its file extension.
    foreach ($entry in $script:NonRuntimeDirectories) {
        if ($normalized.StartsWith($entry, [StringComparison]::OrdinalIgnoreCase)) {
            return [ChangedPathImpact]::NonRuntime
        }
    }

    # Markdown cannot alter a build or runtime. Known independent workflows have their own triggers
    # and jobs; changing one cannot alter this validation workflow. This is an allowlist so a newly
    # added or renamed workflow remains full-validation until its independence is reviewed.
    if ([System.IO.Path]::GetExtension($normalized).Equals('.md', [StringComparison]::OrdinalIgnoreCase)) {
        return [ChangedPathImpact]::NonRuntime
    }
    if ($normalized.StartsWith('.github/workflows/', [StringComparison]::OrdinalIgnoreCase)) {
        foreach ($entry in $script:NonRuntimeWorkflowPaths) {
            if ($normalized.Equals($entry, [StringComparison]::OrdinalIgnoreCase)) {
                return [ChangedPathImpact]::NonRuntime
            }
        }
        return [ChangedPathImpact]::FullValidation
    }

    # Unknown GitHub configuration can affect analysis, generated reports, or a required check. It
    # is intentionally not eligible for coverage-map reduction until classified explicitly. The exception (R9) is a
    # file no workflow, action, script or tool names (CODEOWNERS, issue templates, FUNDING.yml, dependabot.yml): GitHub
    # reads those itself, and none of them can change what a test shard executes.
    if ($normalized.StartsWith('.github/', [StringComparison]::OrdinalIgnoreCase)) {
        if (-not (Test-GitHubFileReferenced -Path $normalized)) { return [ChangedPathImpact]::NonRuntime }
        return [ChangedPathImpact]::FullValidation
    }

    # A standalone tool, or a top-level tree such as api/ (an Azure Function with its own deploy workflow), that
    # nothing in this workflow's build or tests references cannot change what any shard executes. Checked per
    # directory against the tree; any doubt (a failed search, a root-level file, a dot-directory) keeps the old
    # behaviour. src/ and tests/ are the product and are never eligible.
    $top = $normalized.Split('/')[0]
    if ($normalized.Contains('/') -and -not $top.StartsWith('.') -and $top -notin @('src', 'tests') -and
        -not (Test-ToolDirectoryReferenced -Path $normalized)) {
        return [ChangedPathImpact]::NonRuntime
    }

    return [ChangedPathImpact]::MapCandidate
}

<#
.SYNOPSIS
Whether anything this workflow builds, tests or invokes references a tools/<name>/ directory.

.DESCRIPTION
Searched at HEAD in tests/, src/, the solution files, Directory.Build.* and this workflow file. Other workflows are
left out on purpose: they have their own triggers and cannot change what a test shard here executes. A tool
mentioned anywhere in that set keeps the old behaviour (MapCandidate, so it escalates as unmapped). A failed search
counts as referenced: never trade safety for a narrower matrix.
#>
function Test-ToolDirectoryReferenced {
    param([Parameter(Mandatory)] [string] $Path)
    $segments = $Path.Split('/')
    if ($segments[0] -ieq 'tools') {
        if ($segments.Count -lt 3) { return $true }   # a file directly under tools/: no directory to scope by
        $directory = 'tools/' + $segments[1]
        if ($script:ToolReferenceCache.ContainsKey($directory)) { return $script:ToolReferenceCache[$directory] }
        $pattern = 'tools[/\\]' + [regex]::Escape($segments[1]) + '([/\\]|[^A-Za-z0-9_.-]|$)'
        $null = & git grep -q -i -E $pattern HEAD -- 'tests/' 'src/' '*.sln' '*.slnx' 'Directory.Build.props' 'Directory.Build.targets' '.github/workflows/sonarcloud.yml' 2>$null
        # git grep: 0 = found, 1 = not found, anything else = the search itself failed.
        $referenced = $LASTEXITCODE -ne 1
        $script:ToolReferenceCache[$directory] = $referenced
        return $referenced
    }

    # Any other top-level tree (api/, schemas/, ...). A file reaches a test shard through the build (an MSBuild item,
    # a solution entry, this workflow) or by a test reading it at run time. Product .cs files are not searched: a
    # route such as [Route("api/...")] is text, not a path, and product code does not read the checkout's layout.
    # Tests are searched for the directory as a path, or as a Path.Combine/Path.Join argument ("samples").
    $directory = $segments[0]
    if ($script:ToolReferenceCache.ContainsKey($directory)) { return $script:ToolReferenceCache[$directory] }
    $name = [regex]::Escape($directory)
    # POSIX ERE (git grep -E): inside brackets \s is a backslash or an 's', which under -i let base64 fixture text
    # such as "SapI/" count as a reference to api/. [:space:] is the whitespace class.
    $asPath = '(^|["''[:space:]=;(,>]|\.\.[/\\])' + $name + '[/\\]'
    $null = & git grep -q -i -E $asPath HEAD -- '*.csproj' '*.props' '*.targets' '*.sln' '*.slnx' '.github/workflows/sonarcloud.yml' 2>$null
    $referenced = $LASTEXITCODE -ne 1
    if (-not $referenced) {
        $null = & git grep -q -i -E $asPath HEAD -- 'tests/' 2>$null
        $referenced = $LASTEXITCODE -ne 1
    }
    if (-not $referenced) {
        # A Path.Combine/Path.Join argument can sit on a later line than the call, and git grep matches single
        # lines, so this reads the candidate files whole.
        $referenced = Test-HeadTextJoinsDirectory -Name $directory -Pathspec 'tests/'
    }
    $script:ToolReferenceCache[$directory] = $referenced
    return $referenced
}

<#
.SYNOPSIS
Whether a file under a pathspec at HEAD passes a directory name as a Path.Combine/Path.Join argument.

.DESCRIPTION
git grep matches one line at a time, so `Path.Combine(root,` on one line and `"benchmarks", "data")` on the next was
never seen. A fixed-string search finds the few files that hold the name as a whole string literal ("benchmarks" or
'benchmarks'); each is then read whole from HEAD and matched as complete text. The argument may follow others that
are themselves calls: Path.Combine(GetRoot(), "benchmarks"). A failed search or read counts as a reference.
#>
function Test-HeadTextJoinsDirectory {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [string] $Pathspec
    )
    $naming = @(& git -c core.quotepath=false grep -l -i -F -e ('"' + $Name + '"') -e ("'" + $Name + "'") HEAD -- $Pathspec 2>$null)
    if ($LASTEXITCODE -gt 1) { return $true }
    if ($LASTEXITCODE -ne 0 -or $naming.Count -eq 0) { return $false }
    # The argument list before the name: anything but parentheses, or a parenthesised call one level deep.
    $pattern = 'Path\.(Combine|Join)\s*\((?:[^()]|\([^()]*\))*?["'']' + [regex]::Escape($Name) + '["'']'
    $regex = [regex]::new($pattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase, [TimeSpan]::FromSeconds(5))
    foreach ($entry in $naming) {
        $file = ([string] $entry) -replace '^HEAD:', ''
        $lines = @(& git show "HEAD:$file" 2>$null)
        if ($LASTEXITCODE -ne 0) { return $true }
        try {
            if ($regex.IsMatch(($lines -join "`n"))) { return $true }
        }
        catch [Text.RegularExpressions.RegexMatchTimeoutException] { return $true }
    }
    return $false
}

<#
.SYNOPSIS
Whether a test or product project consumes a directory trusted as build-only.

.DESCRIPTION
A build-only directory holds code the solution compiles and no test shard executes. That stops being true when a
project outside it references it (a ProjectReference, a linked Compile item), the shard manifest names it, or a test
reads it at run time. Searched at HEAD: every project and MSBuild file outside the directory itself (another
build-only directory included) and the shard manifest for the directory as a path, then tests/ for it inside a string
literal or as a Path.Combine/Path.Join argument, the latter read as whole files so a call split across lines counts.
A comment that merely names a benchmark is not a reference. A failed search counts as consumed.
#>
function Test-BuildOnlyDirectoryConsumed {
    param([Parameter(Mandatory)] [string] $Directory)
    $key = 'build-only:' + $Directory
    if ($script:ToolReferenceCache.ContainsKey($key)) { return $script:ToolReferenceCache[$key] }
    $name = [regex]::Escape($Directory.TrimEnd('/'))
    # Only the directory itself is left out. A project in another build-only directory that links a file from
    # this one is a consumer too: a test project may reference that project.
    $excluded = @(":(exclude)$Directory")
    # POSIX ERE (git grep -E); see Test-ToolDirectoryReferenced for why [:space:] and not \s.
    $asPath = '(^|["''[:space:]=;(,>]|\.\.[/\\])' + $name + '[/\\]'
    $null = & git grep -q -i -E $asPath HEAD -- '*.csproj' '*.props' '*.targets' '.github/test-shards.yml' @excluded 2>$null
    $consumed = $LASTEXITCODE -ne 1
    if (-not $consumed) {
        # Two passes: the regular expression over all of tests/ takes minutes, a fixed-string search under a
        # second, and only the few files that name the directory at all can match the expression.
        $naming = @(& git -c core.quotepath=false grep -l -i -F -e $Directory.TrimEnd('/') HEAD -- 'tests/' 2>$null)
        if ($LASTEXITCODE -gt 1) { $consumed = $true }
        elseif ($LASTEXITCODE -eq 0 -and $naming.Count -gt 0) {
            $inLiteral = '["''][^"''[:space:]]*' + $name + '[/\\]'
            $files = @($naming | ForEach-Object { ([string] $_) -replace '^HEAD:', '' })
            for ($i = 0; $i -lt $files.Count -and -not $consumed; $i += 100) {
                $batch = @($files[$i..([Math]::Min($i + 99, $files.Count - 1))])
                $null = & git --literal-pathspecs grep -q -i -E $inLiteral HEAD -- @batch 2>$null
                $consumed = $LASTEXITCODE -ne 1
            }
        }
        # A Path.Combine/Path.Join argument can sit on a later line than the call, and git grep matches single
        # lines, so this reads the candidate files whole.
        if (-not $consumed) { $consumed = Test-HeadTextJoinsDirectory -Name $Directory.TrimEnd('/') -Pathspec 'tests/' }
    }
    $script:ToolReferenceCache[$key] = $consumed
    return $consumed
}

<#
.SYNOPSIS
Whether any workflow, composite action, script or tool names a .github/ file.

.DESCRIPTION
Searched at HEAD for the file name in .github/ (workflows, actions, scripts and other config) and tools/. Names
shorter than six characters are too generic to search for and count as referenced, as does any failed search.
#>
function Test-GitHubFileReferenced {
    param([Parameter(Mandatory)] [string] $Path)
    $name = [IO.Path]::GetFileName($Path)
    if ($name.Length -lt 6) { return $true }
    $key = '.github:' + $name
    if ($script:ToolReferenceCache.ContainsKey($key)) { return $script:ToolReferenceCache[$key] }
    # Matches the name only where a path ends, so 'bug_report.yml' is not found inside 'old_bug_report.yml'.
    $pattern = '(^|[^A-Za-z0-9_.-])' + [regex]::Escape($name) + '($|[^A-Za-z0-9_.-])'
    # The file itself is left out: it names itself (CODEOWNERS lists /.github/CODEOWNERS), which is no reference. So is
    # this classifier: it names such files in its comments and self-test, and reads none of them.
    $null = & git grep -q -i -E $pattern HEAD -- '.github/' 'tools/' ":(exclude)$Path" ':(exclude)tools/TestImpact/Select-Shards.ps1' 2>$null
    $referenced = $LASTEXITCODE -ne 1
    $script:ToolReferenceCache[$key] = $referenced
    return $referenced
}

function Test-ScriptUsedOnlyByIndependentWorkflows {
    param([Parameter(Mandatory)] [string] $Path)
    # A .github/scripts file is full validation because a workflow step may run it. When every file
    # that names it is a workflow already reviewed as independent of this validation workflow (a
    # nightly, a deploy), no step of this workflow runs it and no shard can observe the change. Any
    # other referrer (this workflow, an action, another script that may be dot-sourced) keeps it full.
    # #2272 changed .github/scripts/Compare-IsaVerdicts.ps1, used only by the cross-ISA nightly, and
    # ran all 165 shards for it.
    if (-not $Path.StartsWith('.github/scripts/', [StringComparison]::OrdinalIgnoreCase)) { return $false }
    $name = [IO.Path]::GetFileName($Path)
    if ($name.Length -lt 6) { return $false }
    $pattern = '(^|[^A-Za-z0-9_.-])' + [regex]::Escape($name) + '($|[^A-Za-z0-9_.-])'
    $referrers = @(& git -c core.quotepath=false grep -l -i -E $pattern HEAD -- '.github/' 'tools/' ":(exclude)$Path" ':(exclude)tools/TestImpact/Select-Shards.ps1' 2>$null)
    if ($LASTEXITCODE -gt 1 -or $referrers.Count -eq 0) { return $false }
    foreach ($referrer in $referrers) {
        $file = ([string] $referrer) -replace '^HEAD:', ''
        if ($file -notin $script:NonRuntimeWorkflowPaths) { return $false }
    }
    return $true
}
<#
.SYNOPSIS
Fails when a directory trusted as non-runtime has started holding compilable code.

.DESCRIPTION
The entries in $script:NonRuntimeDirectories are trusted because they contain nothing the solution
builds. That is a fact about the tree today, not a law, and the cost of it silently ceasing to be
true is tests skipped on a change that needed them. So it is re-checked rather than remembered.
#>
function Assert-NonRuntimeDirectories {
    param([string] $Root = (Get-Location).Path)
    foreach ($entry in $script:NonRuntimeDirectories) {
        $directory = Join-Path $Root ($entry.TrimEnd('/'))
        if (-not (Test-Path -LiteralPath $directory)) { continue }
        $compilable = Get-ChildItem -LiteralPath $directory -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in @('.cs', '.csproj', '.fsproj', '.vbproj') } |
            Select-Object -First 1
        if ($compilable) {
            throw ("$entry is treated as non-runtime but now contains compilable code " +
                "($($compilable.FullName)); remove it from NonRuntimeDirectories or move the code.")
        }
    }
}

<#
.SYNOPSIS
Fails when a file under .github/workflows has not been classified, or a classification is contradicted by the tree.

.DESCRIPTION
Every workflow file belongs to exactly one list: selection control (SelectionControlPaths), pipeline
(PipelineWorkflowPaths) or independent (NonRuntimeWorkflowPaths). An unlisted file escalates, which is safe but
silent and permanent, and a file listed as independent that the pipeline calls would be skipped wrongly. So a new
or renamed workflow fails this check until someone reads it and lists it, and an independent workflow that a
pipeline workflow calls as a reusable workflow is rejected.
#>
function Assert-WorkflowsClassified {
    param([string] $Root = (Get-Location).Path)
    $directory = Join-Path $Root '.github/workflows'
    if (-not (Test-Path -LiteralPath $directory)) { return }
    $pipeline = @($script:SelectionControlPaths | Where-Object { $_.StartsWith('.github/workflows/', [StringComparison]::Ordinal) }) +
        @($script:PipelineWorkflowPaths)
    $unclassified = [System.Collections.Generic.List[string]]::new()
    foreach ($file in @(Get-ChildItem -LiteralPath $directory -File -Recurse | Sort-Object FullName)) {
        $relative = '.github/workflows/' + $file.FullName.Substring($directory.Length).TrimStart('\', '/').Replace('\', '/')
        $lists = @(@($relative -cin $pipeline), @($relative -cin $script:NonRuntimeWorkflowPaths) | Where-Object { $_ }).Count
        if ($lists -eq 0) { [void] $unclassified.Add($relative) }
        elseif ($lists -gt 1) { throw "$relative is listed both as a pipeline workflow and as an independent one." }
    }
    if ($unclassified.Count -gt 0) {
        throw ("unclassified workflow file(s): $($unclassified -join ', '). Read each one and add it to " +
            'NonRuntimeWorkflowPaths (independent of the test pipeline) or PipelineWorkflowPaths (takes part in it) in Select-Shards.ps1.')
    }
    foreach ($caller in $pipeline) {
        $path = Join-Path $Root $caller
        if (-not (Test-Path -LiteralPath $path)) { continue }
        foreach ($match in [regex]::Matches((Get-Content -LiteralPath $path -Raw), '(?m)^\s*(?:-\s*)?uses:\s*["'']?\./(\.github/workflows/[^\s"''@#]+)')) {
            $callee = $match.Groups[1].Value
            if ($callee -cin $script:NonRuntimeWorkflowPaths) {
                throw "$callee is listed as independent, but the pipeline workflow $caller calls it."
            }
        }
    }
}

# Directory.Packages.props, at the repository root only. A nested one is another central-package scope and keeps
# the shared-infrastructure rule.
$script:PackagePropsPath = 'Directory.Packages.props'
# Set once per run (Resolve-PackagePropsVerdict) when the change being selected for touches the file. $null, the
# default, leaves it full validation.
$script:PackagePropsVerdict = $null
# Packages whose version change always runs everything, whoever references them.
#   - AiDotNet.*: the tensor engine, its native kernels and the other ecosystem packages the product is built on.
#   - The test framework, test host and coverage collector: every test shard runs on them.
#   - The compiler, analyzer and build packages: they decide what is compiled, generated tests included.
$script:AlwaysFullPackagePatterns = @(
    '^AiDotNet\.',
    '^xunit', '^Microsoft\.NET\.Test\.Sdk$', '^Microsoft\.TestPlatform', '^Microsoft\.Testing\.', '^Microsoft\.CodeCoverage$',
    '^coverlet\.', '^MSTest', '^NUnit',
    '^Microsoft\.CodeAnalysis', '^Microsoft\.Build', '^Microsoft\.NET\.', '^Microsoft\.SourceLink'
)

function Read-PackageVersionFile {
    # The file as: each plain <PackageVersion Include Version/> entry, the entries that are anything else, and
    # everything that is left once the entries and the comments are taken out. $null when it is not XML.
    param([AllowNull()] [AllowEmptyString()] [string] $Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return $null }
    $xml = [xml]::new()
    try { $xml.LoadXml($Text.TrimStart([char] 0xFEFF)) } catch { return $null }
    $versions = [System.Collections.Generic.Dictionary[string, string]]::new([StringComparer]::OrdinalIgnoreCase)
    $unclear = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($node in @($xml.SelectNodes('//*[local-name()="PackageVersion"]'))) {
        $include = [string] $node.GetAttribute('Include')
        $version = [string] $node.GetAttribute('Version')
        $parent = $node.ParentNode
        $plain = $include -and $version -and $node.Attributes.Count -eq 2 -and -not $node.HasChildNodes -and
            $include -notmatch '[$@%;*]' -and $version -notmatch '[$@%;]' -and
            $parent.LocalName -ceq 'ItemGroup' -and $parent.Attributes.Count -eq 0 -and
            [object]::ReferenceEquals($parent.ParentNode, $xml.DocumentElement)
        $key = if ($include) { $include } else { $node.OuterXml }
        if (-not $plain -or $versions.ContainsKey($key)) { [void] $unclear.Add($key) }
        $versions[$key] = if ($plain) { $version } else { $node.OuterXml }
        [void] $parent.RemoveChild($node)
    }
    foreach ($comment in @($xml.SelectNodes('//comment()'))) { [void] $comment.ParentNode.RemoveChild($comment) }
    return [pscustomobject]@{ Versions = $versions; Unclear = $unclear; Residual = $xml.OuterXml }
}

function Get-PackageConsumerClass {
    # What a project that receives a package is, for routing: Product and Unknown run everything, Test runs that
    # project's shards, Inert is compiled (or not even that) and executed by no shard.
    param([Parameter(Mandatory)] [string] $Path)
    if ($Path.StartsWith('src/', [StringComparison]::OrdinalIgnoreCase)) { return 'Product' }
    if ($Path.StartsWith('tools/TestImpact/', [StringComparison]::OrdinalIgnoreCase)) { return 'Unknown' }
    if ($Path.StartsWith('tests/', [StringComparison]::OrdinalIgnoreCase)) { return 'Test' }
    foreach ($entry in $script:BuildOnlyDirectories) {
        if ($Path.StartsWith($entry, [StringComparison]::Ordinal)) {
            return $(if (Test-BuildOnlyDirectoryConsumed -Directory $entry) { 'Unknown' } else { 'Inert' })
        }
    }
    $top = $Path.Split('/')[0]
    if ($Path.Contains('/') -and -not $top.StartsWith('.') -and -not (Test-ToolDirectoryReferenced -Path $Path)) { return 'Inert' }
    return 'Unknown'
}

<#
.SYNOPSIS
Decides what a change to the root Directory.Packages.props can reach.

.DESCRIPTION
Pure: the two file texts, the reference lines of every MSBuild file and the tracked projects come in as arguments.
The answer is FullValidation unless every one of these is proved:
  - both versions parse, and nothing differs but the Version of plain <PackageVersion Include Version /> entries
    (an added or removed entry, a property, a condition, a central-package setting is not a version bump);
  - central transitive pinning is off, so an entry reaches only the projects that reference the package;
  - no changed package is on the always-full list (AlwaysFullPackagePatterns);
  - every reference to a changed package is a single-line <PackageReference Include="..."> in a .csproj, and there
    is at least one (a package nothing references, or one named in a shared props/targets file, has no bound);
  - no project that receives the package - the referencing projects and every project that depends on one of them
    through ProjectReference, followed transitively - is product code or of unknown standing.
Then: when all receiving projects are inert the change is BuildOnly, and when some are test projects with shards in
the manifest it is TestRoute, selecting every shard of those projects.

.PARAMETER Lines
Objects with File and Text: the lines of every MSBuild file that mention a reference, a package or a property used
as a ProjectReference target.
#>
function Get-PackagePropsVerdict {
    param(
        [AllowNull()] [AllowEmptyString()] [string] $BaseText,
        [AllowNull()] [AllowEmptyString()] [string] $HeadText,
        [AllowEmptyCollection()] [object[]] $Lines = @(),
        [AllowEmptyCollection()] [string[]] $Projects = @(),
        [AllowNull()] [object[]] $Manifest = $null,
        [scriptblock] $ClassifyProject = { param($Path) Get-PackageConsumerClass -Path $Path }
    )

    $packages = [System.Collections.Generic.List[object]]::new()
    function New-Verdict([string] $Impact, [string] $Why, [string[]] $Shards = @()) {
        return [pscustomobject]@{ Impact = $Impact; Why = $Why; Shards = @($Shards | Sort-Object -Unique); Packages = @($packages) }
    }
    $base = Read-PackageVersionFile $BaseText
    $head = Read-PackageVersionFile $HeadText
    if ($null -eq $base -or $null -eq $head) { return New-Verdict 'FullValidation' 'the file could not be read as XML on both sides of the change' }
    if ($base.Residual -cne $head.Residual) {
        return New-Verdict 'FullValidation' 'something other than a PackageVersion entry changed (a property, a condition or the file structure)'
    }
    if (@($Lines | Where-Object { $_.Text -match 'CentralPackageTransitivePinningEnabled' }).Count -gt 0) {
        return New-Verdict 'FullValidation' 'central transitive pinning is configured, so an entry can reach projects that do not reference the package'
    }
    $names = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in @($base.Versions.Keys) + @($head.Versions.Keys)) {
        if (-not $base.Versions.ContainsKey($name) -or -not $head.Versions.ContainsKey($name) -or
            $base.Versions[$name] -cne $head.Versions[$name]) { [void] $names.Add($name) }
    }
    if ($names.Count -eq 0) { return New-Verdict 'FullValidation' 'no PackageVersion entry changed, so the change cannot be attributed to a package' }

    # Project graph: who depends on whom. A reference that cannot be resolved makes its project a possible
    # dependent of anything.
    $projectByPath = [System.Collections.Generic.Dictionary[string, string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($project in $Projects) { $projectByPath[$project.Replace('\', '/')] = $project.Replace('\', '/') }
    $dependents = @{}
    $wild = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $sharedTargets = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $sharedWild = $false
    function Resolve-ReferenceTargets([string] $Referrer, [string] $Include, [int] $Depth = 0) {
        # The tracked projects an Include can name, as a list (PowerShell would unroll an empty array to $null);
        # $null when that cannot be bounded.
        $none = [System.Collections.Generic.List[string]]::new()
        $value = $Include.Trim().Replace('\', '/')
        if ($value -notmatch '\$\(') {
            $parts = [System.Collections.Generic.List[string]]::new()
            foreach ($segment in @((Split-Path -Parent $Referrer).Replace('\', '/').Split('/', [StringSplitOptions]::RemoveEmptyEntries)) + @($value.Split('/'))) {
                if ($segment -eq '.' -or $segment -eq '') { continue }
                if ($segment -eq '..') { if ($parts.Count -eq 0) { return , $none }; $parts.RemoveAt($parts.Count - 1); continue }
                [void] $parts.Add($segment)
            }
            $resolved = $parts -join '/'
            if ($projectByPath.ContainsKey($resolved)) { [void] $none.Add($projectByPath[$resolved]) }
            return , $none   # empty: outside the repository, or not a tracked project
        }
        $leaf = $value.Substring($value.LastIndexOf('/') + 1)
        if ($leaf -notmatch '\$\(' -and $value.Contains('/')) {
            foreach ($candidate in $projectByPath.Values) { if ([IO.Path]::GetFileName($candidate) -ieq $leaf) { [void] $none.Add($candidate) } }
            return , $none
        }
        if ($Depth -ge 3 -or $value -notmatch '^\$\(([A-Za-z_][A-Za-z0-9_.]*)\)$') { return $null }
        $property = [regex]::Escape($Matches[1])
        $definitions = @($Lines | ForEach-Object { [regex]::Matches([string] $_.Text, "<$property\b[^>]*>([^<]+)</$property>") } |
            ForEach-Object { $_.Groups[1].Value })
        if ($definitions.Count -eq 0) { return $null }
        foreach ($definition in $definitions) {
            $resolved = Resolve-ReferenceTargets $Referrer $definition ($Depth + 1)
            if ($null -eq $resolved) { return $null }
            foreach ($target in $resolved) { [void] $none.Add($target) }
        }
        return , $none
    }
    foreach ($line in $Lines) {
        $text = [string] $line.Text
        if ($text -notmatch '<ProjectReference\b') { continue }
        $file = ([string] $line.File).Replace('\', '/')
        $isProject = $file.EndsWith('.csproj', [StringComparison]::OrdinalIgnoreCase)
        # Each ProjectReference element's own Include, in either quote style: a line can also hold other reference
        # kinds, whose Include is not a project target. One unresolvable element makes the line's targets unknown.
        $targets = [System.Collections.Generic.List[string]]::new()
        foreach ($element in [regex]::Matches($text, '<ProjectReference\b[^>]*>')) {
            $include = [regex]::Match($element.Value, '\bInclude\s*=\s*(?:"(?<v>[^"]+)"|''(?<v>[^'']+)'')')
            $resolved = if ($include.Success) { Resolve-ReferenceTargets $file $include.Groups['v'].Value } else { $null }
            if ($null -eq $resolved) { $targets = $null; break }
            foreach ($target in $resolved) { [void] $targets.Add($target) }
        }
        if (-not $isProject) {
            # A shared props/targets file gives the reference to an unknown set of projects.
            if ($null -eq $targets) { $sharedWild = $true } else { foreach ($target in $targets) { [void] $sharedTargets.Add($target) } }
            continue
        }
        if ($null -eq $targets) { [void] $wild.Add($file); continue }
        foreach ($target in $targets) {
            if (-not $dependents.ContainsKey($target)) { $dependents[$target] = [System.Collections.Generic.List[string]]::new() }
            [void] $dependents[$target].Add($file)
        }
    }

    $routed = [System.Collections.Generic.List[string]]::new()
    $anyTest = $false
    foreach ($name in $names) {
        $from = if ($base.Versions.ContainsKey($name)) { $base.Versions[$name] } else { '' }
        $to = if ($head.Versions.ContainsKey($name)) { $head.Versions[$name] } else { '' }
        $record = [pscustomobject]@{ name = $name; from = $from; to = $to; reach = 'full'; why = '' }
        [void] $packages.Add($record)
        $full = { param($why) $record.why = $why; New-Verdict 'FullValidation' "$name`: $why" }
        if (-not $from -or -not $to) { return & $full 'the entry was added or removed, so who consumes it is not established' }
        if ($base.Unclear.Contains($name) -or $head.Unclear.Contains($name)) {
            return & $full 'the entry is not a plain Include/Version pair in an unconditional item group, or it is declared twice'
        }
        foreach ($pattern in $script:AlwaysFullPackagePatterns) {
            if ($name -imatch $pattern) { return & $full 'the product, every test host or the compiler depends on this package family' }
        }
        # The name as an XML attribute value in either quote style, so Include='X' is a consumer as Include="X" is.
        $quoted = '(["''])' + [regex]::Escape($name) + '\1'
        $reference = '<PackageReference\b[^>]*\bInclude\s*=\s*(["''])' + [regex]::Escape($name) + '\1'
        $consumers = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($line in $Lines) {
            $text = [string] $line.Text
            $file = ([string] $line.File).Replace('\', '/')
            if ($file -ceq $script:PackagePropsPath) { continue }
            if ($text -match '<PackageReference\b' -and $text -notmatch '\b(Include|Update)\s*=') {
                return & $full "$file declares a PackageReference across several lines, so its package cannot be read"
            }
            if ($text -inotmatch $quoted) { continue }
            if (-not $file.EndsWith('.csproj', [StringComparison]::OrdinalIgnoreCase)) {
                return & $full "$file names it, and a shared MSBuild file applies to an unknown set of projects"
            }
            if ($text -inotmatch $reference) { return & $full "$file refers to it in a form other than <PackageReference Include>" }
            if (-not $projectByPath.ContainsKey($file)) { return & $full "$file references it but is not a tracked project" }
            [void] $consumers.Add($projectByPath[$file])
        }
        if ($consumers.Count -eq 0) { return & $full 'no project references it directly, so what it reaches cannot be bounded' }
        if ($sharedWild) { return & $full 'a shared MSBuild file adds a project reference that cannot be resolved' }

        $closure = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        $queue = [System.Collections.Generic.Queue[string]]::new()
        foreach ($project in @($consumers) + @($wild)) { if ($closure.Add($project)) { $queue.Enqueue($project) } }
        while ($queue.Count -gt 0) {
            $project = $queue.Dequeue()
            if ($dependents.ContainsKey($project)) {
                foreach ($dependent in $dependents[$project]) { if ($closure.Add($dependent)) { $queue.Enqueue($dependent) } }
            }
        }
        $testProjects = [System.Collections.Generic.List[string]]::new()
        foreach ($project in ($closure | Sort-Object)) {
            if ($sharedTargets.Contains($project)) { return & $full "$project receives it, and a shared MSBuild file references that project for an unknown set of projects" }
            $class = [string] (& $ClassifyProject $project)
            if ($class -ceq 'Inert') { continue }
            if ($class -cne 'Test') { return & $full "$project receives it and is product code or of unknown standing" }
            $shards = @(if ($null -ne $Manifest) {
                $Manifest | Where-Object { $_.PSObject.Properties['project'] -and ([string] $_.project).Replace('\', '/') -ieq $project } |
                    ForEach-Object { [string] $_.name }
            })
            if ($shards.Count -eq 0) { return & $full "test project $project receives it and no shard in the manifest is known to run that project" }
            [void] $testProjects.Add($project)
            foreach ($shard in $shards) { [void] $routed.Add($shard) }
        }
        if ($testProjects.Count -gt 0) {
            $anyTest = $true
            $record.reach = 'test projects'
            $record.why = "received only by $($testProjects -join ', ') and projects no shard executes"
        }
        else {
            $record.reach = 'build-only'
            $record.why = "received only by projects no shard executes: $(($closure | Sort-Object) -join ', ')"
        }
    }
    if ($anyTest) {
        return New-Verdict 'TestRoute' ("package version change ($($names -join ', ')) reaches only test projects; every shard of those projects runs") @($routed)
    }
    return New-Verdict 'BuildOnly' "package version change ($($names -join ', ')) reaches only projects no shard executes"
}

function Resolve-PackagePropsVerdict {
    # Get-PackagePropsVerdict over the checkout: the file at the base and at HEAD, and HEAD's MSBuild files.
    # Any failed read is full validation.
    param([Parameter(Mandatory)] [string] $BaseSha, [AllowNull()] [object[]] $Manifest = $null)
    $full = { param($why) [pscustomobject]@{ Impact = 'FullValidation'; Why = $why; Shards = @(); Packages = @() } }
    try {
        $baseText = @(& git show "${BaseSha}:$script:PackagePropsPath" 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0) { return & $full 'the file does not exist at the base of the change' }
        $headText = @(& git show "HEAD:$script:PackagePropsPath" 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0) { return & $full 'the file does not exist at HEAD' }
        $buildFiles = @('*.csproj', '*.props', '*.targets', '*.proj', '*.fsproj', '*.vbproj')
        $read = {
            param([string] $Pattern)
            $found = @(& git -c core.quotepath=false grep -n -I -E $Pattern HEAD -- @buildFiles 2>$null)
            if ($LASTEXITCODE -gt 1) { throw 'the MSBuild files could not be searched' }
            foreach ($entry in $found) {
                if ([string] $entry -match '^HEAD:(.+?):\d+:(.*)$') { [pscustomobject]@{ File = $Matches[1]; Text = $Matches[2] } }
            }
        }
        $lines = @(& $read '<[A-Za-z]*(Reference|Package)[A-Za-z]*([[:space:]>/]|$)')
        $properties = @($lines | Where-Object { $_.Text -match '<ProjectReference\b' } |
            ForEach-Object { [regex]::Matches([string] $_.Text, '\$\(([A-Za-z_][A-Za-z0-9_.]*)\)') } |
            ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
        # Twice: a property that names a project may itself be written in terms of another one.
        for ($pass = 0; $pass -lt 2 -and $properties.Count -gt 0; $pass++) {
            $definitions = @(& $read ('<(' + (($properties | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')([[:space:]>])'))
            $lines += $definitions
            $properties = @($definitions | ForEach-Object { [regex]::Matches([string] $_.Text, '\$\(([A-Za-z_][A-Za-z0-9_.]*)\)') } |
                ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
        }
        $projects = @(& git -c core.quotepath=false ls-tree -r --name-only HEAD 2>$null | Where-Object { $_ -like '*.csproj' })
        if ($LASTEXITCODE -ne 0) { return & $full 'the tracked projects could not be listed' }
        return Get-PackagePropsVerdict -BaseText $baseText -HeadText $headText -Lines $lines -Projects $projects -Manifest $Manifest
    }
    catch { return & $full "package routing failed: $($_.Exception.Message)" }
    finally { $global:LASTEXITCODE = 0 }
}

function Test-RangeOverlap {
    param([int] $AStart, [int] $AEnd, [int] $BStart, [int] $BEnd)
    return -not ($AEnd -lt $BStart -or $BEnd -lt $AStart)
}

function ConvertTo-ValidatedInteger {
    param(
        [Parameter(Mandatory)] $Value,
        [Parameter(Mandatory)] [string] $What,
        [int] $Minimum = 0,
        [int] $Maximum = [int]::MaxValue
    )

    $isInteger = $Value -is [byte] -or $Value -is [sbyte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
    if (-not $isInteger) { throw "$What must be an integer." }

    $number = [long] $Value
    if ($number -lt $Minimum -or $number -gt $Maximum) {
        throw "$What must be between $Minimum and $Maximum."
    }
    return [int] $number
}

function Assert-ShardMap {
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [string[]] $Expected
    )

    foreach ($required in 'schemaVersion', 'sha', 'knownShards', 'alwaysRun', 'files') {
        if (-not $Map.PSObject.Properties[$required]) { throw "no '$required' property" }
    }
    $schemaVersion = ConvertTo-ValidatedInteger -Value $Map.schemaVersion -What 'schemaVersion' -Minimum 1
    if ($schemaVersion -ne 1) { throw "unsupported schemaVersion $schemaVersion" }
    if ([string] $Map.sha -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
        throw 'sha must be a complete hexadecimal Git object id'
    }
    if ($Map.knownShards -isnot [array] -or $Map.alwaysRun -isnot [array]) {
        throw 'knownShards and alwaysRun must be arrays'
    }
    if ($Map.files -isnot [pscustomobject]) { throw 'files must be an object' }

    $expectedSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in @($Expected)) {
        if ([string]::IsNullOrWhiteSpace([string] $name)) { throw 'expected shard names must be nonempty' }
        if (-not $expectedSet.Add([string] $name)) { throw "duplicate expected shard '$name'" }
    }
    if ($expectedSet.Count -eq 0) { throw 'the expected shard manifest is empty' }

    $known = @($Map.knownShards)
    $always = @($Map.alwaysRun)
    if ($known.Count -eq 0) { throw 'knownShards is empty' }

    $mapSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($group in @(@{ Name = 'knownShards'; Values = $known }, @{ Name = 'alwaysRun'; Values = $always })) {
        foreach ($nameValue in $group.Values) {
            $name = [string] $nameValue
            if ([string]::IsNullOrWhiteSpace($name)) { throw "$($group.Name) contains an empty shard name" }
            if (-not $mapSet.Add($name)) { throw "duplicate or overlapping shard '$name'" }
        }
    }

    # A map shard the manifest no longer has is a removal or a rename: the map's coverage describes
    # a matrix that no longer exists, and selecting from it could name a job that cannot run.
    # The opposite direction is not a contradiction, only a gap: see Add-UnmappedShardsAsAlwaysRun.
    $extra = @($mapSet | Where-Object { -not $expectedSet.Contains($_) } | Sort-Object)
    if ($extra.Count -gt 0) {
        throw "map names shard(s) the manifest does not have: $($extra -join ', ')"
    }

    $fileProperties = @($Map.files.PSObject.Properties)
    if ($fileProperties.Count -eq 0) { throw 'files index is empty' }
    if ($Map.PSObject.Properties['fileCount']) {
        $fileCount = ConvertTo-ValidatedInteger -Value $Map.fileCount -What 'fileCount'
        if ($fileCount -ne $fileProperties.Count) { throw 'fileCount does not match the files index' }
    }
    foreach ($fileProperty in $fileProperties) {
        if ([string]::IsNullOrWhiteSpace([string] $fileProperty.Name)) { throw 'files contains an empty path' }
        if ($fileProperty.Value -isnot [array]) { throw "'$($fileProperty.Name)' occurrences must be an array" }
        $occurrences = @($fileProperty.Value)
        if ($occurrences.Count -eq 0) { throw "'$($fileProperty.Name)' has no occurrences" }
        foreach ($occurrence in $occurrences) {
            if (-not $occurrence.PSObject.Properties['s'] -or -not $occurrence.PSObject.Properties['r']) {
                throw "'$($fileProperty.Name)' has an occurrence without s or r"
            }
            [void] (ConvertTo-ValidatedInteger -Value $occurrence.s -What "'$($fileProperty.Name)' shard index" `
                -Minimum 0 -Maximum ($known.Count - 1))
            if ($occurrence.r -isnot [array]) { throw "'$($fileProperty.Name)' ranges must be an array" }
            $ranges = @($occurrence.r)
            if ($ranges.Count -eq 0 -or $ranges.Count % 2 -ne 0) {
                throw "'$($fileProperty.Name)' ranges must be nonempty start/end pairs"
            }
            for ($i = 0; $i -lt $ranges.Count; $i += 2) {
                $start = ConvertTo-ValidatedInteger -Value $ranges[$i] -What "'$($fileProperty.Name)' range start" -Minimum 1
                $end = ConvertTo-ValidatedInteger -Value $ranges[$i + 1] -What "'$($fileProperty.Name)' range end" -Minimum 1
                if ($start -gt $end) { throw "'$($fileProperty.Name)' range start $start exceeds end $end" }
            }
        }
    }
}

function ConvertTo-DiffHunks {
    <#
        Parses zero-context hunks into { OldStart, OldCount, NewStart, NewCount } per file, keyed by
        the new path (the old one for a deletion).

        Hunk BODY lines are counted, and headers are only recognised while none is pending, because
        diff body lines are raw file content behind a one-character prefix, and content can forge any
        header: a REMOVED line whose text begins with '-- ' is rendered '--- ...', byte-identical to
        an old-file header. Reproduced with real git: deleting the line '-- remove me' emitted
        '--- remove me', the old parser took it as a header, nulled the current file, and silently
        dropped every later hunk of that file - under-selection with no escalation. A zero-context
        hunk '@@ -a,n +b,m @@' is followed by exactly n+m body lines (plus uncounted '\ No newline'
        markers), so counting them makes body content inert no matter what it says.
    #>
    param([AllowEmptyCollection()] [string[]] $DiffLines)

    $hunks = @{}
    $current = $null
    $deletedPath = $null
    $pendingBody = 0
    foreach ($line in $DiffLines) {
        if ($pendingBody -gt 0) {
            if (-not $line.StartsWith('\')) { $pendingBody-- }
            continue
        }
        if ($line.StartsWith('--- ')) {
            $from = $line.Substring(4).Trim()
            $deletedPath = if ($from -eq '/dev/null') { $null } else { $from -replace '^a/', '' }
            $current = $null
        }
        elseif ($line.StartsWith('+++ ')) {
            $to = $line.Substring(4).Trim()
            $current = if ($to -eq '/dev/null') { $deletedPath } else { $to -replace '^b/', '' }
        }
        elseif ($line.StartsWith('@@') -and $line -match '^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@') {
            $oldStart = [int] $Matches[1]
            $oldCount = if ($Matches[2]) { [int] $Matches[2] } else { 1 }
            $newStart = [int] $Matches[3]
            $newCount = if ($Matches[4]) { [int] $Matches[4] } else { 1 }
            $pendingBody = $oldCount + $newCount
            if ($current) {
                if (-not $hunks.ContainsKey($current)) {
                    $hunks[$current] = [System.Collections.Generic.List[object]]::new()
                }
                [void] $hunks[$current].Add([int[]] @($oldStart, $oldCount, $newStart, $newCount))
            }
        }
    }
    return $hunks
}

function ConvertTo-OldSideRange {
    # One hunk's old side as a start/end pair; a pure insertion is the two old lines it sits between.
    param([Parameter(Mandatory)] [int[]] $Hunk)
    if ($Hunk[1] -gt 0) { return , ([int[]] @($Hunk[0], ($Hunk[0] + $Hunk[1] - 1))) }
    return , ([int[]] @([Math]::Max(1, $Hunk[0]), [Math]::Max(1, $Hunk[0] + 1)))
}

function New-ChangeDetail {
    <#
        What the precise fallbacks need about one C# file's change, beside its flat range list: each
        hunk's old AND new side (the new side is otherwise dropped), the map-numbered ranges that
        hunk became, the revision the old side is numbered in (Base), the map's own revision, and
        the hunks from the map to Base. Swept marks a file whose ranges also carry everything the
        base branch changed since the map, which no hunk of this change accounts for.
    #>
    param(
        [Parameter(Mandatory)] [string] $Base,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Hunks,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $MapHunks,
        [string] $MapRevision = $Base,
        [bool] $Swept
    )
    return [pscustomobject]@{
        Base        = $Base
        MapRevision = $MapRevision
        MapHunks    = $MapHunks
        Swept       = $Swept
        Hunks    = @(foreach ($hunk in $Hunks) {
            [pscustomobject]@{
                Hunk = [int[]] $hunk
                Map  = [int[]] (Convert-RangesThroughHunks -Ranges (ConvertTo-OldSideRange -Hunk $hunk) -Hunks $MapHunks)
            }
        })
    }
}

function ConvertTo-ChangedRanges {
    <#
        The OLD side of each zero-context hunk as flat start/end pairs. A pure insertion has no old
        lines, so it is recorded as the two old lines it sits between. ChangedFiles is authoritative:
        rename-only, binary and mode-only changes do not necessarily have an @@ header, but they must
        still reach the fail-safe selector.
    #>
    param(
        [AllowEmptyCollection()] [string[]] $DiffLines,
        [AllowEmptyCollection()] [string[]] $ChangedFiles = @()
    )

    $changed = @{}
    $hunks = ConvertTo-DiffHunks -DiffLines $DiffLines
    foreach ($path in $hunks.Keys) {
        $ranges = [System.Collections.Generic.List[int]]::new()
        foreach ($hunk in $hunks[$path]) { $ranges.AddRange([int[]] (ConvertTo-OldSideRange -Hunk $hunk)) }
        $changed[$path] = $ranges
    }

    foreach ($path in $ChangedFiles) {
        if (-not [string]::IsNullOrWhiteSpace($path) -and -not $changed.ContainsKey($path)) {
            $changed[$path] = [System.Collections.Generic.List[int]]::new()
        }
    }
    return $changed
}

function Convert-RangesThroughHunks {
    <#
        Carries line ranges expressed in a LATER version of a file back to an EARLIER one, through
        the zero-context hunks of earlier -> later (old = earlier, new = later).

        Selection needs a change's lines in the map commit's numbering. For a pull request that is
        behind master, or a landed commit measured against the tree its pull request validated, the
        change is naturally expressed against a base the map does not describe; diffing the map
        against HEAD instead would sweep in every edit master made to the same file since the map.
        This carries the change's own ranges back instead.

        It never narrows what a line can affect: a later line that is unchanged since the earlier
        version maps to its exact earlier line; one inside a changed region maps to the WHOLE earlier
        region it replaced (or, where nothing was replaced, the two earlier lines it sits between);
        and a range touching the spot where earlier lines were deleted also takes the deleted lines.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [int[]] $Ranges,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Hunks
    )

    if ($Hunks.Count -eq 0) { return , ([int[]] $Ranges) }

    # Alternating segments: Same (a later run of lines with an exact earlier counterpart) and
    # Changed (a later run, possibly empty, that replaced an earlier run, possibly empty).
    $segments = [System.Collections.Generic.List[object]]::new()
    $oldConsumed = 0
    $newConsumed = 0
    foreach ($hunk in @($Hunks | Sort-Object { $_[0] }, { $_[2] })) {
        $oldStart, $oldCount, $newStart, $newCount = $hunk
        $oldBefore = if ($oldCount -gt 0) { $oldStart - 1 } else { $oldStart }
        $newBefore = if ($newCount -gt 0) { $newStart - 1 } else { $newStart }
        $same = $oldBefore - $oldConsumed
        if ($same -ne ($newBefore - $newConsumed)) {
            throw 'inconsistent diff hunks: unchanged runs differ in length'
        }
        if ($same -gt 0) {
            [void] $segments.Add([pscustomobject]@{ Kind = 'Same'; NewFrom = $newConsumed + 1; NewTo = $newBefore; Offset = $oldConsumed - $newConsumed })
        }
        [void] $segments.Add([pscustomobject]@{
            Kind = 'Changed'; NewFrom = $newStart; NewTo = $newStart + $newCount - 1
            OldFrom = $oldStart; OldTo = $oldStart + $oldCount - 1; NewCount = $newCount; OldCount = $oldCount
        })
        $oldConsumed = $oldBefore + $oldCount
        $newConsumed = $newBefore + $newCount
    }
    [void] $segments.Add([pscustomobject]@{ Kind = 'Same'; NewFrom = $newConsumed + 1; NewTo = [int]::MaxValue; Offset = $oldConsumed - $newConsumed })

    $result = [System.Collections.Generic.List[int]]::new()
    for ($i = 0; $i + 1 -lt $Ranges.Count; $i += 2) {
        $from = [int] $Ranges[$i]
        $to = [int] $Ranges[$i + 1]
        foreach ($segment in $segments) {
            if ($segment.Kind -eq 'Same') {
                $a = [Math]::Max($from, $segment.NewFrom)
                $b = [Math]::Min($to, $segment.NewTo)
                if ($a -le $b) {
                    [void] $result.Add($a + $segment.Offset)
                    [void] $result.Add($b + $segment.Offset)
                }
                continue
            }
            # A Changed segment with later lines is hit when the range overlaps them. One with none
            # is a deletion: '@@ -a,n +b,0 @@' removed earlier lines from between later lines b and
            # b + 1, so it is hit when the range touches either of those two neighbours.
            $hit = if ($segment.NewCount -gt 0) {
                $from -le $segment.NewTo -and $to -ge $segment.NewFrom
            } else {
                $from -le $segment.NewFrom + 1 -and $to -ge $segment.NewFrom
            }
            if (-not $hit) { continue }
            if ($segment.OldCount -gt 0) {
                [void] $result.Add($segment.OldFrom)
                [void] $result.Add($segment.OldTo)
            }
            else {
                [void] $result.Add([Math]::Max(1, $segment.OldFrom))
                [void] $result.Add([Math]::Max(1, $segment.OldFrom + 1))
            }
        }
    }
    return , ([int[]] $result.ToArray())
}

function Test-RangesTouchIntroducedLines {
    <#
        Whether any range reaches later lines that did not exist in the earlier version (the new
        side of a hunk). Such a line's earlier home is unknown from the diff alone: if the base
        branch MOVED code since the map, the moved lines appear as fresh insertions, and carrying a
        change to them back would land on the insertion point instead of the code's mapped lines.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [int[]] $Ranges,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Hunks
    )
    foreach ($hunk in $Hunks) {
        if ($hunk[3] -le 0) { continue }
        $newFrom = $hunk[2]
        $newTo = $hunk[2] + $hunk[3] - 1
        for ($i = 0; $i + 1 -lt $Ranges.Count; $i += 2) {
            if ($Ranges[$i] -le $newTo -and $Ranges[$i + 1] -ge $newFrom) { return $true }
        }
    }
    return $false
}

function Get-ScopedChangedRanges {
    <#
        The change from Base to HEAD, for exactly Paths, in the map commit's line numbers: the
        change's own old-side ranges, carried back through map -> Base. Base may be a commit or a
        tree (post-merge delta reuse compares against a rebuilt tree).

        Detail, when given, receives a New-ChangeDetail record per changed C# file.

        OwnDiff, when given, is the Base -> HEAD diff for exactly Paths, already taken (Get-ScopedChangeDiff).
        A rolling map translates one change to several commits; the change itself is the same each time, so
        only the MapSha -> Base half is taken again.
    #>
    param(
        [Parameter(Mandatory)] [string] $MapSha,
        [Parameter(Mandatory)] [string] $Base,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Paths,
        [hashtable] $Detail,
        [AllowEmptyCollection()] [string[]] $OwnDiff
    )

    $changed = @{}
    if ($Paths.Count -eq 0) { return $changed }
    if ($PSBoundParameters.ContainsKey('OwnDiff')) { $changeDiff = @($OwnDiff) }
    else {
        $changeDiff = @(& git -c core.quotepath=false --literal-pathspecs diff --no-renames --no-ext-diff -U0 $Base HEAD -- @Paths)
        if ($LASTEXITCODE -ne 0) { throw "git diff from '$Base' failed" }
    }
    $ownRanges = ConvertTo-ChangedRanges -DiffLines $changeDiff -ChangedFiles $Paths
    $mapDiff = @(& git -c core.quotepath=false --literal-pathspecs diff --no-renames --no-ext-diff -U0 $MapSha $Base -- @Paths)
    if ($LASTEXITCODE -ne 0) { throw "git diff from the map commit '$MapSha' to '$Base' failed" }
    $mapHunks = ConvertTo-DiffHunks -DiffLines $mapDiff
    $ownHunks = if ($null -ne $Detail) { ConvertTo-DiffHunks -DiffLines $changeDiff } else { @{} }

    $sweep = [System.Collections.Generic.List[string]]::new()
    foreach ($path in $ownRanges.Keys) {
        $ranges = [int[]] @($ownRanges[$path])
        # Two statements: an if-expression yielding @() unrolls to $null.
        $hunks = @()
        if ($mapHunks.ContainsKey($path)) { $hunks = @($mapHunks[$path]) }
        $changed[$path] = [System.Collections.Generic.List[int]]::new([int[]] (Convert-RangesThroughHunks -Ranges $ranges -Hunks $hunks))
        $swept = Test-RangesTouchIntroducedLines -Ranges $ranges -Hunks $hunks
        if ($swept) { [void] $sweep.Add($path) }
        if ($ownHunks.ContainsKey($path) -and ([string] $path).EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase)) {
            $Detail[$path] = New-ChangeDetail -Base $Base -MapRevision $MapSha -Hunks @($ownHunks[$path]) -MapHunks $hunks -Swept $swept
        }
    }

    # Where the change reaches lines the base branch introduced since the map, their mapped origin
    # is unknowable from this diff, so add back everything map -> HEAD changed in that file: the
    # original, broader behaviour, which sees a move as the deletion of the mapped lines.
    if ($sweep.Count -gt 0) {
        $sweepDiff = @(& git -c core.quotepath=false --literal-pathspecs diff --no-renames --no-ext-diff -U0 $MapSha HEAD -- @($sweep))
        if ($LASTEXITCODE -ne 0) { throw "git diff from the map commit '$MapSha' to HEAD failed" }
        $sweepRanges = ConvertTo-ChangedRanges -DiffLines $sweepDiff
        foreach ($path in $sweep) {
            if ($sweepRanges.ContainsKey($path)) { $changed[$path].AddRange([int[]] @($sweepRanges[$path])) }
        }
    }
    return $changed
}

function Get-ScopedChangeDiff {
    # The Base -> HEAD half of Get-ScopedChangedRanges, for callers that translate one change to several commits.
    param([Parameter(Mandatory)] [string] $Base, [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Paths)
    if ($Paths.Count -eq 0) { return @() }
    $diff = @(& git -c core.quotepath=false --literal-pathspecs diff --no-renames --no-ext-diff -U0 $Base HEAD -- @Paths)
    if ($LASTEXITCODE -ne 0) { throw "git diff from '$Base' failed" }
    return $diff
}

function Get-ChangedRanges {
    # Detail, when given, receives a New-ChangeDetail record per changed C# file (the map is the base).
    param([string] $MapSha, [hashtable] $Detail)

    $changedFiles = @(& git -c core.quotepath=false diff --no-renames --name-only $MapSha HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$MapSha' failed" }
    $diff = @(& git -c core.quotepath=false diff --no-renames --no-ext-diff -U0 $MapSha HEAD --)
    if ($LASTEXITCODE -ne 0) { throw "git diff from '$MapSha' failed" }
    if ($null -ne $Detail) {
        $hunks = ConvertTo-DiffHunks -DiffLines $diff
        foreach ($path in $hunks.Keys) {
            if (([string] $path).EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase)) {
                $Detail[$path] = New-ChangeDetail -Base $MapSha -Hunks @($hunks[$path]) -MapHunks @() -Swept $false
            }
        }
    }
    return ConvertTo-ChangedRanges -DiffLines $diff -ChangedFiles $changedFiles
}

function New-DirectoryOwnerIndex {
    param([Parameter(Mandatory)] $Map)

    $index = [System.Collections.Generic.Dictionary[string, System.Collections.Generic.SortedSet[string]]]::new(
        [StringComparer]::OrdinalIgnoreCase)
    foreach ($property in $Map.files.PSObject.Properties) {
        $segments = @($property.Name.Replace('\', '/').Split('/'))
        for ($depth = $segments.Count - 1; $depth -ge 2; $depth--) {
            $directory = ($segments[0..($depth - 1)] -join '/') + '/'
            if (-not $index.ContainsKey($directory)) {
                $index[$directory] = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
            }
            foreach ($occurrence in @($property.Value)) {
                [void] $index[$directory].Add([string] $Map.knownShards[[int] $occurrence.s])
            }
        }
    }
    return ,$index
}

function Get-DirectoryOwners {
    <#
        The shards that execute any mapped file in the nearest directory of Path that has one, never
        climbing above a two-segment root such as 'src/Finance'. A brand-new source file has no
        coverage of its own, but the code beside it does; climbing to 'src/' itself would stop
        meaning anything, so an orphan with no mapped neighbour below that depth still escalates.
    #>
    param(
        [Parameter(Mandatory)]
        [System.Collections.Generic.Dictionary[string, System.Collections.Generic.SortedSet[string]]] $Index,
        [Parameter(Mandatory)] [string] $Path
    )

    $segments = @(([string] $Path).Replace('\', '/').Split('/'))
    for ($depth = $segments.Count - 1; $depth -ge 2; $depth--) {
        $directory = ($segments[0..($depth - 1)] -join '/') + '/'
        if ($Index.ContainsKey($directory) -and $Index[$directory].Count -gt 0) {
            return [pscustomobject]@{ Directory = $directory.TrimEnd('/'); Shards = @($Index[$directory]) }
        }
    }
    return $null
}

# ------------------------------------------- precise fallbacks: C# member spans

# Kill switch for the precise fallbacks below. Unset or anything else leaves them on; '0', 'false',
# 'off', 'no' or 'disabled' restores the two file-wide rules exactly as they were (every shard that
# executes the file for a changed range no shard executes), with no member index consulted at all.
# The Select shards job passes the repository variable TEST_IMPACT_PRECISE_FALLBACKS through, so
# it can be turned off from the repository settings without a revert.
function Test-PreciseFallbacksEnabled {
    return ([string] $env:TEST_IMPACT_PRECISE_FALLBACKS) -notmatch '^\s*(0|false|off|no|disabled)\s*$'
}

$script:CSharpSyntaxAvailable = $null
$script:CSharpSyntaxProblem = ''

function Initialize-CSharpSyntax {
    <#
        Loads the C# parser PowerShell itself ships (Add-Type compiles with it, so every pwsh has
        Microsoft.CodeAnalysis.CSharp beside it, built for the runtime that pwsh runs on). Parsing
        only: no SDK, no build, no restore, and nothing the Select shards job does not already have.
        The SDK's own copy is built for the SDK's runtime, which an older pwsh cannot load.

        Any failure leaves the parser unavailable, and every caller then keeps the file-wide rule.
    #>
    if ($null -ne $script:CSharpSyntaxAvailable) { return $script:CSharpSyntaxAvailable }
    $script:CSharpSyntaxAvailable = $false
    try {
        if (-not ('Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree' -as [type])) {
            foreach ($name in @('Microsoft.CodeAnalysis.dll', 'Microsoft.CodeAnalysis.CSharp.dll')) {
                [void] [System.Reflection.Assembly]::LoadFrom((Join-Path $PSHOME $name))
            }
        }
        # A known answer, so an API that loads but does not behave as this code expects is also a failure.
        $script:CSharpSyntaxAvailable = $true
        $probe = Get-CSharpSyntaxModel -Text "class Probe`n{`n    void Run() { }`n}"
        $ok = $null -ne $probe
        if ($ok) {
            $side = Get-CSharpChangeSide -Model $probe -StartLine 3 -Count 1
            $ok = $null -eq $side.Blocker -and $side.Tokens.Count -eq 6 -and
                (Get-CSharpTokenContext -Token $side.Tokens[4]).Level -ceq 'Member' -and
                (Get-CSharpTokenContext -Token $side.Tokens[1]).Level -ceq 'Type'
        }
        if (-not $ok) { throw 'the parser did not classify a known method body' }
    }
    catch {
        $script:CSharpSyntaxAvailable = $false
        $script:CSharpSyntaxProblem = $_.Exception.Message
    }
    return $script:CSharpSyntaxAvailable
}

function Get-CSharpSyntaxModel {
    <#
        A parsed file, or $null when it cannot be trusted: no parser, no text, any syntax error, or
        a line break the parser counts and git does not (a bare carriage return or a Unicode line
        separator), which would shift every line number after it.

        No preprocessor symbols are defined, so code inside '#if SYMBOL' is disabled text; a change
        there is never classified (Get-CSharpChangeSide reports it as a blocker).
    #>
    param([AllowNull()] $Text)

    if ($null -eq $Text -or -not (Initialize-CSharpSyntax)) { return $null }
    $source = [string] $Text
    if ($source.IndexOfAny([char[]] @([char] 0x0D, [char] 0x85, [char] 0x2028, [char] 0x2029)) -ge 0) { return $null }
    try {
        $options = [Microsoft.CodeAnalysis.CSharp.CSharpParseOptions]::Default.WithLanguageVersion(
            [Microsoft.CodeAnalysis.CSharp.LanguageVersion]::Preview)
        # Every argument is passed: PowerShell does not bind these overloads with their optional ones left out.
        $none = [System.Threading.CancellationToken]::None
        $text = [Microsoft.CodeAnalysis.Text.SourceText]::From(
            $source, [System.Text.Encoding] $null, [Microsoft.CodeAnalysis.Text.SourceHashAlgorithm]::Sha1)
        $tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($text, $options, '', $none)
        $root = $tree.GetCompilationUnitRoot($none)
        foreach ($diagnostic in $root.GetDiagnostics()) {
            if ($diagnostic.Severity -eq [Microsoft.CodeAnalysis.DiagnosticSeverity]::Error) { return $null }
        }
        return [pscustomobject]@{ Root = $root; Lines = $text.Lines; Members = $null; Types = $null }
    }
    catch { return $null }
}

function Get-CSharpNodeLines {
    # First and last line (1-based) of a declaration, attributes included, leading comments not.
    param([Parameter(Mandatory)] $Model, [Parameter(Mandatory)] $Node)
    $span = $Node.Span
    return [int[]] @(
        ($Model.Lines.GetLineFromPosition($span.Start).LineNumber + 1),
        ($Model.Lines.GetLineFromPosition([Math]::Max($span.Start, $span.End - 1)).LineNumber + 1))
}

function Get-CSharpTypeKey {
    # Namespace and containing types with generic arity: the identity of a type across two versions of a file.
    param([Parameter(Mandatory)] $Node)
    $parts = [System.Collections.Generic.List[string]]::new()
    for ($current = $Node; $null -ne $current; $current = $current.Parent) {
        if ($current -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax]) {
            $arity = 0
            if ($current -is [Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax] -and $null -ne $current.TypeParameterList) {
                $arity = $current.TypeParameterList.Parameters.Count
            }
            $parts.Insert(0, "$($current.Identifier.Text)``$arity")
        }
        elseif ($current -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseNamespaceDeclarationSyntax]) {
            $parts.Insert(0, ($current.Name.ToString() -replace '\s', ''))
        }
    }
    return $parts -join '/'
}

function Get-CSharpBodyStart {
    # Where a member's executable part begins: its block, its '=>' body or its accessor list.
    param([Parameter(Mandatory)] $Member)
    if ($Member -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseMethodDeclarationSyntax]) {
        if ($null -ne $Member.Body) { return $Member.Body.SpanStart }
        if ($null -ne $Member.ExpressionBody) { return $Member.ExpressionBody.SpanStart }
    }
    elseif ($Member -is [Microsoft.CodeAnalysis.CSharp.Syntax.BasePropertyDeclarationSyntax]) {
        if ($null -ne $Member.AccessorList) { return $Member.AccessorList.SpanStart }
        if (($Member -is [Microsoft.CodeAnalysis.CSharp.Syntax.PropertyDeclarationSyntax] -or
             $Member -is [Microsoft.CodeAnalysis.CSharp.Syntax.IndexerDeclarationSyntax]) -and $null -ne $Member.ExpressionBody) {
            return $Member.ExpressionBody.SpanStart
        }
    }
    return $Member.Span.End
}

function Get-CSharpMemberKey {
    <#
        The identity of a member across two versions of a file: its type and its whole declaration
        header (modifiers, return type, name, parameters, constraints), comments and spacing aside.
        A member whose header changed has no counterpart, which is intended: a changed signature or
        modifier is decided at type level, never by the member's own span.
    #>
    param([Parameter(Mandatory)] $Member)
    $bodyStart = Get-CSharpBodyStart -Member $Member
    $skipUntil = $Member.SpanStart
    foreach ($list in $Member.AttributeLists) { $skipUntil = [Math]::Max($skipUntil, $list.Span.End) }
    $header = [System.Text.StringBuilder]::new()
    foreach ($token in $Member.DescendantTokens($null, $false)) {
        if ($token.SpanStart -ge $bodyStart) { break }
        if ($token.SpanStart -lt $skipUntil) { continue }
        [void] $header.Append($token.Text).Append(' ')
    }
    return "$(Get-CSharpTypeKey -Node $Member.Parent)::$($header.ToString())"
}

function Get-CSharpMemberLabel {
    param([Parameter(Mandatory)] $Node)
    $typeNames = [System.Collections.Generic.List[string]]::new()
    for ($current = $Node; $null -ne $current; $current = $current.Parent) {
        if ($current -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax]) { $typeNames.Insert(0, $current.Identifier.Text) }
    }
    if ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax]) { return $typeNames -join '.' }
    $name = if ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.MethodDeclarationSyntax] -or
                $Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.PropertyDeclarationSyntax] -or
                $Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.EventDeclarationSyntax]) { $Node.Identifier.Text }
            elseif ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.ConstructorDeclarationSyntax]) { 'constructor' }
            elseif ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.DestructorDeclarationSyntax]) { 'finalizer' }
            elseif ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.IndexerDeclarationSyntax]) { 'this[]' }
            elseif ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.OperatorDeclarationSyntax]) { "operator $($Node.OperatorToken.Text)" }
            elseif ($Node -is [Microsoft.CodeAnalysis.CSharp.Syntax.ConversionOperatorDeclarationSyntax]) { "operator $($Node.Type.ToString())" }
            else { 'member' }
    return (@($typeNames) + @($name)) -join '.'
}

function Initialize-CSharpDeclarationIndex {
    # Every type and member of the file by key. A key two declarations share resolves to neither.
    param([Parameter(Mandatory)] $Model)
    if ($null -ne $Model.Members) { return }
    $members = @{}
    $types = @{}
    $pending = [System.Collections.Generic.Stack[object]]::new()
    $pending.Push($Model.Root)
    while ($pending.Count -gt 0) {
        $node = $pending.Pop()
        $isType = $node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax]
        if ($isType) {
            $key = Get-CSharpTypeKey -Node $node
            if (-not $types.ContainsKey($key)) { $types[$key] = [System.Collections.Generic.List[object]]::new() }
            [void] $types[$key].Add($node)
        }
        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.CompilationUnitSyntax] -or
            $node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseNamespaceDeclarationSyntax] -or
            $node -is [Microsoft.CodeAnalysis.CSharp.Syntax.TypeDeclarationSyntax]) {
            foreach ($child in $node.Members) {
                if ($child -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax] -or
                    $child -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseNamespaceDeclarationSyntax]) {
                    $pending.Push($child)
                }
                elseif ($isType -and ($child -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseMethodDeclarationSyntax] -or
                                      $child -is [Microsoft.CodeAnalysis.CSharp.Syntax.BasePropertyDeclarationSyntax])) {
                    $key = Get-CSharpMemberKey -Member $child
                    if (-not $members.ContainsKey($key)) { $members[$key] = [System.Collections.Generic.List[object]]::new() }
                    [void] $members[$key].Add($child)
                }
            }
        }
    }
    $Model.Members = $members
    $Model.Types = $types
}

function Get-CSharpTokenContext {
    <#
        What one changed token can affect, as the narrowest level that is certainly wide enough:

          Member  inside the body of a method, constructor, finalizer, operator, or of a property,
                  indexer or event accessor (local functions and lambdas belong to the member that
                  contains them). Only code that runs the member can observe it.
          Type    anywhere else in a type: a member's signature, modifiers, attributes, parameter
                  list or constructor initializer, a field or event field, a property or field
                  initializer, an auto-property, the type's own header (attributes, base list,
                  constraints, primary constructor). An added 'override' or a changed signature
                  changes what existing callers bind to with no changed caller line, and an
                  initializer runs in the constructors, so the whole type is the unit.
          File    everything else, and everything whose reach no span in this file bounds: usings,
                  namespace and assembly level constructs, delegates and types outside a type,
                  constants and enum members (their values are compiled into the callers), and
                  anything in a partial type, whose other parts are in other files.
    #>
    param([Parameter(Mandatory)] $Token)

    $member = $null
    for ($node = $Token.Parent; $null -ne $node; $node = $node.Parent) {
        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberDeclarationSyntax]) { $member = $node; break }
    }
    function New-Context([string] $Level, [string] $Why, $Member = $null, $Type = $null) {
        return [pscustomobject]@{ Level = $Level; Why = $Why; Member = $Member; Type = $Type }
    }
    if ($null -eq $member) { return New-Context 'File' 'a using, an assembly attribute or another file-level construct changed' }

    $outermost = $null
    $partial = $false
    for ($node = $member; $null -ne $node; $node = $node.Parent) {
        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax]) {
            $outermost = $node
            foreach ($modifier in $node.Modifiers) { if ($modifier.Text -ceq 'partial') { $partial = $true } }
        }
    }
    if ($null -eq $outermost) { return New-Context 'File' 'a namespace-level construct changed' }

    $inBody = $false
    if ($member -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseMethodDeclarationSyntax]) {
        $inBody = ($null -ne $member.Body -and $member.Body.Span.Contains($Token.Span)) -or
                  ($null -ne $member.ExpressionBody -and $member.ExpressionBody.Span.Contains($Token.Span))
    }
    elseif ($member -is [Microsoft.CodeAnalysis.CSharp.Syntax.BasePropertyDeclarationSyntax]) {
        if (($member -is [Microsoft.CodeAnalysis.CSharp.Syntax.PropertyDeclarationSyntax] -or
             $member -is [Microsoft.CodeAnalysis.CSharp.Syntax.IndexerDeclarationSyntax]) -and
            $null -ne $member.ExpressionBody -and $member.ExpressionBody.Span.Contains($Token.Span)) {
            $inBody = $true
        }
        for ($node = $Token.Parent; -not $inBody -and $null -ne $node -and -not [object]::ReferenceEquals($node, $member); $node = $node.Parent) {
            if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.AccessorDeclarationSyntax]) {
                $inBody = ($null -ne $node.Body -and $node.Body.Span.Contains($Token.Span)) -or
                          ($null -ne $node.ExpressionBody -and $node.ExpressionBody.Span.Contains($Token.Span))
                break
            }
        }
    }
    elseif ($member -is [Microsoft.CodeAnalysis.CSharp.Syntax.EnumMemberDeclarationSyntax]) {
        return New-Context 'File' 'an enum member changed, and its value is compiled into the code that uses it'
    }
    elseif ($member -is [Microsoft.CodeAnalysis.CSharp.Syntax.BaseFieldDeclarationSyntax]) {
        foreach ($modifier in $member.Modifiers) {
            if ($modifier.Text -ceq 'const') { return New-Context 'File' 'a constant changed, and its value is compiled into the code that uses it' }
        }
    }
    elseif ($member -isnot [Microsoft.CodeAnalysis.CSharp.Syntax.BaseTypeDeclarationSyntax] -and
            $member -isnot [Microsoft.CodeAnalysis.CSharp.Syntax.DelegateDeclarationSyntax]) {
        return New-Context 'File' 'a construct the member index does not classify changed'
    }

    if ($inBody) {
        # 'No shard executes it' is only evidence for code the coverage run instruments.
        for ($node = $member; $null -ne $node; $node = $node.Parent) {
            if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberDeclarationSyntax] -and
                $node.AttributeLists.ToString() -match 'ExcludeFromCodeCoverage|GeneratedCode|CompilerGenerated|DebuggerNonUserCode') {
                return New-Context 'File' 'the member is excluded from coverage, so its executed lines are unknown'
            }
        }
        return New-Context 'Member' 'a member body changed' $member $outermost
    }
    if ($partial) { return New-Context 'File' 'a declaration in a partial type changed, and the type has parts in other files' }
    return New-Context 'Type' 'a declaration changed (signature, modifier, attribute, field, initializer or type header)' $null $outermost
}

function Get-CSharpChangeSide {
    <#
        One side of a hunk: the tokens on its lines, each with a kind-and-text key for comparing the
        two sides, and a blocker when the lines hold something whose effect is not a token's:
        a preprocessor directive, code a directive disables, or an XML documentation comment (the
        repository's source generators read those, so they are not inert).
    #>
    param([Parameter(Mandatory)] $Model, [Parameter(Mandatory)] [int] $StartLine, [Parameter(Mandatory)] [int] $Count)

    $side = [pscustomobject]@{
        Tokens = [System.Collections.Generic.List[object]]::new(); Blocker = $null
        Keys = [System.Collections.Generic.List[string]]::new()
    }
    if ($Count -le 0) { return $side }
    $endLine = $StartLine + $Count - 1
    if ($StartLine -lt 1 -or $endLine -gt $Model.Lines.Count) {
        $side.Blocker = 'the changed lines are outside the parsed file'
        return $side
    }
    $from = $Model.Lines[$StartLine - 1].Start
    $to = $Model.Lines[$endLine - 1].EndIncludingLineBreak
    $span = [Microsoft.CodeAnalysis.Text.TextSpan]::FromBounds($from, $to)

    $directive = [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::DisabledTextTrivia
    $skipped = [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::SkippedTokensTrivia
    $documentation = @(
        [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::SingleLineDocumentationCommentTrivia,
        [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::MultiLineDocumentationCommentTrivia)
    foreach ($trivia in $Model.Root.DescendantTrivia($span, $null, $false)) {
        if ($trivia.Span.End -le $from -or $trivia.SpanStart -ge $to) { continue }
        if ($trivia.IsDirective) { $side.Blocker = 'a preprocessor directive changed'; return $side }
        if ($trivia.RawKind -eq $directive) { $side.Blocker = 'code a preprocessor directive disables changed'; return $side }
        if ($trivia.RawKind -eq $skipped) { $side.Blocker = 'tokens the parser skipped changed'; return $side }
        if ($documentation -contains $trivia.RawKind) {
            $side.Blocker = 'an XML documentation comment changed, and source generators read those'
            return $side
        }
    }

    foreach ($token in $Model.Root.DescendantTokens($span, $null, $false)) {
        if ($token.Span.Length -eq 0 -or $token.Span.End -le $from -or $token.SpanStart -ge $to) { continue }
        [void] $side.Tokens.Add($token)
        [void] $side.Keys.Add("$($token.RawKind):$($token.Text)")
    }
    return $side
}

function Get-CSharpTokenContexts {
    # The distinct contexts of Tokens[From..To], one per member or type they sit in.
    param([Parameter(Mandatory)] [AllowEmptyCollection()] $Tokens, [int] $From, [int] $To)
    $items = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $lastParent = $null
    $context = $null
    for ($i = $From; $i -le $To; $i++) {
        $token = $Tokens[$i]
        # Consecutive tokens usually share a parent node, and so a context.
        if ($null -eq $lastParent -or -not [object]::ReferenceEquals($lastParent, $token.Parent)) {
            $lastParent = $token.Parent
            $context = Get-CSharpTokenContext -Token $token
        }
        $anchor = if ($null -ne $context.Member) { $context.Member.SpanStart } elseif ($null -ne $context.Type) { $context.Type.SpanStart } else { -1 }
        if ($seen.Add("$($context.Level):$anchor")) { [void] $items.Add($context) }
    }
    return , $items
}

function Resolve-UncoveredChangeScope {
    <#
        Decides how far one changed hunk that no shard executes can reach, from the file before the
        change (BaseModel) and after it (HeadModel). Hunk is { oldStart, oldCount, newStart, newCount }.

        Returns Level and, for Member and Type, the declarations to select by, each with its node
        in the file BEFORE the change, so the caller can find the same declaration in the map's
        version of the file:

          Inert   both sides hold the same tokens, so only comments or spacing changed.
          Member  every changed token is in a member body, and each such member exists unchanged in
                  signature before the change.
          Type    anything declared in a type; the spans are the outermost enclosing types.
          File    everything else, including every case this function cannot decide. The caller
                  then applies the file-wide rule, so an unknown never selects less than before.
    #>
    param(
        [AllowNull()] $BaseModel,
        [AllowNull()] $HeadModel,
        [Parameter(Mandatory)] [int[]] $Hunk
    )

    function New-Scope([string] $Level, [string] $Why, $Spans = @()) {
        return [pscustomobject]@{ Level = $Level; Why = $Why; Spans = @($Spans) }
    }
    try {
        if ($null -eq $BaseModel -or $null -eq $HeadModel) {
            return New-Scope 'File' 'the file could not be parsed before or after the change'
        }
        $old = Get-CSharpChangeSide -Model $BaseModel -StartLine $Hunk[0] -Count $Hunk[1]
        $new = Get-CSharpChangeSide -Model $HeadModel -StartLine $Hunk[2] -Count $Hunk[3]
        foreach ($side in @($old, $new)) { if ($side.Blocker) { return New-Scope 'File' $side.Blocker } }

        # A diff is by line, so a changed line also carries tokens that did not change (the
        # signature of a one-line member whose body changed). The tokens the two sides share at
        # the start and at the end are not the change; what is left between them is.
        $prefix = 0
        $limit = [Math]::Min($old.Keys.Count, $new.Keys.Count)
        while ($prefix -lt $limit -and $old.Keys[$prefix] -ceq $new.Keys[$prefix]) { $prefix++ }
        $suffix = 0
        while ($suffix -lt $limit - $prefix -and
               $old.Keys[$old.Keys.Count - 1 - $suffix] -ceq $new.Keys[$new.Keys.Count - 1 - $suffix]) { $suffix++ }
        $oldItems = Get-CSharpTokenContexts -Tokens $old.Tokens -From $prefix -To ($old.Tokens.Count - 1 - $suffix)
        $newItems = Get-CSharpTokenContexts -Tokens $new.Tokens -From $prefix -To ($new.Tokens.Count - 1 - $suffix)
        if ($oldItems.Count -eq 0 -and $newItems.Count -eq 0) {
            return New-Scope 'Inert' 'only comments or whitespace changed'
        }

        $all = @(@($oldItems) + @($newItems))
        $file = @($all | Where-Object { $_.Level -ceq 'File' } | Select-Object -First 1)
        if ($file.Count -gt 0) { return New-Scope 'File' $file[0].Why }

        Initialize-CSharpDeclarationIndex -Model $BaseModel
        $typeLevel = @($all | Where-Object { $_.Level -ceq 'Type' }).Count -gt 0
        $members = [System.Collections.Generic.List[object]]::new()
        if (-not $typeLevel) {
            foreach ($item in $oldItems) { [void] $members.Add($item.Member) }
            foreach ($item in $newItems) {
                $match = $BaseModel.Members[(Get-CSharpMemberKey -Member $item.Member)]
                # A member with no single counterpart before the change is new, renamed or re-signed:
                # an added override or overload changes existing callers, so the type is the unit.
                if ($null -eq $match -or $match.Count -ne 1) { $typeLevel = $true; break }
                [void] $members.Add($match[0])
            }
        }
        if (-not $typeLevel) {
            $spans = @(foreach ($member in ($members | Sort-Object { $_.SpanStart } -Unique)) {
                $lines = Get-CSharpNodeLines -Model $BaseModel -Node $member
                [pscustomobject]@{ Label = Get-CSharpMemberLabel -Node $member; Start = $lines[0]; End = $lines[1]; Node = $member }
            })
            return New-Scope 'Member' 'a member body changed' $spans
        }

        $types = [System.Collections.Generic.List[object]]::new()
        foreach ($item in $oldItems) { [void] $types.Add($item.Type) }
        foreach ($item in $newItems) {
            $match = $BaseModel.Types[(Get-CSharpTypeKey -Node $item.Type)]
            if ($null -eq $match -or $match.Count -ne 1) {
                return New-Scope 'File' 'the change adds or renames a type, which has no lines before the change'
            }
            [void] $types.Add($match[0])
        }
        foreach ($type in $types) {
            foreach ($modifier in $type.Modifiers) {
                if ($modifier.Text -ceq 'partial') {
                    return New-Scope 'File' 'a declaration in a partial type changed, and the type has parts in other files'
                }
            }
        }
        $why = @($all | Where-Object { $_.Level -ceq 'Type' } | Select-Object -First 1)
        $spans = @(foreach ($type in ($types | Sort-Object { $_.SpanStart } -Unique)) {
            $lines = Get-CSharpNodeLines -Model $BaseModel -Node $type
            [pscustomobject]@{ Label = Get-CSharpMemberLabel -Node $type; Start = $lines[0]; End = $lines[1]; Node = $type }
        })
        return New-Scope 'Type' $(if ($why.Count -gt 0) { $why[0].Why } else { 'the change adds, renames or re-signs a member' }) $spans
    }
    catch {
        return New-Scope 'File' "the member index failed ($($_.Exception.Message))"
    }
}

function Get-UncoveredRangeDecision {
    <#
        How one changed range that no shard executes is selected for. Rule is:

          Inert    nothing: only comments or whitespace changed.
          Precise  the shards whose executed lines overlap each span in Spans (a member or a type,
                   in the map's line numbers). A member no shard executes has no shards; the caller
                   reports it as untested instead of fanning out.
          File     the file-wide rule the caller applied before this existed. Every doubt lands
                   here: no parser, a parse error, a change the index does not classify, lines the
                   base branch also changed since the map, a declaration the map's version of the
                   file does not have, a type no shard executes, any exception.
    #>
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [object[]] $Range,
        [Parameter(Mandatory)] $Detail,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Occurrences,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $KnownShards,
        [Parameter(Mandatory)] [scriptblock] $ReadSource,
        [Parameter(Mandatory)] [hashtable] $Models
    )

    function New-Decision([string] $Rule, [string] $Why, $Spans = @()) {
        return [pscustomobject]@{ Rule = $Rule; Why = $Why; Spans = @($Spans) }
    }
    try {
        if ($Detail.Swept) { return New-Decision 'File' 'the base branch changed the same lines since the map was built' }
        $owners = @($Detail.Hunks | Where-Object {
            $carried = $_.Map
            $found = $false
            for ($i = 0; $i + 1 -lt $carried.Count; $i += 2) {
                if ($carried[$i] -eq [int] $Range[0] -and $carried[$i + 1] -eq [int] $Range[1]) { $found = $true }
            }
            $found
        })
        if ($owners.Count -eq 0) { return New-Decision 'File' 'the changed lines do not correspond to a hunk of this change' }

        # The file before the change, after it, and as the map measured it. The map's version is the
        # one before the change unless the base branch has edited the file since the map.
        $revisions = [ordered]@{ base = [string] $Detail.Base; head = 'HEAD' }
        $behindBase = @($Detail.MapHunks).Count -gt 0
        if ($behindBase) { $revisions['map'] = [string] $Detail.MapRevision }
        foreach ($side in $revisions.Keys) {
            $key = "$side`n$Path"
            if (-not $Models.ContainsKey($key)) {
                $Models[$key] = Get-CSharpSyntaxModel -Text (& $ReadSource $revisions[$side] $Path)
            }
        }
        $baseModel = $Models["base`n$Path"]
        $mapModel = if ($behindBase) { $Models["map`n$Path"] } else { $baseModel }
        if ($null -eq $mapModel) { return New-Decision 'File' "the map's version of the file could not be parsed" }

        $spans = [System.Collections.Generic.List[object]]::new()
        foreach ($owner in $owners) {
            $scope = Resolve-UncoveredChangeScope -BaseModel $baseModel -HeadModel $Models["head`n$Path"] -Hunk $owner.Hunk
            if ($scope.Level -ceq 'File') { return New-Decision 'File' $scope.Why }
            if ($scope.Level -ceq 'Inert') { continue }
            foreach ($span in $scope.Spans) {
                # The map's line numbers are those of the map's version. The declaration is found
                # there by name, not by carrying its lines back: the base branch may have edited or
                # moved it since, and a name survives both where a line number does not.
                $carried = [int[]] @($span.Start, $span.End)
                if ($behindBase) {
                    Initialize-CSharpDeclarationIndex -Model $mapModel
                    if ($scope.Level -ceq 'Member') {
                        $match = $mapModel.Members[(Get-CSharpMemberKey -Member $span.Node)]
                        if ($null -eq $match -or $match.Count -ne 1) {
                            return New-Decision 'File' "the base branch added or re-signed $($span.Label) since the map was built"
                        }
                    }
                    else {
                        $typeKey = Get-CSharpTypeKey -Node $span.Node
                        $match = $mapModel.Types[$typeKey]
                        if ($null -eq $match -or $match.Count -ne 1) {
                            return New-Decision 'File' "the base branch added or renamed $($span.Label) since the map was built"
                        }
                        # A member the type gained since the map may be code moved in from another type
                        # of the file, whose executed lines the map holds outside this type's span.
                        Initialize-CSharpDeclarationIndex -Model $baseModel
                        foreach ($memberKey in $baseModel.Members.Keys) {
                            if (($memberKey.StartsWith("${typeKey}::", [StringComparison]::Ordinal) -or
                                 $memberKey.StartsWith("${typeKey}/", [StringComparison]::Ordinal)) -and
                                -not $mapModel.Members.ContainsKey($memberKey)) {
                                return New-Decision 'File' "the base branch added a member to $($span.Label) since the map was built"
                            }
                        }
                    }
                    # The exclusion that matters is the map's: a declaration the map measured under
                    # [ExcludeFromCodeCoverage] (or inside an excluded type) has no executed lines there
                    # even when the base has since removed the attribute, so its span proves nothing.
                    for ($node = $match[0]; $null -ne $node; $node = $node.Parent) {
                        if ($node -is [Microsoft.CodeAnalysis.CSharp.Syntax.MemberDeclarationSyntax] -and
                            $node.AttributeLists.ToString() -match 'ExcludeFromCodeCoverage|GeneratedCode|CompilerGenerated|DebuggerNonUserCode') {
                            return New-Decision 'File' "the map measured $($span.Label) excluded from coverage, so its executed lines are unknown"
                        }
                    }
                    $carried = Get-CSharpNodeLines -Model $mapModel -Node $match[0]
                }
                # Several hunks of one member or type share its span, and so its shards.
                $shardsKey = "shards`n$Path`n$($carried[0])-$($carried[1])"
                if (-not $Models.ContainsKey($shardsKey)) {
                    $found = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
                    foreach ($occurrence in $Occurrences) {
                        $executed = @($occurrence.r)
                        for ($i = 0; $i + 1 -lt $executed.Count; $i += 2) {
                            if ([int] $executed[$i] -le $carried[1] -and [int] $executed[$i + 1] -ge $carried[0]) {
                                [void] $found.Add([string] $KnownShards[[int] $occurrence.s])
                                break
                            }
                        }
                    }
                    $Models[$shardsKey] = $found
                }
                $shards = $Models[$shardsKey]
                # A member nothing executes is untested code. A type nothing executes is not the same
                # claim: a field, attribute or base list is read by reflection and by generated code
                # without one of the type's own lines running, so the file's shards still run.
                if ($shards.Count -eq 0 -and $scope.Level -ceq 'Type') {
                    return New-Decision 'File' "no shard executes any line of $($span.Label)"
                }
                [void] $spans.Add([pscustomobject]@{
                    Label = $span.Label; Level = $scope.Level; Shards = @($shards)
                    Start = $carried[0]; End = $carried[1]
                })
            }
        }
        if ($spans.Count -eq 0) { return New-Decision 'Inert' 'only comments or whitespace changed' }
        return New-Decision 'Precise' '' $spans
    }
    catch {
        return New-Decision 'File' "the member index failed ($($_.Exception.Message))"
    }
}

function Format-LineRanges {
    param([AllowEmptyCollection()] [object[]] $Ranges)
    return (@($Ranges | ForEach-Object { "$($_[0])-$($_[1])" }) -join ', ')
}

function Add-UnmappedShardsAsAlwaysRun {
    <#
    .SYNOPSIS
        Runs every manifest shard the map has not seen on every selection, until a map includes it.
    .DESCRIPTION
        A shard added to the manifest after the map was built has no coverage in it, so nothing can
        say which changes reach it. Refusing the whole map for that reason used to send every pull
        request back to the full matrix from the moment a shard was added until a new map was
        generated and certified - hours of full runs to learn about one shard. Running just the
        unmapped shard every time is the honest reading of "no evidence": it costs what that shard
        cost before it could be selected, and every mapped shard stays selective.
    .OUTPUTS
        The unmapped shard names, sorted. The map is updated in place.
    #>
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [string[]] $Expected
    )

    $mapSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($name in @($Map.knownShards) + @($Map.alwaysRun)) { [void] $mapSet.Add([string] $name) }
    $unmapped = @($Expected | Where-Object { -not $mapSet.Contains([string] $_) } | Sort-Object -Unique)
    if ($unmapped.Count -gt 0) {
        $Map.alwaysRun = @(@($Map.alwaysRun) + $unmapped)
        $Map | Add-Member -NotePropertyName unmappedShards -NotePropertyValue $unmapped -Force
    }
    return $unmapped
}

# Environment variables that slice the reflected model inventory by POSITION: the conformance
# windows Skip/Take a sorted type list and the parameter-count sweep takes index % 8. Adding one
# model moves others between such shards without touching their executed lines, so their coverage
# index is only trusted while the inventory is unchanged (see Test-AuxiliaryInventoryChange).
$script:InventoryWindowEnvironment = @('ADNSHAPE_CONF_OFFSET', 'ADNSHAPE_CONF_WINDOW', 'AIDOTNET_PARAMETER_COUNT_SHARD')
# Of those, the windows keyed by model name rather than position (see Get-AuxiliaryWindowInvalidation).
$script:NameKeyedWindowEnvironment = @('ADNSHAPE_CONF_WINDOW', 'AIDOTNET_PARAMETER_COUNT_SHARD')

function Test-NameKeyedWindow {
    param([Parameter(Mandatory)] [object] $Shard)
    $environment = if ($Shard -is [System.Collections.IDictionary]) { $Shard['env'] } else { $Shard.PSObject.Properties['env']?.Value }
    if ($null -eq $environment) { return $false }
    $names = if ($environment -is [System.Collections.IDictionary]) { @($environment.Keys) } else { @($environment.PSObject.Properties.Name) }
    return @($names | Where-Object { [string] $_ -cin $script:NameKeyedWindowEnvironment }).Count -gt 0 -and
        @($names | Where-Object { [string] $_ -ceq 'ADNSHAPE_CONF_OFFSET' }).Count -eq 0
}

function Test-InventoryWindow {
    param([Parameter(Mandatory)] [object] $Shard)

    function Get-Field([object] $Object, [string] $Name) {
        if ($Object -is [System.Collections.IDictionary]) {
            if ($Object.Contains($Name)) { return , $Object[$Name] }
            return $null
        }
        $property = $Object.PSObject.Properties[$Name]
        if ($null -eq $property) { return $null }
        return , $property.Value
    }
    $workload = Get-Field $Shard 'workload'
    if ($null -ne $workload -and [string] $workload -cne 'Tests') { return $true }
    $environment = Get-Field $Shard 'env'
    if ($null -eq $environment) { return $false }
    $names = if ($environment -is [System.Collections.IDictionary]) { @($environment.Keys) }
             else { @($environment.PSObject.Properties.Name) }
    return @($names | Where-Object { [string] $_ -cin $script:InventoryWindowEnvironment }).Count -gt 0
}

function ConvertTo-CanonicalShardJson {
    <#
        One text per shard definition regardless of how it was loaded. yq output, hashtables and
        ConvertFrom-Json objects disagree on key order and container type, not on content.
    #>
    param([AllowNull()] $Value)

    if ($null -eq $Value) { return 'null' }
    if ($Value -is [string]) { return ConvertTo-Json -InputObject $Value -Compress }
    if ($Value -is [System.Collections.IDictionary]) {
        $pairs = foreach ($key in @($Value.Keys | ForEach-Object { [string] $_ } | Sort-Object -CaseSensitive)) {
            (ConvertTo-Json -InputObject $key -Compress) + ':' + (ConvertTo-CanonicalShardJson $Value[$key])
        }
        return '{' + (@($pairs) -join ',') + '}'
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $pairs = foreach ($key in @($Value.PSObject.Properties.Name | Sort-Object -CaseSensitive)) {
            (ConvertTo-Json -InputObject $key -Compress) + ':' +
                (ConvertTo-CanonicalShardJson $Value.PSObject.Properties[$key].Value)
        }
        return '{' + (@($pairs) -join ',') + '}'
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $items = foreach ($item in $Value) { ConvertTo-CanonicalShardJson $item }
        return '[' + (@($items) -join ',') + ']'
    }
    return ConvertTo-Json -InputObject $Value -Compress
}

function Read-ShardManifestAtRevision {
    <#
        The shard entries committed at a revision, or $null when that revision has no
        .github/test-shards.yml. A manifest that exists but cannot be parsed throws.
    #>
    param([Parameter(Mandatory)] [string] $Revision)

    $source = @(& git show "${Revision}:.github/test-shards.yml" 2>$null)
    if ($LASTEXITCODE -ne 0) { return $null }
    $json = ($source -join "`n") | & yq -o=json -I=0 '.shard' '-'
    if ($LASTEXITCODE -ne 0) { throw "cannot parse .github/test-shards.yml at $Revision" }
    $entries = @(($json -join "`n") | ConvertFrom-Json)
    if ($entries.Count -eq 0) { throw ".github/test-shards.yml at $Revision has no shards" }
    return , $entries
}

function Get-RedefinedShards {
    <#
        Pure. The indexed shards whose current definition differs from the one the map measured.
        MapManifest is the manifest at the map's commit; an empty one means it could not be
        established, so every indexed shard counts as redefined.

        Per-shard comparison is sufficient because a filter that moves tests between shards changes
        the text of both: catch-all shards here list their exclusions explicitly.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $MapManifest,
        [Parameter(Mandatory)] [object[]] $Manifest,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $IndexedShards
    )

    $measured = [System.Collections.Generic.Dictionary[string, string]]::new([StringComparer]::Ordinal)
    foreach ($entry in $MapManifest) { $measured[[string] $entry.name] = ConvertTo-CanonicalShardJson $entry }
    $indexed = [System.Collections.Generic.HashSet[string]]::new([string[]] @($IndexedShards), [StringComparer]::Ordinal)
    $redefined = foreach ($entry in $Manifest) {
        $name = [string] $entry.name
        if (-not $indexed.Contains($name)) { continue }
        if (-not $measured.ContainsKey($name) -or $measured[$name] -cne (ConvertTo-CanonicalShardJson $entry)) { $name }
    }
    return @($redefined | Sort-Object -Unique)
}

function Select-ImpactedShards {
    <#
        CurrentPaths is the change being validated. With -ScopeToCurrentPaths, only those paths are
        selected for: Changed still supplies their line ranges in the MAP's coordinates, but a file
        that differs from the map only because master moved on since the map was built belongs to
        commits that were validated when they landed, not to this pull request.

        Without the switch (the nightly audit, which replays everything since the map as one
        change) CurrentPaths only decides whether a selection-control edit is current.

        TestRoutes carries the precomputed routing for test sources the coverage map cannot index;
        see Get-TestFileRoutes. Every selected shard is recorded in Routes with the reason it was
        selected, so a log reader can see why a shard runs without re-deriving it.

        ChangeDetail (New-ChangeDetail records by path) and ReadSource ({ param($revision, $path) }
        returning a file's text, or $null) enable the precise fallbacks for a changed range no
        shard executes; see Get-UncoveredRangeDecision. Without either, or with PreciseFallbacks
        off, such a range selects every shard that executes the file, as it always did. Untested
        lists the changed members no shard executes, and NewSourcePaths the unmapped C# sources
        routed to their directory's owners.
    #>
    param(
        [Parameter(Mandatory)] $Map,
        [Parameter(Mandatory)] [hashtable] $Changed,
        [AllowEmptyCollection()] [string[]] $CurrentPaths = @(),
        [switch] $ScopeToCurrentPaths,
        [hashtable] $TestRoutes = @{},
        [switch] $AuditUnchangedMap,
        [string[]] $ReviewedControlPaths = @(),
        [string[]] $RequiredShards = @(),
        [hashtable] $ChangeDetail = @{},
        [scriptblock] $ReadSource = $null,
        [bool] $PreciseFallbacks = (Test-PreciseFallbacksEnabled),
        # Set only for one layer of a rolling map (New-RollingSelectionContext); Map is then that layer's view
        # and Changed is the change in that layer's own commit numbering. $null leaves every rule as it was.
        $Rolling = $null
    )

    # Indexed shards whose manifest definition changed since the map was measured: their index
    # describes a test set they may no longer run, so they are mandatory like always-run shards.
    $redefined = @(if ($Map.PSObject.Properties['redefinedShards']) { @($Map.redefinedShards) })
    $mandatory = @(@($Map.alwaysRun) + $redefined)
    $selected = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $mappedPaths = [System.Collections.Generic.List[string]]::new()
    $reasons = [System.Collections.Generic.List[string]]::new()
    $routes = [System.Collections.Generic.List[string]]::new()
    # One record per pull-request path, written where it is decided, so the selection report shows the decision
    # itself (category, outcome, shards, reason) instead of reconstructing it from the flat route strings.
    $files = [System.Collections.Generic.List[object]]::new()
    # One record per member, however many of its ranges changed.
    $untested = [ordered]@{}
    $newSourcePaths = [System.Collections.Generic.List[string]]::new()
    $syntaxModels = @{}
    $precise = $PreciseFallbacks -and $null -ne $ReadSource -and $ChangeDetail.Count -gt 0
    function Add-FileDecision([string] $Path, [string] $Category, [string] $Outcome, [string[]] $Shards, [string] $Why) {
        [void] $files.Add([pscustomobject]@{
            path = $Path; category = $Category; outcome = $Outcome; shards = @($Shards | Where-Object { $_ }); why = $Why
        })
    }
    # Per invocation and lazy: mapped changes pay no indexing cost, and selecting a
    # different map in the same process cannot reuse another map's directory owners.
    $directoryOwnerIndex = $null
    $escalate = $false
    foreach ($name in $RequiredShards) {
        if ($name -cnotin (@($Map.knownShards) + @($Map.alwaysRun))) { throw "Required policy workload is absent from the map: $name" }
        [void] $selected.Add($name)
        [void] $routes.Add("$name <= its manifest or execution policy changed")
    }
    $currentPathSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $effectiveCurrentPaths = if ($PSBoundParameters.ContainsKey('CurrentPaths')) {
        @($CurrentPaths)
    }
    else {
        @($Changed.Keys)
    }
    foreach ($currentPath in $effectiveCurrentPaths) {
        if (-not [string]::IsNullOrWhiteSpace([string] $currentPath)) {
            [void] $currentPathSet.Add(([string] $currentPath).Replace('\', '/'))
        }
    }

    if ($ScopeToCurrentPaths) {
        # Scoping discards every path outside the pull request, so an empty pull-request path set
        # would otherwise discard everything and read as a deliberate non-runtime result.
        if ($currentPathSet.Count -eq 0) {
            return [pscustomobject]@{
                Escalate           = $true
                RequiresValidation = $true
                Reasons            = @("the pull request's own changed path set was empty")
                Shards             = @()
                Routes             = @()
            }
        }

        # A pull-request path whose content equals the map's own copy has no map-to-HEAD hunks, yet
        # it still differs from the merge base (master changed it after the map and this pull
        # request reverted that). Its effect cannot be expressed in the map's line numbers.
        $changedPathSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        foreach ($key in $Changed.Keys) { [void] $changedPathSet.Add(([string] $key).Replace('\', '/')) }
        foreach ($currentPath in ($currentPathSet | Sort-Object)) {
            if ($currentPath -cin $ReviewedControlPaths) { continue }
            if ((Get-ChangedPathImpact -Path $currentPath) -eq [ChangedPathImpact]::NonRuntime) { continue }
            if (-not $changedPathSet.Contains($currentPath)) {
                $escalate = $true
                [void] $reasons.Add("changed by this pull request but identical to the map's copy, so its effect has no map line numbers: $currentPath")
                Add-FileDecision $currentPath 'MapCandidate' 'escalated' @() "identical to the map's copy, so its effect has no map line numbers"
            }
        }
    }

    foreach ($path in ($Changed.Keys | Sort-Object)) {
        if ($ScopeToCurrentPaths -and -not $currentPathSet.Contains(([string] $path).Replace('\', '/'))) {
            continue
        }
        $impact = Get-ChangedPathImpact -Path $path
        $isCurrent = $currentPathSet.Contains(([string] $path).Replace('\', '/'))
        if ($path -cin $ReviewedControlPaths) {
            if ($isCurrent) { Add-FileDecision $path $impact.ToString() 'reviewed' @() 'reviewed control path; its policy shards are required instead' }
            continue
        }
        switch ($impact) {
            ([ChangedPathImpact]::NonRuntime) {
                if ($isCurrent) { Add-FileDecision $path 'NonRuntime' 'skipped' @() 'cannot affect any test' }
                continue
            }
            ([ChangedPathImpact]::BuildOnly) {
                # Compiled, never routed: it can raise an analyzer to error but cannot change what a test executes.
                if ($isCurrent) { Add-FileDecision $path 'BuildOnly' 'skipped' @() 'compiled by the build jobs; cannot change what any test executes' }
                continue
            }
            ([ChangedPathImpact]::SelectionControl) {
                # A control-path edit in THIS pull request must exercise the complete matrix. The
                # same path in the older map-to-HEAD delta was already validated when it landed and
                # cannot change source line ranges; treating it as permanently current otherwise
                # wedges every future runtime PR in full-matrix mode until another map is built.
                if ($currentPathSet.Contains(([string] $path).Replace('\', '/'))) {
                    $escalate = $true
                    [void] $reasons.Add("current validation-selection control change: $path")
                    Add-FileDecision $path 'SelectionControl' 'escalated' @() 'changes the selection tooling itself, so the whole matrix validates it'
                }
                continue
            }
            ([ChangedPathImpact]::FullValidation) {
                $escalate = $true
                [void] $reasons.Add("validation infrastructure or unknown GitHub configuration: $path")
                if ($isCurrent) { Add-FileDecision $path 'FullValidation' 'escalated' @() 'validation infrastructure or unknown GitHub configuration' }
                continue
            }
            ([ChangedPathImpact]::MapCandidate) {
                [void] $mappedPaths.Add([string] $path)
            }
        }
    }

    # Audits must exercise the mandatory shard set even when the only changes since
    # the candidate map are non-runtime. Ordinary docs-only PRs still select nothing.
    if ($Changed.Count -eq 0 -or ($AuditUnchangedMap -and $mappedPaths.Count -eq 0 -and -not $escalate)) {
        if ($AuditUnchangedMap) {
            foreach ($shard in $mandatory) { [void] $selected.Add([string] $shard) }
            # Certificate policy requires both a nonempty selected set and a nonempty skipped set.
            # With no always-run shards, an unchanged audit is vacuous and must wait for a real
            # mapped change rather than certifying that selection was exercised when it was not.
            if ($selected.Count -gt 0) {
                return [pscustomobject]@{
                    Escalate           = $false
                    RequiresValidation = $true
                    Reasons            = @('audited tree has no runtime delta; exercising mandatory shards')
                    Shards             = @($selected | Sort-Object)
                    Routes             = @($selected | Sort-Object | ForEach-Object { "$_ <= is always run" })
                }
            }
        }

        $emptyReason = $(if ($Changed.Count -eq 0) { 'changed path set was empty' }
            else { 'audit has no mandatory shards and no runtime delta' })
        if ($null -ne $Rolling) {
            # One layer with nothing to say is not a failed selection: another layer may select, and
            # Merge-RollingLayerSelections escalates with this reason only when none does.
            return [pscustomobject]@{
                Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @(); Routes = @(); Vacuous = $emptyReason
            }
        }
        return [pscustomobject]@{
            Escalate            = $true
            RequiresValidation = $true
            Reasons             = @($emptyReason)
            Shards              = @()
            Routes              = @()
        }
    }

    # A non-runtime-only change is a deliberate empty selection, distinct from a selector failure.
    # Keep that state typed here and expose only a JSON boolean at the workflow boundary.
    if ($mappedPaths.Count -eq 0) {
        if ($selected.Count -gt 0) {
            foreach ($shard in $mandatory) { [void] $selected.Add([string] $shard) }
        }
        return [pscustomobject]@{
            Escalate          = $escalate
            RequiresValidation = $escalate -or $selected.Count -gt 0
            Reasons           = $reasons
            Shards            = @($selected | Sort-Object)
            Routes            = @($routes)
            Files             = @($files)
        }
    }

    $unmappedShards = if ($Map.PSObject.Properties['unmappedShards']) { @($Map.unmappedShards) } else { @() }
    foreach ($shard in @($Map.alwaysRun)) {
        [void] $selected.Add([string] $shard)
        if ($unmappedShards -contains $shard) {
            [void] $routes.Add("$shard <= is not in the coverage map yet, so it runs until a map includes it")
        }
        else {
            [void] $routes.Add("$shard <= is always run")
        }
    }
    foreach ($shard in $redefined) {
        [void] $selected.Add([string] $shard)
        [void] $routes.Add("$shard <= changed its manifest definition since the map measured it, so it runs until a map measures it again")
    }

    foreach ($path in $mappedPaths) {

        $entry = $Map.files.PSObject.Properties[$path]
        if (-not $entry) {
            if ($TestRoutes.ContainsKey($path)) {
                # Test sources are never instrumented, so the map can never index them. Their owning
                # shards come from the manifest's own filters instead; see Get-TestFileRoutes.
                $route = $TestRoutes[$path]
                if ($route.Routable) {
                    foreach ($shard in @($route.Shards)) {
                        [void] $selected.Add([string] $shard)
                        [void] $routes.Add("$shard <= runs tests affected by $path ($($route.Why))")
                    }
                    Add-FileDecision $path $(if ($route.PSObject.Properties['Category']) { [string] $route.Category } else { 'TestSource' }) 'routed' @($route.Shards) $route.Why
                }
                elseif ($route.PSObject.Properties['Unassigned'] -and $route.Unassigned) {
                    [void] $routes.Add("(none) <= $path`: $($route.Why); the shard inventory gate fails the build until one does")
                    Add-FileDecision $path 'TestSource' 'unassigned' @() "$($route.Why); the shard inventory gate fails the build until one does"
                }
                else {
                    $escalate = $true
                    [void] $reasons.Add("$($route.Why): $path")
                    Add-FileDecision $path 'TestSource' 'escalated' @() $route.Why
                }
                continue
            }

            # An unmapped source file - normally one this pull request adds - has no coverage of its
            # own. Only C# is routed to its directory's owners: an unmapped project, props or data
            # file can change how everything builds or loads, which no neighbour's coverage bounds.
            if ($null -ne $Rolling -and $Rolling.MappedPaths.Contains([string] $path) -and
                $Rolling.SettledPaths.Contains([string] $path)) {
                # Rolling map: another layer indexes this file, and at THIS layer's commit it already existed
                # with exactly the content the change edits (unchanged from there to the change's base). So
                # this layer's shards were measured with it and executed none of it: the same fact a single
                # map records by leaving a shard out of the file's occurrences. A file the base branch has
                # touched since this layer's commit does not qualify, and takes the unmapped-file rule below.
                Add-FileDecision $path 'MapCandidate' 'routed' @() "no shard measured at $(([string] $Rolling.Sha).Substring(0, 8)) executes it"
                continue
            }
            $directoryOwners = $null
            if ($path.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase) -and
                -not $path.StartsWith('tests/', [StringComparison]::OrdinalIgnoreCase)) {
                if ($null -ne $Rolling) {
                    # Unmapped everywhere, or added after this layer was measured (its shards' coverage of it
                    # is unknown until they are measured again): the unmapped-file rule, for this layer's shards.
                    $directoryOwners = Get-RollingDirectoryOwners -Rolling $Rolling -Path $path
                }
                else {
                    if ($null -eq $directoryOwnerIndex) { $directoryOwnerIndex = New-DirectoryOwnerIndex -Map $Map }
                    $directoryOwners = Get-DirectoryOwners -Index $directoryOwnerIndex -Path $path
                }
            }
            if ($null -eq $directoryOwners) {
                $escalate = $true
                [void] $reasons.Add("not executed by any mapped shard: $path")
                Add-FileDecision $path 'MapCandidate' 'escalated' @() 'not executed by any mapped shard, and not C# source with mapped neighbours'
                continue
            }
            foreach ($shard in @($directoryOwners.Shards)) {
                [void] $selected.Add([string] $shard)
                [void] $routes.Add("$shard <= executes mapped files in $($directoryOwners.Directory), beside the unmapped source $path")
            }
            Add-FileDecision $path 'MapCandidate' 'routed' @($directoryOwners.Shards) "unmapped source; routed to the shards executing $($directoryOwners.Directory)"
            [void] $newSourcePaths.Add([string] $path)
            continue
        }

        $hunks = @($Changed[$path])
        if ($hunks.Count -eq 0 -or $hunks.Count % 2 -ne 0) {
            $escalate = $true
            [void] $reasons.Add("changed file has no trustworthy line hunks: $path")
            Add-FileDecision $path 'MapCandidate' 'escalated' @() 'no trustworthy line hunks'
            continue
        }

        $covered = [bool[]]::new([int] ($hunks.Count / 2))
        $fileOwners = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
        $hitsByShard = [ordered]@{}
        foreach ($occurrence in @($entry.Value)) {
            $ranges = @($occurrence.r)
            $shardName = [string] $Map.knownShards[[int] $occurrence.s]
            [void] $fileOwners.Add($shardName)
            for ($h = 0; $h + 1 -lt $hunks.Count; $h += 2) {
                $hit = $false
                for ($i = 0; $i + 1 -lt $ranges.Count; $i += 2) {
                    if (Test-RangeOverlap -AStart ([int] $hunks[$h]) -AEnd ([int] $hunks[$h + 1]) `
                                          -BStart ([int] $ranges[$i]) -BEnd ([int] $ranges[$i + 1])) {
                        $hit = $true
                        break
                    }
                }
                if ($hit) {
                    $covered[[int] ($h / 2)] = $true
                    [void] $selected.Add($shardName)
                    if (-not $hitsByShard.Contains($shardName)) {
                        $hitsByShard[$shardName] = [System.Collections.Generic.List[object]]::new()
                    }
                    [void] $hitsByShard[$shardName].Add(@($hunks[$h], $hunks[$h + 1]))
                }
            }
        }
        foreach ($shardName in $hitsByShard.Keys) {
            [void] $routes.Add("$shardName <= executes changed lines $(Format-LineRanges $hitsByShard[$shardName]) of $path")
        }

        # A changed range no shard executes is not dead code as a rule: it is a declaration coverage
        # never records (a field, an attribute, a brace between methods), a comment, or the
        # insertion point of new code. Each is first selected by what it changed - nothing for
        # comments and spacing, the enclosing member for a body, the enclosing type for a
        # declaration (Get-UncoveredRangeDecision) - and whatever that cannot decide is routed to
        # every shard that executes ANY line of the same file, the rule that applied to all of
        # them before. This is the heuristic the nightly miss audit exists to check.
        $uncovered = [System.Collections.Generic.List[object]]::new()
        for ($h = 0; $h + 1 -lt $hunks.Count; $h += 2) {
            if (-not $covered[[int] ($h / 2)]) { [void] $uncovered.Add(@($hunks[$h], $hunks[$h + 1])) }
        }
        $fileRule = [System.Collections.Generic.List[object]]::new()
        $fileRuleWhy = ''
        $inertRanges = [System.Collections.Generic.List[object]]::new()
        $untestedRanges = [System.Collections.Generic.List[object]]::new()
        $spanRanges = [System.Collections.Generic.List[object]]::new()
        $spanShards = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
        $spanRoutes = [ordered]@{}
        $decideByMember = $precise -and $ChangeDetail.ContainsKey($path) -and
            ([string] $path).EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase)
        $decidedRanges = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($range in $uncovered) {
            $decision = $null
            if ($decideByMember) {
                # Several hunks can carry back to the same map lines; one decision covers them all.
                if (-not $decidedRanges.Add("$($range[0]),$($range[1])")) { continue }
                $decision = Get-UncoveredRangeDecision -Path $path -Range $range -Detail $ChangeDetail[$path] `
                    -Occurrences @($entry.Value) -KnownShards @($Map.knownShards) -ReadSource $ReadSource -Models $syntaxModels
            }
            if ($null -eq $decision -or $decision.Rule -ceq 'File') {
                [void] $fileRule.Add($range)
                if ($null -ne $decision -and -not $fileRuleWhy) { $fileRuleWhy = [string] $decision.Why }
                # One range under the file-wide rule already runs every shard executing the file, so
                # nothing a later range decides can add to or take from the selection.
                $decideByMember = $false
                continue
            }
            if ($decision.Rule -ceq 'Inert') {
                [void] $inertRanges.Add($range)
                continue
            }
            $reached = $false
            foreach ($span in $decision.Spans) {
                if (@($span.Shards).Count -eq 0) {
                    $untestedKey = "$path`n$($span.Label)`n$($span.Start)-$($span.End)"
                    if ($untested.Contains($untestedKey)) {
                        $untested[$untestedKey].lines += ", $(Format-LineRanges @(, $range))"
                    }
                    else {
                        $untested[$untestedKey] = [pscustomobject]@{
                            path = [string] $path; member = [string] $span.Label
                            lines = Format-LineRanges @(, $range); memberLines = "$($span.Start)-$($span.End)"
                        }
                    }
                    continue
                }
                $reached = $true
                foreach ($shardName in $span.Shards) {
                    [void] $selected.Add($shardName)
                    [void] $spanShards.Add($shardName)
                    $routeKey = "$shardName <= executes $($span.Label) (lines $($span.Start)-$($span.End)) of $path"
                    if (-not $spanRoutes.Contains($routeKey)) { $spanRoutes[$routeKey] = [System.Collections.Generic.List[object]]::new() }
                    [void] $spanRoutes[$routeKey].Add($range)
                }
            }
            if ($reached) { [void] $spanRanges.Add($range) } else { [void] $untestedRanges.Add($range) }
        }
        foreach ($routeKey in $spanRoutes.Keys) {
            [void] $routes.Add("$routeKey, which holds changed lines $(Format-LineRanges $spanRoutes[$routeKey]) that no shard executes")
        }
        if ($fileRule.Count -gt 0) {
            foreach ($shardName in $fileOwners) {
                [void] $selected.Add($shardName)
                [void] $routes.Add("$shardName <= executes $path, whose changed lines $(Format-LineRanges $fileRule) no shard executes")
            }
        }
        $fileShards = @(@($hitsByShard.Keys) + @($spanShards) + @(if ($fileRule.Count -gt 0) { @($fileOwners) }) | Sort-Object -Unique)
        $fileWhy = "$($hitsByShard.Count) shard(s) execute its changed lines" +
            $(if ($fileRule.Count -gt 0) {
                "; lines $(Format-LineRanges $fileRule) run in no shard, so every $($fileOwners.Count) shard(s) executing the file run" +
                $(if ($fileRuleWhy) { " ($fileRuleWhy)" } else { '' })
            } else { '' }) +
            $(if ($spanRanges.Count -gt 0) { "; lines $(Format-LineRanges $spanRanges) run in no shard and select the $($spanShards.Count) shard(s) executing their enclosing member or type" } else { '' }) +
            $(if ($untestedRanges.Count -gt 0) { "; lines $(Format-LineRanges $untestedRanges) are in a member no shard executes (untested code changed)" } else { '' }) +
            $(if ($inertRanges.Count -gt 0) { "; lines $(Format-LineRanges $inertRanges) changed only comments or whitespace" } else { '' })
        Add-FileDecision $path 'MapCandidate' 'routed' $fileShards $fileWhy
    }

    $vacuous = ''
    if ($selected.Count -eq 0) {
        if ($null -ne $Rolling) { $vacuous = 'selection was empty' }
        else {
            $escalate = $true
            [void] $reasons.Add('selection was empty')
        }
    }

    $result = [pscustomobject]@{
        Escalate           = $escalate
        RequiresValidation = $true
        Reasons            = $reasons
        Shards             = @($selected | Sort-Object)
        Routes             = @($routes)
        Files              = @($files)
        Untested           = @($untested.Values)
        NewSourcePaths     = @($newSourcePaths)
    }
    if ($null -ne $Rolling) { $result | Add-Member -NotePropertyName Vacuous -NotePropertyValue $vacuous }
    return $result
}

# ---------------------------------------------------------------- rolling map
#
# A rolling map (RollingShardMap.ps1) holds one ordinary map per measured commit. NOTHING here compares a
# layer's ranges with anything but the change translated to that layer's own commit: each layer is selected
# for by Select-ImpactedShards exactly as a single map is, with its own Get-ScopedChangedRanges /
# Get-ChangedRanges result, and only the resulting shard NAMES are merged. Names carry no coordinates.

function Test-RollingMapFile {
    # A cheap look at the head of the file, so an ordinary map is never parsed twice. A file that is missing or
    # cannot be read is not a rolling map: the ordinary path below then reports it exactly as it always has.
    param([Parameter(Mandatory)] [string] $Path)
    try {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        $buffer = [char[]]::new(256)
        $reader = [System.IO.StreamReader]::new((Resolve-Path -LiteralPath $Path).ProviderPath)
        try { $read = $reader.Read($buffer, 0, $buffer.Length) } finally { $reader.Dispose() }
        return ([string]::new($buffer, 0, [Math]::Max(0, $read))) -match '"kind"\s*:\s*"rolling"'
    }
    catch { return $false }
}

function New-RollingSelectionContext {
    <#
        What every layer needs to know about the OTHER layers, none of it line numbers: which paths any layer
        indexes, and which shards execute files in each directory. Paths and directory membership mean the
        same at every commit; ranges do not, and are not here.
    #>
    param([Parameter(Mandatory)] [object[]] $Layers)

    $mapped = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $owners = [System.Collections.Generic.Dictionary[string, System.Collections.Generic.SortedSet[string]]]::new(
        [StringComparer]::OrdinalIgnoreCase)
    $fileCount = [System.Collections.Generic.Dictionary[string, int]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($layer in $Layers) {
        $known = @($layer.map.knownShards)
        foreach ($property in $layer.map.files.PSObject.Properties) {
            $isNew = $mapped.Add([string] $property.Name)
            $segments = @($property.Name.Replace('\', '/').Split('/'))
            for ($depth = $segments.Count - 1; $depth -ge 2; $depth--) {
                $directory = ($segments[0..($depth - 1)] -join '/') + '/'
                if (-not $owners.ContainsKey($directory)) {
                    $owners[$directory] = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
                    $fileCount[$directory] = 0
                }
                if ($isNew) { $fileCount[$directory]++ }
                foreach ($occurrence in @($property.Value)) { [void] $owners[$directory].Add([string] $known[[int] $occurrence.s]) }
            }
        }
    }
    return [pscustomobject]@{ MappedPaths = $mapped; DirectoryOwners = $owners; DirectoryFileCount = $fileCount }
}

function New-RollingLayerView {
    <#
        One layer as the map Select-ImpactedShards reads: the layer's own index, with the whole shard universe
        around it so universe rules (policy-required workloads, always-run) read the same in every layer.
        The layer's shards keep their positions, which the file index addresses; other layers' shards follow
        and have no occurrences here.
    #>
    param(
        [Parameter(Mandatory)] $Layer,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $AllKnown,
        [AllowEmptyCollection()] [string[]] $AlwaysRun = @(),
        [AllowEmptyCollection()] [string[]] $Unmapped = @()
    )

    $homeNames = @($Layer.map.knownShards | ForEach-Object { [string] $_ })
    $homeSet = [System.Collections.Generic.HashSet[string]]::new([string[]] $homeNames, [StringComparer]::Ordinal)
    $view = [pscustomobject]@{
        schemaVersion = 1; sha = [string] $Layer.sha
        knownShards = @($homeNames + @($AllKnown | Where-Object { -not $homeSet.Contains([string] $_) }))
        alwaysRun = @($AlwaysRun); files = $Layer.map.files
    }
    if ($Unmapped.Count -gt 0) { $view | Add-Member -NotePropertyName unmappedShards -NotePropertyValue @($Unmapped) }
    return [pscustomobject]@{ View = $view; Home = $homeSet }
}

function Get-RollingDirectoryOwners {
    <#
        Get-DirectoryOwners for one layer of a rolling map: the nearest directory that holds a mapped file
        OTHER than Path, in any layer, and this layer's shards among its owners. With a single layer this is
        Get-DirectoryOwners exactly. $null (an orphan: escalate) only when no layer has a neighbour; an empty
        shard list means the neighbours are all executed by shards of other layers, which answer for themselves.
    #>
    param([Parameter(Mandatory)] $Rolling, [Parameter(Mandatory)] [string] $Path)

    $self = if ($Rolling.MappedPaths.Contains($Path)) { 1 } else { 0 }
    $segments = @(([string] $Path).Replace('\', '/').Split('/'))
    for ($depth = $segments.Count - 1; $depth -ge 2; $depth--) {
        $directory = ($segments[0..($depth - 1)] -join '/') + '/'
        if ($Rolling.DirectoryFileCount.ContainsKey($directory) -and $Rolling.DirectoryFileCount[$directory] - $self -gt 0) {
            return [pscustomobject]@{
                Directory = $directory.TrimEnd('/')
                Shards    = @($Rolling.DirectoryOwners[$directory] | Where-Object { $Rolling.Home.Contains($_) })
            }
        }
    }
    return $null
}

function Merge-RollingLayerSelections {
    <#
    .SYNOPSIS
        Pure. One selection from the per-layer selections of a rolling map.
    .DESCRIPTION
        Each entry of Layers is { Sha; Home; Selection; optional Index }: the layer's commit, the shards whose
        coverage it holds, Select-ImpactedShards' result for the change translated to that commit, and the
        layer's file index. A shard's fate is
        decided by its own layer alone - the only one whose line numbers describe it - so a layer's result is
        kept for its own shards plus the shards that run whatever the change (Mandatory: the anchor's
        always-run list, shards the map has never seen, and policy-required workloads).

        A layer that escalates cannot bound the change for ITS shards. When every layer does, that is the
        whole matrix and the result escalates as a single map's would. When only some do, exactly their shards
        run: the others were measured after whatever made the change unbounded, and stay selective.

        With a single layer the result equals Select-ImpactedShards' own (the self-test compares them).
    #>
    param(
        [Parameter(Mandatory)] [object[]] $Layers,
        [AllowEmptyCollection()] [string[]] $Mandatory = @()
    )

    $ordinal = [StringComparer]::Ordinal
    $always = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    foreach ($name in $Mandatory) { if ($name) { [void] $always.Add([string] $name) } }
    $escalatedLayers = @($Layers | Where-Object { [bool] $_.Selection.Escalate })
    $allEscalated = $Layers.Count -gt 0 -and $escalatedLayers.Count -eq $Layers.Count

    $selected = [System.Collections.Generic.SortedSet[string]]::new($ordinal)
    $routes = [System.Collections.Generic.List[string]]::new()
    $routeSeen = [System.Collections.Generic.HashSet[string]]::new($ordinal)
    $reasons = [System.Collections.Generic.List[string]]::new()
    $files = [ordered]@{}
    $untested = [ordered]@{}
    $newSources = [System.Collections.Generic.SortedSet[string]]::new($ordinal)
    $vacuous = ''
    $requiresValidation = $false
    $summaries = [System.Collections.Generic.List[object]]::new()
    $outcomeRank = @{ escalated = 4; routed = 3; unassigned = 2; reviewed = 1; skipped = 0 }
    foreach ($layer in $Layers) {
        $selection = $layer.Selection
        $homeShards = $layer.Home
        $short = ([string] $layer.Sha).Substring(0, [Math]::Min(8, ([string] $layer.Sha).Length))
        if ([bool] $selection.RequiresValidation) { $requiresValidation = $true }
        $own = [System.Collections.Generic.List[string]]::new()
        foreach ($value in @($selection.Shards)) {
            $name = [string] $value
            if ($homeShards.Contains($name) -or $always.Contains($name)) {
                [void] $selected.Add($name)
                if ($homeShards.Contains($name)) { [void] $own.Add($name) }
            }
        }
        foreach ($route in @($selection.Routes)) {
            $text = [string] $route
            $split = $text.IndexOf(' <= ', [StringComparison]::Ordinal)
            $name = if ($split -ge 0) { $text.Substring(0, $split) } else { '' }
            # '(none) <= ...' names no shard and is kept; a route for another layer's shard is that layer's to give.
            if ($split -ge 0 -and $name -cne '(none)' -and -not $homeShards.Contains($name) -and -not $always.Contains($name)) { continue }
            if ($routeSeen.Add($text)) { [void] $routes.Add($text) }
        }
        foreach ($reason in @($selection.Reasons)) { if ($reason -and $reasons -cnotcontains [string] $reason) { [void] $reasons.Add([string] $reason) } }
        if ([bool] $selection.Escalate -and -not $allEscalated) {
            $why = @($selection.Reasons | Where-Object { $_ } | Select-Object -First 1)
            foreach ($name in $homeShards) {
                [void] $selected.Add([string] $name)
                $text = "$name <= was measured at $short, and the change cannot be bounded for the shards measured there" +
                    $(if ($why.Count -gt 0) { " ($($why[0]))" } else { '' })
                if ($routeSeen.Add($text)) { [void] $routes.Add($text) }
            }
        }
        if ($selection.PSObject.Properties['Vacuous'] -and $selection.Vacuous -and -not $vacuous) { $vacuous = [string] $selection.Vacuous }
        if ($selection.PSObject.Properties['Files']) {
            foreach ($file in @($selection.Files)) {
                $kept = @($file.shards | Where-Object { $homeShards.Contains([string] $_) -or $always.Contains([string] $_) })
                if (-not $files.Contains([string] $file.path)) {
                    $files[[string] $file.path] = [pscustomobject]@{
                        path = [string] $file.path; category = [string] $file.category; outcome = [string] $file.outcome
                        shards = @($kept); why = [string] $file.why
                    }
                    continue
                }
                $merged = $files[[string] $file.path]
                $merged.shards = @(@($merged.shards) + $kept | Sort-Object -Unique)
                if ($outcomeRank[[string] $file.outcome] -gt $outcomeRank[[string] $merged.outcome]) { $merged.outcome = [string] $file.outcome }
                if ($file.why -and -not ([string] $merged.why).Contains([string] $file.why)) {
                    $merged.why = "$($merged.why) | at ${short}: $($file.why)"
                }
            }
        }
        if ($selection.PSObject.Properties['Untested']) {
            # A member is untested only when no layer's shards execute it. Line numbers differ per layer, so
            # the member is identified by file and name; the first layer's record is the one reported.
            $reported = [System.Collections.Generic.HashSet[string]]::new($ordinal)
            foreach ($entry in @($selection.Untested)) {
                $key = "$($entry.path)`n$($entry.member)"
                if (-not $reported.Add($key)) { continue }
                if (-not $untested.Contains($key)) { $untested[$key] = [pscustomobject]@{ Entry = $entry; Layers = 0 } }
                $untested[$key].Layers++
            }
        }
        if ($selection.PSObject.Properties['NewSourcePaths']) {
            foreach ($path in @($selection.NewSourcePaths)) { [void] $newSources.Add([string] $path) }
        }
        [void] $summaries.Add([pscustomobject]@{
            sha = [string] $layer.Sha; shards = $homeShards.Count; selected = $own.Count
            escalated = [bool] $selection.Escalate
            changedSinceMeasured = $(if ($layer.PSObject.Properties['ChangedCount']) { [int] $layer.ChangedCount } else { $null })
        })
    }

    $untestedEverywhere = foreach ($record in $untested.Values) {
        $path = [string] $record.Entry.path
        $indexedBy = @($Layers | Where-Object { -not $_.PSObject.Properties['Index'] -or $null -ne $_.Index.PSObject.Properties[$path] }).Count
        if ($record.Layers -ge $indexedBy) { $record.Entry }
    }
    $escalate = $allEscalated
    if (-not $escalate -and $requiresValidation -and $selected.Count -eq 0) {
        # No layer selected anything for a change that needs validating: the single-map rule, on the union.
        $escalate = $true
        [void] $reasons.Add($(if ($vacuous) { $vacuous } else { 'selection was empty' }))
    }
    return [pscustomobject]@{
        Escalate           = $escalate
        RequiresValidation = $requiresValidation
        Reasons            = $reasons
        Shards             = @($selected)
        Routes             = @($routes)
        Files              = @($files.Values)
        Untested           = @($untestedEverywhere)
        NewSourcePaths     = @($newSources)
        Layers             = @($summaries)
    }
}

# ------------------------------------------------------------- test routing

function ConvertTo-TestFilter {
    <#
        Parses a VSTest filter into an expression tree. VSTest's grammar: conditions
        Property(=|!=|~|!~)Value joined by '&' and '|', '&' binding tighter, with parentheses.
        Whitespace around tokens is insignificant (folded YAML inserts it at line breaks).
        Anything outside that grammar throws, and the caller then routes nothing - it escalates.
    #>
    param([Parameter(Mandatory)] [string] $Text)

    if ($Text.Contains('\')) { throw 'escaped filter characters are not supported' }
    $tokens = [System.Collections.Generic.List[string]]::new()
    foreach ($match in [regex]::Matches($Text, '[()&|]|[^()&|]+')) {
        $value = $match.Value.Trim()
        if ($value.Length -gt 0) { [void] $tokens.Add($value) }
    }

    function Read-Or {
        $items = [System.Collections.Generic.List[object]]::new()
        [void] $items.Add((Read-And))
        while ($script:filterPosition -lt $tokens.Count -and $tokens[$script:filterPosition] -eq '|') {
            $script:filterPosition++
            [void] $items.Add((Read-And))
        }
        if ($items.Count -eq 1) { return $items[0] }
        return [pscustomobject]@{ Kind = 'Or'; Items = @($items) }
    }
    function Read-And {
        $items = [System.Collections.Generic.List[object]]::new()
        [void] $items.Add((Read-Factor))
        while ($script:filterPosition -lt $tokens.Count -and $tokens[$script:filterPosition] -eq '&') {
            $script:filterPosition++
            [void] $items.Add((Read-Factor))
        }
        if ($items.Count -eq 1) { return $items[0] }
        return [pscustomobject]@{ Kind = 'And'; Items = @($items) }
    }
    function Read-Factor {
        if ($script:filterPosition -ge $tokens.Count) { throw 'filter ended where a condition was expected' }
        $token = $tokens[$script:filterPosition]
        if ($token -eq '(') {
            $script:filterPosition++
            $inner = Read-Or
            if ($script:filterPosition -ge $tokens.Count -or $tokens[$script:filterPosition] -ne ')') {
                throw 'unbalanced parenthesis in filter'
            }
            $script:filterPosition++
            return $inner
        }
        if ($token -in @(')', '&', '|')) { throw "unexpected '$token' in filter" }
        if ($token -notmatch '^(?<p>[A-Za-z][\w.]*)\s*(?<op>!=|!~|=|~)\s*(?<v>\S(?:.*\S)?)$') {
            throw "unparseable filter condition '$token'"
        }
        $script:filterPosition++
        return [pscustomobject]@{ Kind = 'Condition'; Property = $Matches.p; Operator = $Matches.op; Value = $Matches.v }
    }

    $script:filterPosition = 0
    $tree = Read-Or
    if ($script:filterPosition -ne $tokens.Count) { throw "unexpected '$($tokens[$script:filterPosition])' in filter" }
    return $tree
}

# Three-valued results. Unknown means "this test might match", so a shard is skipped only when its
# filter is definitely false for every test the changed file can contain.
$script:FilterFalse = 0
$script:FilterTrue = 1
$script:FilterUnknown = 2

function Invert-FilterResult {
    param([int] $Value)
    if ($Value -eq $script:FilterUnknown) { return $script:FilterUnknown }
    return 1 - $Value
}

function Test-OpenSuffixCanComplete {
    <#
        Whether Prefix followed by some unknown suffix can contain (or, with -Exact, equal) Value.
        AnySuffix admits any text; otherwise the suffix is one method identifier, which contains no
        '.' or '+', so a namespace or class term that is not already in the prefix cannot appear.
    #>
    param([string] $Prefix, [string] $Value, [bool] $AnySuffix, [switch] $Exact)

    for ($k = 0; $k -le $Value.Length; $k++) {
        $head = $Value.Substring(0, $k)
        $tail = $Value.Substring($k)
        $headFits = if ($Exact) { $Prefix.Equals($head, [StringComparison]::OrdinalIgnoreCase) }
                    else { $Prefix.EndsWith($head, [StringComparison]::OrdinalIgnoreCase) }
        if (-not $headFits) { continue }
        if ($AnySuffix -or $tail -match '^\w*$') { return $true }
    }
    return $false
}

function Test-FilterStringCondition {
    <#
        Actual is either a test's complete fully qualified name, or - when PrefixOnly - the known
        start of names whose remainder cannot be read from this file (inherited tests, or a test
        whose declaration could not be read). Each comparison is made both ordinally and ignoring
        case; if the two disagree the result is Unknown, so neither VSTest casing behaviour can
        make a matching test look excluded.
    #>
    param([string] $Actual, [bool] $PrefixOnly, [bool] $AnySuffix, [string] $Operator, [string] $Value)

    $negated = $Operator.StartsWith('!')
    if ($Operator.EndsWith('~')) {
        $ordinal = $Actual.Contains($Value, [StringComparison]::Ordinal)
        $ignoreCase = $Actual.Contains($Value, [StringComparison]::OrdinalIgnoreCase)
        $result = if ($ordinal) { $script:FilterTrue }
                  elseif ($ignoreCase) { $script:FilterUnknown }
                  elseif ($PrefixOnly -and (Test-OpenSuffixCanComplete -Prefix $Actual -Value $Value -AnySuffix $AnySuffix)) {
                      $script:FilterUnknown
                  }
                  else { $script:FilterFalse }
    }
    else {
        if ($PrefixOnly) {
            $result = if (Test-OpenSuffixCanComplete -Prefix $Actual -Value $Value -AnySuffix $AnySuffix -Exact) {
                          $script:FilterUnknown
                      }
                      else { $script:FilterFalse }
        }
        else {
            $ordinal = $Actual.Equals($Value, [StringComparison]::Ordinal)
            $ignoreCase = $Actual.Equals($Value, [StringComparison]::OrdinalIgnoreCase)
            $result = if ($ordinal) { $script:FilterTrue }
                      elseif ($ignoreCase) { $script:FilterUnknown }
                      else { $script:FilterFalse }
        }
    }
    if ($negated) { return Invert-FilterResult $result }
    return $result
}

function Test-TestFilter {
    param([Parameter(Mandatory)] $Node, [Parameter(Mandatory)] $Candidate)

    switch ($Node.Kind) {
        'And' {
            $sawUnknown = $false
            foreach ($item in $Node.Items) {
                $value = Test-TestFilter -Node $item -Candidate $Candidate
                if ($value -eq $script:FilterFalse) { return $script:FilterFalse }
                if ($value -eq $script:FilterUnknown) { $sawUnknown = $true }
            }
            if ($sawUnknown) { return $script:FilterUnknown }
            return $script:FilterTrue
        }
        'Or' {
            $sawUnknown = $false
            foreach ($item in $Node.Items) {
                $value = Test-TestFilter -Node $item -Candidate $Candidate
                if ($value -eq $script:FilterTrue) { return $script:FilterTrue }
                if ($value -eq $script:FilterUnknown) { $sawUnknown = $true }
            }
            if ($sawUnknown) { return $script:FilterUnknown }
            return $script:FilterFalse
        }
        'Condition' {
            if ($Node.Property -ieq 'FullyQualifiedName') {
                return Test-FilterStringCondition -Actual $Candidate.Fqn -PrefixOnly $Candidate.PrefixOnly -AnySuffix $Candidate.AnySuffix `
                    -Operator $Node.Operator -Value $Node.Value
            }
            if ($Node.Property -ieq 'Category') {
                # Categories are known only as the union over the whole file, so a category that
                # appears somewhere in the file may or may not be on this particular test.
                if ($null -eq $Candidate.Categories) { return $script:FilterUnknown }
                $present = $false
                foreach ($category in $Candidate.Categories) {
                    $hit = if ($Node.Operator.EndsWith('~')) {
                        $category.Contains($Node.Value, [StringComparison]::OrdinalIgnoreCase)
                    } else {
                        $category.Equals($Node.Value, [StringComparison]::OrdinalIgnoreCase)
                    }
                    if ($hit) { $present = $true; break }
                }
                if ($present) { return $script:FilterUnknown }
                if ($Node.Operator.StartsWith('!')) { return $script:FilterTrue }
                return $script:FilterFalse
            }
            # Any other property is not modelled, so it can never be what excludes a test.
            return $script:FilterUnknown
        }
        default { throw "unknown filter node '$($Node.Kind)'" }
    }
}

function Skip-CSharpHoleToken {
    <#
        Advances one token inside an interpolation hole, where the text is CODE. Returns the new
        index and adjusts hole depth through the reference.

        Comments count here: `$"{/* } " */ 0}"` is valid C#, and treating the brace and quote
        inside that comment as code ends the literal early, which surfaces as 'unbalanced braces'
        and escalates selection to the full matrix. Nested literals recurse so their own quotes
        and braces cannot be mistaken for the enclosing literal's.
    #>
    param(
        [Parameter(Mandatory)] [string] $Text,
        [Parameter(Mandatory)] [int] $Index,
        [Parameter(Mandatory)] [ref] $HoleDepth
    )

    $n = $Text.Length
    $i = $Index
    $ch = $Text[$i]

    if ($ch -eq '/' -and $i + 1 -lt $n -and $Text[$i + 1] -eq '/') {
        $e = $Text.IndexOf("`n", $i)
        return $(if ($e -lt 0) { $n } else { $e })
    }
    if ($ch -eq '/' -and $i + 1 -lt $n -and $Text[$i + 1] -eq '*') {
        $e = $Text.IndexOf('*/', $i + 2)
        return $(if ($e -lt 0) { $n } else { $e + 2 })
    }
    if ($ch -eq '"' -or $ch -eq "'" -or $ch -eq '$' -or $ch -eq '@') {
        $inner = Get-CSharpLiteralEnd -Text $Text -Start $i
        if ($inner -gt $i) { return $inner }
    }
    if ($ch -eq '{') { $HoleDepth.Value++ }
    elseif ($ch -eq '}') { $HoleDepth.Value-- }
    return $i + 1
}

function Get-CSharpLiteralEnd {
    <#
        Returns the index one past the end of the C# string or character literal starting at
        $Start, or $Start when nothing starts there. Understands raw ("""), verbatim (@"),
        interpolated ($") and combined forms.

        Interpolation holes are CODE, not text: `$"{string.Join(", ", xs)}"` contains a nested
        string whose quotes must not end the outer literal. A regex cannot express that, and the
        one this replaced ended the literal at the first inner quote - which swallowed the hole's
        opening brace, left its closing brace as code, and made Get-CSharpTestShape report
        'unbalanced braces' for NeuralNetworkModelTestBase.cs. That is an escalation to the full
        matrix for every change that reaches it.
    #>
    param([Parameter(Mandatory)] [string] $Text, [Parameter(Mandatory)] [int] $Start)

    $n = $Text.Length
    $i = $Start

    if ($Text[$i] -eq "'") {
        $i++
        while ($i -lt $n) {
            if ($Text[$i] -eq '\') { $i += 2; continue }
            if ($Text[$i] -eq "'") { return $i + 1 }
            if ($Text[$i] -eq "`n") { return $i }
            $i++
        }
        return $n
    }

    $dollars = 0
    $verbatim = $false
    while ($i -lt $n -and ($Text[$i] -eq '$' -or $Text[$i] -eq '@')) {
        if ($Text[$i] -eq '$') { $dollars++ } else { $verbatim = $true }
        $i++
    }
    $interpolated = $dollars -gt 0
    if ($i -ge $n -or $Text[$i] -ne '"') { return $Start }

    $quotes = 0
    while ($i + $quotes -lt $n -and $Text[$i + $quotes] -eq '"') { $quotes++ }
    $holeDepth = 0
    if ($quotes -ge 3) {
        # Raw literal. Scanning rather than IndexOf on the fence: in a raw INTERPOLATED literal a
        # hole is code and may itself contain a fence, so `$"""{ """ }"""` would otherwise end at
        # the inner one. C# opens a hole with as many braces as there are leading '$'.
        $i += $quotes
        while ($i -lt $n) {
            $ch = $Text[$i]
            if ($holeDepth -gt 0) {
                $i = Skip-CSharpHoleToken -Text $Text -Index $i -HoleDepth ([ref] $holeDepth)
                continue
            }
            if ($interpolated -and $ch -eq '{') {
                $run = 0
                while ($i + $run -lt $n -and $Text[$i + $run] -eq '{') { $run++ }
                if ($run -ge $dollars) { $holeDepth++; $i += $dollars } else { $i += $run }
                continue
            }
            if ($ch -eq '"') {
                $run = 0
                while ($i + $run -lt $n -and $Text[$i + $run] -eq '"') { $run++ }
                if ($run -ge $quotes) { return $i + $quotes }
                $i += $run
                continue
            }
            $i++
        }
        return $n
    }

    $i++
    while ($i -lt $n) {
        $ch = $Text[$i]

        if ($holeDepth -gt 0) {
            $i = Skip-CSharpHoleToken -Text $Text -Index $i -HoleDepth ([ref] $holeDepth)
            continue
        }

        if ($verbatim) {
            if ($ch -eq '"') {
                if ($i + 1 -lt $n -and $Text[$i + 1] -eq '"') { $i += 2; continue }
                return $i + 1
            }
        }
        else {
            if ($ch -eq '\') { $i += 2; continue }
            if ($ch -eq '"') { return $i + 1 }
            if ($ch -eq "`n") { return $i }
        }

        if ($interpolated) {
            if ($ch -eq '{') {
                if ($i + 1 -lt $n -and $Text[$i + 1] -eq '{') { $i += 2; continue }
                $holeDepth++; $i++; continue
            }
            if ($ch -eq '}' -and $i + 1 -lt $n -and $Text[$i + 1] -eq '}') { $i += 2; continue }
        }

        $i++
    }
    return $n
}

function ConvertTo-CodeOnlyCSharp {
    <#
        Blanks the CONTENT of comments, strings and character literals (newlines are kept, so
        offsets and line numbers survive), leaving only code. Brace matching and declaration
        matching then cannot be fooled by '{' in a string or 'class X' in a comment.

        Scanned rather than pattern-matched: see Get-CSharpLiteralEnd for why a regex cannot
        classify an interpolation hole correctly.
    #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Text, [switch] $PreserveStrings)

    $n = $Text.Length
    if ($n -eq 0) { return $Text }
    $chars = $Text.ToCharArray()
    $i = 0

    while ($i -lt $n) {
        $ch = $Text[$i]
        $from = $i
        $to = -1
        $isComment = $false

        if ($ch -eq '/' -and $i + 1 -lt $n -and $Text[$i + 1] -eq '/') {
            $isComment = $true
            $to = $Text.IndexOf("`n", $i)
            if ($to -lt 0) { $to = $n }
        }
        elseif ($ch -eq '/' -and $i + 1 -lt $n -and $Text[$i + 1] -eq '*') {
            $isComment = $true
            $to = $Text.IndexOf('*/', $i + 2)
            $to = if ($to -lt 0) { $n } else { $to + 2 }
        }
        elseif ($ch -eq '"' -or $ch -eq "'" -or $ch -eq '$' -or $ch -eq '@') {
            $end = Get-CSharpLiteralEnd -Text $Text -Start $i
            if ($end -gt $i) { $to = $end }
        }

        if ($to -lt 0) { $i++; continue }
        if (-not ($PreserveStrings -and -not $isComment)) {
            for ($k = $from; $k -lt $to; $k++) {
                $c = $chars[$k]
                if ($c -ne "`r" -and $c -ne "`n") { $chars[$k] = ' ' }
            }
        }
        $i = $to
    }

    return [string]::new($chars)
}

function Get-CSharpTestShape {
    <#
        Reads, from one C# test source, what VSTest filters can see: every test's fully qualified
        name, the Category traits the file can carry, and the top-level types other files could
        reference. Returns ParseError instead of guessing when the braces do not balance.
    #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Text,
        [bool] $GeneratedDependenciesKnown = $false)

    $code = ConvertTo-CodeOnlyCSharp -Text $Text
    $shape = [pscustomobject]@{
        Types          = [System.Collections.Generic.List[object]]::new()
        Tests          = [System.Collections.Generic.List[object]]::new()
        Categories     = $null
        Hazard         = $null
        ParseError     = $null
        Collections    = [System.Collections.Generic.List[string]]::new()
    }

    # Brace pairs by position.
    $open = [System.Collections.Generic.Stack[int]]::new()
    $closeOf = @{}
    foreach ($brace in [regex]::Matches($code, '[{}]')) {
        if ($brace.Value -eq '{') { $open.Push($brace.Index); continue }
        if ($open.Count -eq 0) { $shape.ParseError = 'unbalanced braces'; return $shape }
        $closeOf[$open.Pop()] = $brace.Index
    }
    if ($open.Count -ne 0) { $shape.ParseError = 'unbalanced braces'; return $shape }

    function Get-BodyRange([int] $From) {
        # The first '{' after a declaration opens its body, unless a ';' ends the declaration first.
        for ($k = $From; $k -lt $code.Length; $k++) {
            if ($code[$k] -eq ';') { return $null }
            if ($code[$k] -eq '{') { return @($k, [int] $closeOf[$k]) }
        }
        return $null
    }

    $namespaces = [System.Collections.Generic.List[object]]::new()
    $fileNamespace = ''
    foreach ($match in [regex]::Matches($code, '(?m)^[ \t]*namespace\s+(?<n>[A-Za-z_][\w.]*)\s*(?<t>[;{])')) {
        if ($match.Groups['t'].Value -eq ';') { $fileNamespace = $match.Groups['n'].Value; continue }
        $start = $match.Groups['t'].Index
        [void] $namespaces.Add([pscustomobject]@{ Name = $match.Groups['n'].Value; Start = $start; End = [int] $closeOf[$start] })
    }

    $typePattern = '\b(?<kind>class|struct|interface|enum|record(?:\s+(?:class|struct))?)\s+(?<name>[A-Za-z_]\w*)'
    foreach ($match in [regex]::Matches($code, $typePattern)) {
        # 'where T : class where U : new()' puts a keyword where a type name would be.
        if ($match.Groups['name'].Value -cin @('where', 'new', 'class', 'struct', 'unmanaged', 'notnull')) { continue }
        $body = Get-BodyRange ($match.Index + $match.Length)
        $header = if ($null -ne $body) { $code.Substring($match.Index + $match.Length, $body[0] - $match.Index - $match.Length) } else { '' }
        # Modifiers run back to the end of the previous declaration or attribute list, not to the
        # line start: 'public abstract' written on the line above 'class' is still one declaration,
        # and reading only the keyword's line missed the abstract-base and file-local hazards.
        $boundary = if ($match.Index -gt 0) { $code.LastIndexOfAny([char[]] ';{}]', $match.Index - 1) } else { -1 }
        $modifiers = $code.Substring($boundary + 1, $match.Index - $boundary - 1)

        # The base list follows the name, after any type parameters and primary constructor, and
        # before any constraint clause. Interfaces are recognised by the I-prefix convention; any
        # other base is a class whose tests and traits this type inherits from elsewhere.
        $baseText = [regex]::Replace($header, '<[^<>]*>', '')
        while ($baseText -match '\([^()]*\)') { $baseText = [regex]::Replace($baseText, '\([^()]*\)', '') }
        $baseText = ($baseText -split '\bwhere\b')[0]
        $hasClassBase = $false
        $colon = $baseText.IndexOf(':')
        if ($colon -ge 0) {
            foreach ($base in $baseText.Substring($colon + 1).Split(',')) {
                $baseName = ($base.Trim() -split '\.')[-1]
                if ($baseName -and $baseName -cnotmatch '^I[A-Z]') { $hasClassBase = $true }
            }
        }

        [void] $shape.Types.Add([pscustomobject]@{
            Name         = $match.Groups['name'].Value
            Kind         = ($match.Groups['kind'].Value -split '\s+')[0]
            Start        = $match.Index
            BodyStart    = if ($null -ne $body) { $body[0] } else { -1 }
            BodyEnd      = if ($null -ne $body) { $body[1] } else { -1 }
            FileLocal    = $modifiers -match '\bfile\b'
            IsAbstract   = $modifiers -match '\babstract\b'
            HasClassBase = $hasClassBase
            Parent       = $null
            Namespace    = $fileNamespace
        })
    }

    # Nesting and namespaces by containment.
    foreach ($type in $shape.Types) {
        $innermost = $null
        foreach ($candidate in $shape.Types) {
            if ($candidate -eq $type -or $candidate.BodyStart -lt 0) { continue }
            if ($type.Start -gt $candidate.BodyStart -and $type.Start -lt $candidate.BodyEnd) {
                if ($null -eq $innermost -or $candidate.BodyStart -gt $innermost.BodyStart) { $innermost = $candidate }
            }
        }
        $type.Parent = $innermost
        foreach ($namespace in $namespaces) {
            if ($type.Start -gt $namespace.Start -and $type.Start -lt $namespace.End) { $type.Namespace = $namespace.Name }
        }
    }

    function Get-TypeChain($Type) {
        $names = [System.Collections.Generic.List[string]]::new()
        for ($t = $Type; $null -ne $t; $t = $t.Parent) { $names.Insert(0, $t.Name) }
        return ($names -join '+')
    }
    function Get-TypePrefix($Type) {
        $chain = Get-TypeChain $Type
        if ($Type.Namespace) { return "$($Type.Namespace).$chain" }
        return $chain
    }

    # Test methods: an attribute list naming a *Fact or *Theory attribute, then the method whose
    # parameter list is the next '(' - its name is the last identifier before it. An attribute
    # list only starts a declaration, so it must follow a line start, '{', '}', ';' or ']'.
    foreach ($attribute in [regex]::Matches($code, '\[(?<body>[^\[\]]*)\]')) {
        $isTest = $false
        foreach ($part in $attribute.Groups['body'].Value.Split(',')) {
            if ($part.Trim() -match '^(?:[\w.]+\.)?\w*(?:Fact|Theory)(?:Attribute)?\s*(?:\(|$)') { $isTest = $true }
        }
        if (-not $isTest) { continue }
        $before = $code.Substring(0, $attribute.Index).TrimEnd(' ', "`t")
        if ($before.Length -gt 0 -and $before[-1] -notin @("`n", "`r", '{', '}', ';', ']')) { continue }

        $after = $attribute.Index + $attribute.Length
        $owner = $null
        foreach ($type in $shape.Types) {
            if ($type.BodyStart -lt 0) { continue }
            if ($after -gt $type.BodyStart -and $after -lt $type.BodyEnd) {
                if ($null -eq $owner -or $type.BodyStart -gt $owner.BodyStart) { $owner = $type }
            }
        }

        # Skip any further attribute lists (bracket depth counted, so '[InlineData(new[] { 1 })]'
        # is one list), then read the name before the parameter list.
        $position = $after
        while ($true) {
            while ($position -lt $code.Length -and [char]::IsWhiteSpace($code[$position])) { $position++ }
            if ($position -ge $code.Length -or $code[$position] -ne '[') { break }
            $depth = 0
            do {
                if ($code[$position] -eq '[') { $depth++ } elseif ($code[$position] -eq ']') { $depth-- }
                $position++
            } while ($depth -gt 0 -and $position -lt $code.Length)
        }
        $method = $null
        $paren = $code.IndexOf('(', $position)
        if ($paren -ge 0) {
            $signature = [regex]::Replace($code.Substring($position, $paren - $position), '<[^<>]*>\s*$', '')
            if ($signature -notmatch '[;{}=\[\]]' -and $signature -match '(?<m>[A-Za-z_]\w*)\s*$') { $method = $Matches.m }
        }

        # A test whose name cannot be read still exists. It is kept as an open-ended candidate so a
        # filter naming its method can never be what makes its shard look unaffected.
        if ($null -eq $owner) {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = $(if ($fileNamespace) { "$fileNamespace." } else { '' }); PrefixOnly = $true; AnySuffix = $true })
        }
        elseif ($null -eq $method) {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = "$(Get-TypePrefix $owner)."; PrefixOnly = $true; AnySuffix = $false })
        }
        else {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = "$(Get-TypePrefix $owner).$method"; PrefixOnly = $false; AnySuffix = $false })
        }
    }

    # A class with a class base can inherit tests whose method names live in another file.
    foreach ($type in $shape.Types) {
        if ($type.Kind -in @('class', 'record') -and $type.HasClassBase) {
            [void] $shape.Tests.Add([pscustomobject]@{ Fqn = "$(Get-TypePrefix $type)."; PrefixOnly = $true; AnySuffix = $false })
        }
    }

    # Categories are knowable only when every Category trait is a literal and no class inherits
    # traits from a base declared elsewhere.
    $categories = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $categoriesKnown = -not @($shape.Types | Where-Object { $_.HasClassBase }).Count
    foreach ($trait in [regex]::Matches($Text, '\bTrait(?:Attribute)?\s*\(\s*"(?i:category)"\s*,\s*(?<v>[^)]*)\)')) {
        $value = $trait.Groups['v'].Value.Trim()
        if ($value -match '^"(?<c>[^"]*)"$') { [void] $categories.Add($Matches.c) }
        else { $categoriesKnown = $false }
    }
    if ($categoriesKnown) { $shape.Categories = @($categories) }

    # Constructs that reach tests without naming this file's types, which reference search cannot
    # follow: extension methods, global usings, assembly-level attributes, module initializers, and
    # xUnit collection definitions (bound by string name).
    $hazards = [ordered]@{
        'declares extension methods'           = '\(\s*this\s+[A-Za-z_]'
        'declares global usings'               = '(?m)^\s*global\s+using\b'
        'declares assembly-level attributes'   = '\[\s*assembly\s*:'
        'declares a module initializer'        = '\bModuleInitializer\b'
    }
    foreach ($hazard in $hazards.Keys) {
        if ($code -match $hazards[$hazard]) { $shape.Hazard = $hazard; break }
    }
    # Use code-only offsets so examples in comments/string literals are not attributes.
    $definitions = @([regex]::Matches($code, '\bCollectionDefinition(?:Attribute)?\b'))
    foreach ($definition in $definitions) {
        $tail = $Text.Substring($definition.Index)
        $literal = [regex]::Match($tail, '^CollectionDefinition(?:Attribute)?\s*\(\s*"(?<name>[A-Za-z0-9_. -]+)"\s*(?:,\s*DisableParallelization\s*=\s*(?:true|false)\s*)?\)')
        $nameof = [regex]::Match($tail, '^CollectionDefinition(?:Attribute)?\s*\(\s*nameof\(\s*(?:[\w.]+\.)?(?<name>[A-Za-z_]\w*)\s*\)\s*(?:,\s*DisableParallelization\s*=\s*(?:true|false)\s*)?\)')
        if ($literal.Success) { $shape.Collections.Add($literal.Groups['name'].Value) }
        elseif ($nameof.Success) { $shape.Collections.Add($nameof.Groups['name'].Value) }
        else { $shape.Hazard = 'declares an unresolved xUnit collection definition' }
    }
    if ($shape.Collections.Count -gt 0 -and -not $GeneratedDependenciesKnown) {
        $shape.Hazard = 'declares an xUnit collection definition without reviewed generated dependencies'
    }
    # An abstract class is a base by construction, and the scaffold generator derives test classes
    # from bases whose names it can compose at build time. Those derived classes are not in the
    # tree, so reference search cannot find them.
    if (-not $GeneratedDependenciesKnown -and -not $shape.Hazard -and @($shape.Types | Where-Object { $_.IsAbstract }).Count -gt 0) {
        $shape.Hazard = 'declares an abstract class, which build-time generated test classes may derive from'
    }
    return $shape
}

function Get-ShardProjectDirectory {
    param([Parameter(Mandatory)] $Shard)
    $project = ([string] $Shard.project).Replace('\', '/')
    $slash = $project.LastIndexOf('/')
    if ($slash -lt 0) { return '' }
    return $project.Substring(0, $slash + 1)
}

function Get-TestProjectDirectory {
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Manifest)

    $best = $null
    foreach ($shard in $Manifest) {
        $directory = Get-ShardProjectDirectory -Shard $shard
        if ($directory -and $Path.StartsWith($directory, [StringComparison]::OrdinalIgnoreCase) -and
            ($null -eq $best -or $directory.Length -gt $best.Length)) {
            $best = $directory
        }
    }
    return $best
}

function Get-TestFileRoutes {
    <#
        Routes changed test sources to the shards whose manifest filters select their tests.

        A test file affects the tests it declares and, through its types, every test file that uses
        them: a base class, fixture or helper changes the behaviour of its consumers. So routing
        follows the transitive closure of files that name this file's top-level types, and unions
        the shards that select any test in that closure. Constructs whose consumers cannot be found
        by name (see Get-CSharpTestShape hazards), a closure too wide to be selective, a file whose
        text cannot be parsed, and tests no shard filter selects all return Routable = $false,
        which the selector turns into an escalation.

        FindBuildTimeReferences reports which of a file's type names build-time code (the source
        generators) mentions: tests the generator emits exist only inside the compiler, so a type
        they depend on cannot be routed by reading the tree.

        ReadFile, FindReferrers and FindBuildTimeReferences are injected so the self-test can
        exercise routing without a repository; in production they read HEAD and use git grep.
    #>
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Paths,
        [Parameter(Mandatory)] [AllowEmptyCollection()] [object[]] $Manifest,
        [Parameter(Mandatory)] [scriptblock] $ReadFile,
        [scriptblock] $ReadPriorFile,
        [Parameter(Mandatory)] [scriptblock] $FindReferrers,
        [Parameter(Mandatory)] [scriptblock] $FindBuildTimeReferences,
        [scriptblock] $FindCollectionReferrers,
        [bool] $GeneratedDependenciesKnown = $false,
        [string[]] $GeneratedDependencyProjects = @(),
        [int] $MaximumClosure = 200
    )

    $filters = @{}
    foreach ($shard in $Manifest) {
        try { $filters[[string] $shard.name] = ConvertTo-TestFilter -Text ([string] $shard.filter) }
        catch { $filters[[string] $shard.name] = $null }
    }

    $routes = @{}
    foreach ($path in $Paths) {
        $normalized = $path.Replace('\', '/')
        $projectDirectory = Get-TestProjectDirectory -Path $normalized -Manifest $Manifest
        if ($null -eq $projectDirectory) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'test source belongs to no project any shard runs' }
            continue
        }
        $projectShards = @($Manifest | Where-Object { (Get-ShardProjectDirectory -Shard $_) -ieq $projectDirectory })
        $projectGenerationKnown = $GeneratedDependenciesKnown -and $projectDirectory -cin $GeneratedDependencyProjects

        $visited = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        $queue = [System.Collections.Generic.Queue[string]]::new()
        [void] $visited.Add($normalized)
        $queue.Enqueue($normalized)
        $candidates = [System.Collections.Generic.List[object]]::new()
        $failure = $null
        $ownTests = 0
        $ownReferrers = 0

        while ($queue.Count -gt 0 -and $null -eq $failure) {
            $file = $queue.Dequeue()
            $text = & $ReadFile $file
            if ($null -eq $text) { $failure = "cannot read test source $file"; break }
            $shape = Get-CSharpTestShape -Text $text -GeneratedDependenciesKnown $projectGenerationKnown
            if ($shape.ParseError) { $failure = "cannot parse test source $file ($($shape.ParseError))"; break }
            if ($shape.Hazard) { $failure = "test source $file $($shape.Hazard), whose consumers cannot be found by name"; break }
            $priorShape = $null
            if ($file -ieq $normalized -and $null -ne $ReadPriorFile) {
                $priorText = & $ReadPriorFile $file
                if ($null -ne $priorText) {
                    $priorShape = Get-CSharpTestShape -Text $priorText -GeneratedDependenciesKnown $projectGenerationKnown
                    if ($priorShape.ParseError -or $priorShape.Hazard) {
                        $failure = "prior test source $file cannot be routed: $($priorShape.ParseError) $($priorShape.Hazard)"
                        break
                    }
                }
            }

            foreach ($test in $shape.Tests) {
                [void] $candidates.Add([pscustomobject]@{ Fqn = $test.Fqn; PrefixOnly = $test.PrefixOnly; AnySuffix = $test.AnySuffix; Categories = $shape.Categories })
            }
            if ($file -ieq $normalized) { $ownTests = $shape.Tests.Count }

            $names = @($shape.Types | Where-Object { $null -eq $_.Parent -and -not $_.FileLocal } |
                ForEach-Object { $_.Name } | Sort-Object -Unique)
            $collectionNames = @($shape.Collections)
            if ($null -ne $priorShape) {
                $names = @(@($names) + @($priorShape.Types | Where-Object { $null -eq $_.Parent -and -not $_.FileLocal } | ForEach-Object Name) | Sort-Object -Unique)
                $collectionNames = @(@($collectionNames) + @($priorShape.Collections) | Sort-Object -Unique)
            }
            $dependencyNames = @($names) + @($collectionNames)
            if ($dependencyNames.Count -eq 0) { continue }
            $generated = @(& $FindBuildTimeReferences $dependencyNames)
            if ($generated.Count -gt 0) {
                $failure = "test source $file declares $($generated -join ', '), which a source generator references, so tests generated at build time may depend on it"
                break
            }
            $consumerPaths = @(if ($names.Count -gt 0) { & $FindReferrers $names $projectDirectory })
            if ($collectionNames.Count -gt 0) {
                if ($null -eq $FindCollectionReferrers) { $failure = 'collection consumer discovery is unavailable'; break }
                $consumerPaths += @(& $FindCollectionReferrers $collectionNames $projectDirectory)
            }
            foreach ($referrer in $consumerPaths) {
                $referrerPath = ([string] $referrer).Replace('\', '/')
                if ($referrerPath -ieq $file) { continue }
                if ($file -ieq $normalized) { $ownReferrers++ }
                if ($visited.Add($referrerPath)) {
                    if ($visited.Count -gt $MaximumClosure) {
                        $failure = "test source is shared by more than $MaximumClosure test files"
                        break
                    }
                    $queue.Enqueue($referrerPath)
                }
            }
        }

        if ($null -ne $failure) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = $failure }
            continue
        }
        if ($ownTests -eq 0 -and $ownReferrers -eq 0) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'test support source declares no tests and no test file names its types' }
            continue
        }

        $shards = [System.Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
        $unparsed = $false
        foreach ($shard in $projectShards) {
            $filter = $filters[[string] $shard.name]
            if ($null -eq $filter) { $unparsed = $true; break }
            foreach ($candidate in $candidates) {
                if ((Test-TestFilter -Node $filter -Candidate $candidate) -ne $script:FilterFalse) {
                    [void] $shards.Add([string] $shard.name)
                    break
                }
            }
        }
        if ($unparsed) {
            $routes[$path] = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'a shard filter for this test project cannot be parsed' }
        }
        elseif ($shards.Count -eq 0) {
            # Not an escalation: the full matrix would not run these tests either. The Build job's inventory gate
            # (Assert-ShardInventory.ps1) fails the pull request until a shard filter selects them.
            $routes[$path] = [pscustomobject]@{ Routable = $false; Unassigned = $true; Shards = @(); Why = 'no shard filter selects any test this source affects' }
        }
        else {
            $closure = $visited.Count - 1
            $why = if ($closure -gt 0) { "its filter selects tests in this file or in $closure test file(s) that use it" }
                   else { 'its filter selects tests in this file' }
            $routes[$path] = [pscustomobject]@{ Routable = $true; Shards = @($shards); Why = $why }
        }
    }
    return $routes
}

function Get-GitFileText {
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Revisions)
    foreach ($revision in $Revisions) {
        if ([string]::IsNullOrWhiteSpace($revision)) { continue }
        $lines = @(& git -c core.quotepath=false show "${revision}:$Path" 2>$null)
        if ($LASTEXITCODE -eq 0) { return ($lines -join "`n") }
    }
    return $null
}

function Get-GitSourceText {
    <#
        A file's text at a revision for the C# parser, or $null. Read as UTF-8 whatever the console's
        code page is: a byte-order mark read through another code page is three stray characters,
        a syntax error on line 1 that sends the whole file back to the file-wide rule. The mark
        itself is dropped; it is not a line, so no line number moves.
    #>
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [string] $Revision)
    $saved = $null
    try {
        $saved = [Console]::OutputEncoding
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    }
    catch { $saved = $null }
    try { $text = Get-GitFileText -Path $Path -Revisions @($Revision) }
    finally { if ($null -ne $saved) { try { [Console]::OutputEncoding = $saved } catch { } } }
    if ($null -ne $text -and $text.Length -gt 0 -and $text[0] -eq [char] 0xFEFF) { $text = $text.Substring(1) }
    return $text
}

function Find-GitReferrers {
    param([Parameter(Mandatory)] [string[]] $Names, [Parameter(Mandatory)] [string] $Directory)
    $arguments = @('-c', 'core.quotepath=false', 'grep', '-l', '-w', '-F')
    foreach ($name in $Names) { $arguments += @('-e', $name) }
    $arguments += @('HEAD', '--', ":(glob)$Directory**/*.cs")
    $output = @(& git @arguments 2>$null)
    # git grep exits 1 when nothing matches, which is an answer, not a failure.
    if ($LASTEXITCODE -gt 1) { throw "git grep for test-type references failed with exit code $LASTEXITCODE" }
    return @($output | ForEach-Object { ([string] $_) -replace '^HEAD:', '' })
}

function Test-GeneratedDependencyInputPath {
    param([Parameter(Mandatory)] [string] $Path)
    # The inputs that decide which bases and collections build-time generated test classes use: the
    # generators themselves and every project/build file that configures them.
    return $Path -match '^(src/AiDotNet\.Generators/|(.*/)?global\.json$)' -or
        $Path -match '\.(csproj|props|targets)$'
}

function Test-ReviewedGeneratedDependencies {
    param([Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $ChangedPaths)
    # Reviewed emitter contract, NOT an exemption for a particular base/collection:
    # TestScaffoldGenerator's GetBaseClassName and algorithm switch return literal
    # base names; remaining emitters spell their base/collection dependencies directly.
    # Find-GitBuildTimeReferences therefore detects every generated dependency for the
    # generator revision this change builds with. Other generators augment existing
    # declarations, not independent xUnit descendants.
    #
    # Fail closed when THIS change edits a generator or build input, because then the
    # contract above is exactly what may no longer hold. It used to pin a SHA-256 of every
    # generator and project file instead; master's next package bump changed that digest,
    # so from the day it merged the contract was "unknown" for every pull request and each
    # edit to a collection definition or abstract test base ran the full matrix (#2243).
    foreach ($path in $ChangedPaths) {
        if (Test-GeneratedDependencyInputPath -Path $path) { return $false }
    }
    return $true
}
function Test-CSharpCollectionConsumer {
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Text,
        [Parameter(Mandatory)] [string[]] $Names)
    $code = ConvertTo-CodeOnlyCSharp $Text
    $uses = @([regex]::Matches($code, '\bCollection(?:Attribute)?\b'))
    foreach ($use in $uses) {
        # A constant, nameof, alias or escaped/computed name may equal any requested
        # collection. Include its file rather than guessing the expression's value.
        $literal = [regex]::Match($Text.Substring($use.Index), '^Collection(?:Attribute)?\s*\(\s*"(?<name>[A-Za-z0-9_. -]+)"\s*\)')
        if (-not $literal.Success -or $literal.Groups['name'].Value -in $Names) { return $true }
    }
    return $false
}

function Find-GitCollectionReferrers {
    param([Parameter(Mandatory)] [string[]] $Names, [Parameter(Mandatory)] [string] $Directory)
    # Cache source, not answers: multiple collection definitions can share the index.
    if (-not (Get-Variable CollectionSourceIndex -Scope Script -ErrorAction SilentlyContinue)) {
        $script:CollectionSourceIndex = @{}
    }
    if (-not $script:CollectionSourceIndex.ContainsKey($Directory)) {
        $paths = @(& git -c core.quotepath=false grep -l -w -E 'Collection(Attribute)?' HEAD -- ":(glob)$Directory**/*.cs" 2>$null)
        if ($LASTEXITCODE -gt 1) { throw 'Cannot enumerate collection consumers.' }
        $index = @{}
        foreach ($path in $paths) {
            $file = ([string] $path) -replace '^HEAD:', ''
            $text = Get-GitFileText -Path $file -Revisions @('HEAD')
            if ($null -eq $text) { throw "Cannot read collection consumer $file" }
            $index[$file] = $text
        }
        $script:CollectionSourceIndex[$Directory] = $index
    }
    foreach ($entry in $script:CollectionSourceIndex[$Directory].GetEnumerator()) {
        if (Test-CSharpCollectionConsumer -Text $entry.Value -Names $Names) { $entry.Key }
    }
}

function Find-GitBuildTimeReferences {
    <#
        Which of Names build-time code mentions. One search per name keeps the answer exact; the
        lists are short (a file's top-level types) and git grep over the generator tree is fast.
    #>
    param([Parameter(Mandatory)] [string[]] $Names)
    # An empty pathspec would search the whole repository and implicate every type there is.
    $directories = @($script:BuildTimeDirectories | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($directories.Count -eq 0) { throw 'no build-time directories are configured' }
    $found = [System.Collections.Generic.List[string]]::new()
    foreach ($name in $Names) {
        $arguments = @('-c', 'core.quotepath=false', 'grep', '-l', '-i', '-w', '-F', '-e', $name, 'HEAD', '--')
        $arguments += $directories
        $paths = @(& git @arguments 2>$null)
        if ($LASTEXITCODE -gt 1) { throw "git grep for build-time references failed with exit code $LASTEXITCODE" }
        if (-not (Get-Variable GeneratorSourceIndex -Scope Script -ErrorAction SilentlyContinue)) { $script:GeneratorSourceIndex = @{} }
        foreach ($path in $paths) {
            $file = ([string] $path) -replace '^HEAD:', ''
            if (-not $script:GeneratorSourceIndex.ContainsKey($file)) {
                $source = Get-GitFileText -Path $file -Revisions @('HEAD')
                if ($null -eq $source) { throw "Cannot read generator $file" }
                # Emitted C# resides in strings, which MUST remain searchable.
                # Comments mentioning a regression test are not dependencies.
                $script:GeneratorSourceIndex[$file] = ConvertTo-CodeOnlyCSharp -Text $source -PreserveStrings
            }
            if ($script:GeneratorSourceIndex[$file] -match ('(?<!\w)' + [regex]::Escape($name) + '(?!\w)')) {
                $found.Add($name)
                break
            }
        }
    }
    return @($found)
}

function Resolve-PullRequestBase {
    <#
        The pull_request event's base.sha is the base branch as it was when the pull request was
        opened or last retargeted - not what GitHub merged it onto for this run. For a pull request
        behind master it attributes every commit master has gained since to the pull request.

        The checked-out merge ref is authoritative: its first parent IS the base this run tests
        against, and its second parent is the pull request head. Both are verified, so a checkout
        that is not the expected two-parent merge fails closed rather than guessing.
    #>
    param([Parameter(Mandatory)] [string] $PullRequestHeadSha)

    if ($PullRequestHeadSha -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
        throw "pull request head '$PullRequestHeadSha' is not a complete Git object id"
    }
    $parents = @((& git rev-list --parents -n 1 HEAD) -split '\s+' | Where-Object { $_ })
    if ($LASTEXITCODE -ne 0) { throw 'cannot read the checked-out commit' }
    if ($parents.Count -ne 3) {
        throw "the checkout is not a two-parent pull-request merge commit ($($parents.Count - 1) parent(s))"
    }
    if (-not $parents[2].Equals($PullRequestHeadSha, [StringComparison]::OrdinalIgnoreCase)) {
        throw "the merge commit's second parent $($parents[2]) is not the pull request head $PullRequestHeadSha"
    }
    return $parents[1]
}

function Resolve-MapLineageBase {
    <#
        The base to select from when the map was measured on another branch than the pull request's
        base: a stacked pull request (base branch B, itself ahead of master) using master's map.

        Selecting on the diff from B alone would trust the map for everything B added to master,
        and the map has never seen it: a test B adds can execute the lines this pull request
        changes without any shard being recorded for them, and no map run validated B's commits
        when they landed, because they have not landed. So the change is everything the checkout
        adds to the map's branch - the diff from their merge base - which is exactly the shape of
        an ordinary pull request against that branch: the merge base is a commit of the map's own
        lineage, and both diffs Get-ScopedChangedRanges takes (map -> base, base -> HEAD) mean
        what they mean there.

        That holds only when the checkout contains the map's commit. A stack cut from master
        before the map's commit is missing the commits between its merge base and the map; the
        map describes code the tested tree does not have, and a diff from the merge base does not
        show that difference. It fails closed, as does an ambiguous (criss-cross) merge base.
    #>
    param(
        [Parameter(Mandatory)] [string] $LineageRef,
        [Parameter(Mandatory)] [string] $MapSha
    )

    $lineage = [string] (& git rev-parse --verify --quiet "$LineageRef^{commit}" 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($lineage)) {
        throw "the map's branch '$LineageRef' is not present in this checkout"
    }
    & git merge-base --is-ancestor $MapSha HEAD 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "the checkout does not contain the map's commit $MapSha, so a map from '$LineageRef' does not describe it; update the base branch from '$LineageRef'"
    }
    $bases = @(& git merge-base --all HEAD $lineage 2>$null | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($LASTEXITCODE -ne 0 -or $bases.Count -eq 0) {
        throw "the checkout shares no history with the map's branch '$LineageRef'"
    }
    if ($bases.Count -gt 1) {
        throw "the checkout has $($bases.Count) merge bases with the map's branch '$LineageRef', so what it adds to that branch is ambiguous"
    }
    return ([string] $bases[0]).Trim()
}

# ---------------------------------------------------------------- self-test

if ($SelfTest) {
    $failures = [System.Collections.Generic.List[string]]::new()
    function Assert-True {
        param([bool] $Condition, [string] $What)
        if (-not $Condition) { [void] $failures.Add($What) }
    }
    function Assert-Throws {
        param([scriptblock] $Action, [string] $What)
        $threw = $false
        try { & $Action } catch { $threw = $true }
        Assert-True $threw $What
    }

    $map = @{
        schemaVersion = 1
        sha           = '0123456789abcdef0123456789abcdef01234567'
        knownShards   = @('Alpha', 'Beta')
        alwaysRun     = @('HeavyNoCoverage')
        files         = @{
            'src/Covered.cs' = @(
                @{ s = 0; r = @(10, 20) },
                @{ s = 1; r = @(100, 110) }
            )
            'src/SingleRange.cs' = @( @{ s = 0; r = @(5, 6) } )
        }
    } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
    $expected = @('Alpha', 'Beta', 'HeavyNoCoverage')
    try { Assert-ShardMap -Map $map -Expected $expected } catch { [void] $failures.Add("valid map rejected: $_") }

    # CLASSIFICATION PINNED TO REAL PULL REQUESTS, because the failures this feature keeps shipping
    # are classification failures and none of them were visible from a synthetic map. Each case below
    # is the exact path set of a merged or open pull request, with the shard count it actually ran.
    #
    #   #1889  CHANGELOG.md + .release-please-manifest.json          ran 1 of 164   (correct)
    #   #2223  website/package-lock.json                             ran 164 of 164 (the defect)
    #   #2204  src/ActivationFunctions/GumbelSoftmaxActivation.cs    ran 49 of 164
    #   #2098  98 .cs files across generators and models             ran 161 of 164
    $classification = @(
        @{ Pr = 1889; Path = 'CHANGELOG.md'; Expect = [ChangedPathImpact]::NonRuntime },
        @{ Pr = 1889; Path = '.release-please-manifest.json'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 2223; Path = 'website/package-lock.json'; Expect = [ChangedPathImpact]::NonRuntime },
        @{ Pr = 2223; Path = 'website/src/pages/index.tsx'; Expect = [ChangedPathImpact]::NonRuntime },
        @{ Pr = 2204; Path = 'src/ActivationFunctions/GumbelSoftmaxActivation.cs'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 2098; Path = 'src/AiDotNet.Generators/PaperOptimizerAnalyzer.cs'; Expect = [ChangedPathImpact]::FullValidation },
        @{ Pr = 0; Path = 'Directory.Build.props'; Expect = [ChangedPathImpact]::FullValidation },
        @{ Pr = 0; Path = 'website/Directory.Build.props'; Expect = [ChangedPathImpact]::FullValidation },
        @{ Pr = 0; Path = '.github/workflows/sonarcloud.yml'; Expect = [ChangedPathImpact]::SelectionControl },
        # Paths that sent otherwise mapped pull requests to the full matrix in November 2026 (35 of
        # 42 runs ran every shard). #2578 changed DeserializationHelper.cs and one benchmark and ran
        # all 167 shards for the benchmark.
        @{ Pr = 2578; Path = 'AiDotNetBenchmarkTests/BenchmarkTests/GaussianProcessBenchmarks.cs'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 2578; Path = 'src/Helpers/DeserializationHelper.cs'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'benchmarks/AiDotNet.Benchmarks/Program.cs'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'benchmarks/AiDotNet.PyTorchParity/AiDotNet.PyTorchParity.csproj'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'src/PublicAPI.Unshipped.txt'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'src/PublicAPI.Shipped.txt'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'src/PublicAPI/net471/PublicAPI.Unshipped.txt'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'src/PublicAPI/net10.0/PublicAPI.Shipped.txt'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'src/ElementWiseBaseline.txt'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'src/CompatibilitySuppressions.xml'; Expect = [ChangedPathImpact]::BuildOnly },
        @{ Pr = 0; Path = 'tools/reference-data/vallex_phone_reference.py'; Expect = [ChangedPathImpact]::NonRuntime },
        @{ Pr = 0; Path = 'tools/reference-data/requirements.txt'; Expect = [ChangedPathImpact]::NonRuntime },
        @{ Pr = 0; Path = 'vercel.json'; Expect = [ChangedPathImpact]::NonRuntime },
        # The lists are exact. A lookalike, another text or XML file beside them, the same name in
        # another project or in a different case, and a build file inside a benchmark tree keep
        # the behaviour they had.
        @{ Pr = 0; Path = 'src/PublicAPI.Unshipped.txt.bak'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'src/Resources/Vocabulary.txt'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'src/AiDotNet.Serving/PublicAPI.Unshipped.txt'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'src/AiDotNet.Serving/CompatibilitySuppressions.xml'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'src/PublicAPI/net471/Other.txt'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'SRC/ElementWiseBaseline.txt'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'src/vercel.json'; Expect = [ChangedPathImpact]::MapCandidate },
        @{ Pr = 0; Path = 'benchmarks/Directory.Build.props'; Expect = [ChangedPathImpact]::FullValidation },
        @{ Pr = 0; Path = 'benchmarks/AiDotNet.Benchmarks/global.json'; Expect = [ChangedPathImpact]::FullValidation },
        @{ Pr = 0; Path = 'AiDotNetBenchmarkTests/Directory.Packages.props'; Expect = [ChangedPathImpact]::FullValidation },
        @{ Pr = 0; Path = 'tests/AiDotNet.Tests/Benchmarks/SomeBenchmarkTests.cs'; Expect = [ChangedPathImpact]::MapCandidate }
    )
    foreach ($case in $classification) {
        $actual = Get-ChangedPathImpact -Path $case.Path
        Assert-True ($actual -eq $case.Expect) (
            "classification: $($case.Path) expected $($case.Expect) but got $actual" +
            $(if ($case.Pr) { " (PR #$($case.Pr))" } else { '' }))
    }
    # THE TRIP-WIRE, ARMED. Pointed at the repository root rather than $PSScriptRoot: this script
    # lives in tools/TestImpact, so the original looked for tools/TestImpact/website, found nothing,
    # and passed without checking anything -- a guard that cannot fail is not a guard.
    $repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    try { Assert-NonRuntimeDirectories -Root $repositoryRoot } catch {
        [void] $failures.Add("non-runtime directory trip-wire: $_")
    }
    # And proved to fire, against a tree built for the purpose, so the check above is not merely
    # passing because the directory happens to be absent from this checkout.
    $tripwire = Join-Path ([IO.Path]::GetTempPath()) ("nonruntime-" + [Guid]::NewGuid().ToString('n'))
    try {
        [void] (New-Item -ItemType Directory -Path (Join-Path $tripwire 'website/src') -Force)
        Set-Content -LiteralPath (Join-Path $tripwire 'website/src/Leaked.cs') -Value 'class Leaked {}'
        Assert-Throws { Assert-NonRuntimeDirectories -Root $tripwire } `
            'a non-runtime directory holding compilable code must be rejected'
        Remove-Item -LiteralPath (Join-Path $tripwire 'website/src/Leaked.cs') -Force
        try { Assert-NonRuntimeDirectories -Root $tripwire } catch {
            [void] $failures.Add("trip-wire rejected a clean non-runtime directory: $_")
        }
    } finally {
        Remove-Item -LiteralPath $tripwire -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Every workflow file is classified. Against this checkout first: a workflow added without being read and
    # listed fails here, so it is neither silently independent nor escalating for ever.
    try { Assert-WorkflowsClassified -Root $repositoryRoot } catch {
        [void] $failures.Add("workflow classification: $_")
    }
    foreach ($workflow in $script:NonRuntimeWorkflowPaths) {
        Assert-True ((Get-ChangedPathImpact -Path $workflow) -eq [ChangedPathImpact]::NonRuntime) `
            "an independent workflow is not classified non-runtime: $workflow"
        $r = Select-ImpactedShards -Map $map -Changed @{ $workflow = @(1, 2) }
        Assert-True (-not $r.Escalate -and -not $r.RequiresValidation -and $r.Shards.Count -eq 0) `
            "an independent workflow selected shards or escalated: $workflow"
    }
    foreach ($workflow in @($script:PipelineWorkflowPaths) + @($script:SelectionControlPaths)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ $workflow = @(1, 2) }
        Assert-True ($r.Escalate -and $r.RequiresValidation) "a pipeline workflow did not escalate: $workflow"
    }
    Assert-True ((Get-ChangedPathImpact -Path '.github/workflows/resume-deferred-ci.yml') -eq [ChangedPathImpact]::FullValidation) `
        'the deferred-validation reconciler is not full validation'
    foreach ($lookalike in '.github/workflows/release.yml', '.github/workflows/publish.yml', '.github/workflows/website-e2e.yml.bak',
        '.github/workflows/nested/website-health.yml') {
        Assert-True ((Get-ChangedPathImpact -Path $lookalike) -eq [ChangedPathImpact]::FullValidation) `
            "a workflow that is not on the list was treated as independent: $lookalike"
    }
    # And proved to fire: a tree holding exactly the classified files passes, one more file fails, a re-enabled
    # release workflow fails, and an independent workflow the pipeline calls fails.
    $workflowRoot = Join-Path ([IO.Path]::GetTempPath()) ("workflows-" + [Guid]::NewGuid().ToString('n'))
    try {
        [void] (New-Item -ItemType Directory -Path (Join-Path $workflowRoot '.github/workflows') -Force)
        foreach ($workflow in @($script:NonRuntimeWorkflowPaths) + @($script:PipelineWorkflowPaths) + @($script:SelectionControlPaths)) {
            Set-Content -LiteralPath (Join-Path $workflowRoot $workflow) -Value 'name: fixture'
        }
        try { Assert-WorkflowsClassified -Root $workflowRoot } catch {
            [void] $failures.Add("workflow classification rejected a fully classified tree: $_")
        }
        foreach ($added in 'brand-new.yml', 'release.yml', 'notes.txt') {
            $addedPath = Join-Path $workflowRoot ".github/workflows/$added"
            Set-Content -LiteralPath $addedPath -Value 'name: unread'
            $message = ''
            try { Assert-WorkflowsClassified -Root $workflowRoot } catch { $message = [string] $_ }
            Assert-True ($message -like "*unclassified workflow file(s): .github/workflows/$added*") `
                "a workflow file added without being classified was accepted: $added"
            Remove-Item -LiteralPath $addedPath -Force
        }
        # A listed workflow whose file is not there (codeql.yml until #2585 lands) is not an error, and a change
        # to it selects nothing from the day it appears.
        Remove-Item -LiteralPath (Join-Path $workflowRoot '.github/workflows/codeql.yml') -Force
        try { Assert-WorkflowsClassified -Root $workflowRoot } catch {
            [void] $failures.Add("workflow classification rejected a listed workflow with no file yet: $_")
        }
        Assert-True ((Get-ChangedPathImpact -Path '.github/workflows/codeql.yml') -eq [ChangedPathImpact]::NonRuntime -and
            (Get-ChangedPathImpact -Path '.github/workflows/codeql-analysis.yml') -eq [ChangedPathImpact]::FullValidation) `
            'the planned code-scanning workflow is not independent, or a workflow named like it is'
        Set-Content -LiteralPath (Join-Path $workflowRoot '.github/workflows/sonarcloud.yml') -Value @(
            'jobs:', '  e2e:', '    uses: ./.github/workflows/website-e2e.yml')
        $message = ''
        try { Assert-WorkflowsClassified -Root $workflowRoot } catch { $message = [string] $_ }
        Assert-True ($message -like '*website-e2e.yml is listed as independent, but the pipeline workflow*') `
            'an independent workflow called by the pipeline was accepted'
    } finally {
        Remove-Item -LiteralPath $workflowRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Dependency bumps route by package. A synthetic tree: one product project, two test projects with shards, a
    # worker test project with none, a benchmark, a standalone tool, a tool the tests reference, and a shared
    # props file that hands one package to everything.
    $packageBase = [ordered]@{
        'AiDotNet.Tensors' = '0.90.2'; 'Newtonsoft.Json' = '13.0.3'; 'Moq' = '4.20.0'; 'TorchSharp' = '0.105.0'
        'xunit' = '2.9.0'; 'coverlet.collector' = '6.0.0'; 'Microsoft.NET.Test.Sdk' = '17.0.0'; 'Orphan.Package' = '1.0.0'
        'Shared.Package' = '1.0.0'; 'Tool.Package' = '1.0.0'; 'Lib.Package' = '1.0.0'; 'Serving.TestOnly' = '1.0.0'
        'Worker.Package' = '1.0.0'; 'Microsoft.CodeAnalysis.CSharp' = '4.14.0'
    }
    function New-PackageProps {
        param([System.Collections.IDictionary] $Versions, [string] $Property = '', [string] $Comment = 'versions', [string] $Extra = '')
        $entries = @($Versions.GetEnumerator() | ForEach-Object { "    <PackageVersion Include=`"$($_.Key)`" Version=`"$($_.Value)`" />" })
        return (@('<Project>', '  <PropertyGroup>', '    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>', $Property,
            '  </PropertyGroup>', '  <ItemGroup>', "    <!-- $Comment -->") + $entries + @($Extra, '  </ItemGroup>', '</Project>')) -join "`n"
    }
    function New-Bump {
        param([hashtable] $Changes)
        $versions = [ordered]@{}
        foreach ($entry in $packageBase.GetEnumerator()) { $versions[$entry.Key] = $entry.Value }
        foreach ($name in $Changes.Keys) { $versions[$name] = $Changes[$name] }
        return $versions
    }
    function New-BuildLine([string] $File, [string] $Text) { [pscustomobject]@{ File = $File; Text = $Text } }
    $testsProject = 'tests/AiDotNet.Tests/AiDotNetTests.csproj'
    $servingProject = 'tests/AiDotNet.Serving.Tests/AiDotNet.Serving.Tests.csproj'
    $packageProjects = @('src/AiDotNet.csproj', $testsProject, $servingProject, 'tests/AiDotNet.Worker/Worker.csproj',
        'AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj', 'tools/Standalone/Standalone.csproj', 'tools/Referenced/Referenced.csproj')
    $packageLines = @(
        (New-BuildLine 'Directory.Packages.props' '    <PackageVersion Include="Moq" Version="4.20.0" />'),
        (New-BuildLine 'src/AiDotNet.csproj' '    <PackageReference Include="AiDotNet.Tensors" />'),
        (New-BuildLine 'src/AiDotNet.csproj' '    <PackageReference Include="Newtonsoft.Json" />'),
        (New-BuildLine 'src/AiDotNet.csproj' '    <PackageReference Include="Microsoft.CodeAnalysis.CSharp" VersionOverride="4.14.0" />'),
        (New-BuildLine 'src/AiDotNet.csproj' '    <EvolutionProjectPath Condition="''$(EvolutionProjectPath)'' == ''''">$(MSBuildThisFileDirectory)..\..\AiDotNet.Evolution\src\AiDotNet.Evolution.csproj</EvolutionProjectPath>'),
        (New-BuildLine 'src/AiDotNet.csproj' '    <ProjectReference Include="$(EvolutionProjectPath)" />'),
        (New-BuildLine $testsProject '    <PackageReference Include="Moq" />'),
        (New-BuildLine $testsProject '    <PackageReference Include="xunit" />'),
        (New-BuildLine $testsProject '    <PackageReference Include="coverlet.collector" PrivateAssets="all" />'),
        (New-BuildLine $testsProject '    <PackageReference Include="Microsoft.NET.Test.Sdk" />'),
        (New-BuildLine $testsProject '    <ProjectReference Include="..\..\src\AiDotNet.csproj" />'),
        (New-BuildLine $testsProject '    <ProjectReference Include="..\..\tools\Referenced\Referenced.csproj" />'),
        (New-BuildLine $servingProject '    <PackageReference Include="Serving.TestOnly" />'),
        (New-BuildLine $servingProject '    <PackageReference Include="xunit" />'),
        (New-BuildLine 'tests/AiDotNet.Worker/Worker.csproj' '    <PackageReference Include="Worker.Package" />'),
        (New-BuildLine 'AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj' '    <PackageReference Include="TorchSharp" />'),
        (New-BuildLine 'AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj' '    <ProjectReference Include="..\src\AiDotNet.csproj" />'),
        (New-BuildLine 'tools/Standalone/Standalone.csproj' '    <PackageReference Include="Tool.Package" />'),
        (New-BuildLine 'tools/Referenced/Referenced.csproj' '    <PackageReference Include="Lib.Package" />'),
        (New-BuildLine 'Directory.Build.props' '    <PackageReference Include="Shared.Package" PrivateAssets="all" />')
    )
    $packageManifest = @(
        [pscustomobject]@{ name = 'Alpha'; project = $testsProject; filter = 'A' },
        [pscustomobject]@{ name = 'Beta'; project = $testsProject; filter = 'B' },
        [pscustomobject]@{ name = 'Serving'; project = $servingProject; filter = 'S' }
    )
    $packageClassifier = {
        param($Path)
        if ($Path.StartsWith('src/')) { return 'Product' }
        if ($Path.StartsWith('tests/')) { return 'Test' }
        if ($Path.StartsWith('AiDotNetBenchmarkTests/') -or $Path.StartsWith('tools/Standalone/')) { return 'Inert' }
        return 'Unknown'
    }
    function Get-TestPackageVerdict {
        param([string] $Head, [string] $Base = (New-PackageProps $packageBase), [object[]] $Lines = $packageLines,
            [string[]] $Projects = $packageProjects, $Manifest = $packageManifest)
        return Get-PackagePropsVerdict -BaseText $Base -HeadText $Head -Lines $Lines -Projects $Projects -Manifest $Manifest `
            -ClassifyProject $packageClassifier
    }
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'TorchSharp' = '0.106.0' }))
    Assert-True ($v.Impact -ceq 'BuildOnly' -and $v.Shards.Count -eq 0 -and $v.Packages[0].name -ceq 'TorchSharp' -and
        $v.Packages[0].from -ceq '0.105.0' -and $v.Packages[0].to -ceq '0.106.0') `
        "a benchmark-only package bump was not build-only: $($v.Why)"
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Tool.Package' = '2.0.0' }))
    Assert-True ($v.Impact -ceq 'BuildOnly') "a standalone-tool package bump was not build-only: $($v.Why)"
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'TorchSharp' = '0.106.0' }) -Comment 'bumped TorchSharp')
    Assert-True ($v.Impact -ceq 'BuildOnly') "a comment edited beside a version bump changed the verdict: $($v.Why)"
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Moq' = '4.21.0' }))
    Assert-True ($v.Impact -ceq 'TestRoute' -and ($v.Shards -join ',') -ceq 'Alpha,Beta') `
        "a test-only package bump did not select exactly its test project's shards: $($v.Impact) $($v.Shards -join ',') $($v.Why)"
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Serving.TestOnly' = '1.1.0' }))
    Assert-True ($v.Impact -ceq 'TestRoute' -and ($v.Shards -join ',') -ceq 'Serving') `
        "a package of one test project selected another project's shards: $($v.Shards -join ',')"
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Moq' = '4.21.0'; 'TorchSharp' = '0.106.0'; 'Serving.TestOnly' = '1.1.0' }))
    Assert-True ($v.Impact -ceq 'TestRoute' -and ($v.Shards -join ',') -ceq 'Alpha,Beta,Serving') `
        'a mixed test-only and benchmark-only bump did not select the union of the test shards'
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Moq' = '4.21.0' })) -Manifest $null
    Assert-True ($v.Impact -ceq 'FullValidation') 'a test-only package bump was routed without a shard manifest'
    $v = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'TorchSharp' = '0.106.0' })) -Manifest $null
    Assert-True ($v.Impact -ceq 'BuildOnly') 'a benchmark-only bump needed a shard manifest'
    foreach ($case in @(
        @{ Why = 'a product package bump'; Changes = @{ 'Newtonsoft.Json' = '13.0.4' } },
        @{ Why = 'a product package bumped beside a benchmark package'; Changes = @{ 'Newtonsoft.Json' = '13.0.4'; 'TorchSharp' = '0.106.0' } },
        @{ Why = 'AiDotNet.Tensors'; Changes = @{ 'AiDotNet.Tensors' = '0.91.0' } },
        @{ Why = 'the xunit framework'; Changes = @{ 'xunit' = '2.9.1' } },
        @{ Why = 'the coverage collector'; Changes = @{ 'coverlet.collector' = '6.0.1' } },
        @{ Why = 'the test SDK'; Changes = @{ 'Microsoft.NET.Test.Sdk' = '17.1.0' } },
        @{ Why = 'the compiler package'; Changes = @{ 'Microsoft.CodeAnalysis.CSharp' = '4.15.0' } },
        @{ Why = 'a package nothing references'; Changes = @{ 'Orphan.Package' = '1.0.1' } },
        @{ Why = 'a package a shared props file references'; Changes = @{ 'Shared.Package' = '1.0.1' } },
        @{ Why = 'a package of a test project no shard runs'; Changes = @{ 'Worker.Package' = '1.0.1' } },
        @{ Why = 'a package of a tool the tests reference'; Changes = @{ 'Lib.Package' = '1.0.1' } },
        @{ Why = 'a new entry'; Changes = @{ 'Brand.New' = '1.0.0' } }
    )) {
        $v = Get-TestPackageVerdict (New-PackageProps (New-Bump $case.Changes))
        Assert-True ($v.Impact -ceq 'FullValidation' -and $v.Shards.Count -eq 0) "$($case.Why) did not keep the full matrix: $($v.Impact) - $($v.Why)"
    }
    # Edits that are not a version bump of a plain entry.
    $removed = New-Bump @{}
    $removed.Remove('TorchSharp')
    $bumped = New-Bump @{ 'TorchSharp' = '0.106.0' }
    foreach ($case in @(
        @{ Why = 'a removed entry'; Head = (New-PackageProps $removed) },
        @{ Why = 'a comment-only edit'; Head = (New-PackageProps $packageBase -Comment 'reworded') },
        @{ Why = 'a new property'; Head = (New-PackageProps $packageBase -Property '    <CentralPackageVersionOverrideEnabled>false</CentralPackageVersionOverrideEnabled>') },
        @{ Why = 'a property changed beside a benchmark bump'; Head = (New-PackageProps $bumped -Property '    <NoWarn>NU1507</NoWarn>') },
        @{ Why = 'a conditional entry'; Head = (New-PackageProps $packageBase).Replace('Include="TorchSharp" Version="0.105.0"', 'Include="TorchSharp" Version="0.106.0" Condition="''$(OS)'' == ''Windows_NT''"') },
        @{ Why = 'a version taken from a property'; Head = (New-PackageProps (New-Bump @{ 'TorchSharp' = '$(TorchVersion)' })) },
        @{ Why = 'an entry declared twice'; Head = (New-PackageProps $bumped -Extra '    <PackageVersion Include="TorchSharp" Version="0.107.0" />') },
        @{ Why = 'a conditional item group'; Head = (New-PackageProps $bumped).Replace('<ItemGroup>', '<ItemGroup Condition="''$(X)'' == ''1''">') },
        @{ Why = 'a file that is not XML'; Head = '<Project><ItemGroup>' },
        @{ Why = 'a deleted file'; Head = '' }
    )) {
        $v = Get-TestPackageVerdict $case.Head
        Assert-True ($v.Impact -ceq 'FullValidation') "$($case.Why) did not keep the full matrix: $($v.Impact) - $($v.Why)"
    }
    $v = Get-TestPackageVerdict (New-PackageProps $bumped) -Base ''
    Assert-True ($v.Impact -ceq 'FullValidation') 'a props file absent at the base did not keep the full matrix'
    # What the tree can add: transitive pinning, a product project depending on the benchmark, a reference that
    # cannot be resolved, a reference declared over several lines, and a shared file referencing the consumer.
    foreach ($case in @(
        @{ Why = 'central transitive pinning'; Line = (New-BuildLine 'Directory.Build.props' '    <CentralPackageTransitivePinningEnabled>true</CentralPackageTransitivePinningEnabled>') },
        @{ Why = 'a product project that depends on the benchmark project'; Line = (New-BuildLine 'src/AiDotNet.csproj' '    <ProjectReference Include="..\AiDotNetBenchmarkTests\AiDotNetBenchmarkTests.csproj" />') },
        @{ Why = 'a product project reference through an undefined property'; Line = (New-BuildLine 'src/AiDotNet.csproj' '    <ProjectReference Include="$(SomewhereElse)" />') },
        @{ Why = 'a product project reference by file name through a property'; Line = (New-BuildLine 'src/AiDotNet.csproj' '    <ProjectReference Include="$(Root)/AiDotNetBenchmarkTests.csproj" />') },
        @{ Why = 'a package reference declared over several lines'; Line = (New-BuildLine 'tools/Standalone/Standalone.csproj' '    <PackageReference') },
        @{ Why = 'a shared file that references the benchmark project'; Line = (New-BuildLine 'Directory.Build.targets' '    <ProjectReference Include="$(MSBuildThisFileDirectory)AiDotNetBenchmarkTests/AiDotNetBenchmarkTests.csproj" />') },
        @{ Why = 'a shared file with an unresolvable project reference'; Line = (New-BuildLine 'Directory.Build.targets' '    <ProjectReference Include="$(Anything)" />') },
        @{ Why = 'a reference to the package in another element'; Line = (New-BuildLine 'tools/Standalone/Standalone.csproj' '    <PackageDownload Include="TorchSharp" Version="[0.105.0]" />') }
    )) {
        $v = Get-TestPackageVerdict (New-PackageProps $bumped) -Lines (@($packageLines) + @($case.Line))
        Assert-True ($v.Impact -ceq 'FullValidation') "$($case.Why) did not keep the full matrix for a benchmark bump: $($v.Impact) - $($v.Why)"
    }
    # A property-valued reference whose definition names an untracked project (a sibling repository) is no
    # dependency on anything here, and an unresolvable reference from an inert project stays inert.
    $v = Get-TestPackageVerdict (New-PackageProps $bumped) -Lines (@($packageLines) +
        @(New-BuildLine 'tools/Standalone/Standalone.csproj' '    <ProjectReference Include="$(SomewhereElse)" />'))
    Assert-True ($v.Impact -ceq 'BuildOnly') "an unresolvable reference from a standalone tool changed a benchmark bump: $($v.Why)"

    # The verdict reaches classification and selection, for the root file only, and only while it is set.
    try {
        $script:PackagePropsVerdict = Get-TestPackageVerdict (New-PackageProps $bumped)
        Assert-True ((Get-ChangedPathImpact -Path 'Directory.Packages.props') -eq [ChangedPathImpact]::BuildOnly) `
            'a build-only package verdict did not classify the props file as build-only'
        Assert-True ((Get-ChangedPathImpact -Path 'benchmarks/Directory.Packages.props') -eq [ChangedPathImpact]::FullValidation -and
            (Get-ChangedPathImpact -Path 'directory.packages.props') -eq [ChangedPathImpact]::FullValidation) `
            'a package verdict for the root props file reached another props file'
        $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props' = @(1, 2) }
        Assert-True (-not $r.Escalate -and $r.Shards.Count -eq 0) 'a build-only package bump selected shards or escalated'
        $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props' = @(1, 2); 'src/Covered.cs' = @(12, 14) }
        Assert-True (-not $r.Escalate -and ($r.Shards -join ',') -ceq 'Alpha,HeavyNoCoverage') `
            'a build-only package bump widened or suppressed a mapped change beside it'

        $script:PackagePropsVerdict = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Moq' = '4.21.0' }))
        Assert-True ((Get-ChangedPathImpact -Path 'Directory.Packages.props') -eq [ChangedPathImpact]::MapCandidate) `
            'a test-route package verdict did not make the props file routable'
        $route = [pscustomobject]@{ Routable = $true; Shards = @($script:PackagePropsVerdict.Shards); Why = $script:PackagePropsVerdict.Why; Category = 'PackageVersion' }
        $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props' = @(1, 2) } -TestRoutes @{ 'Directory.Packages.props' = $route }
        Assert-True (-not $r.Escalate -and $r.RequiresValidation -and ($r.Shards -join ',') -ceq 'Alpha,Beta,HeavyNoCoverage') `
            "a test-only package bump did not run its test project's shards: $($r.Shards -join ',')"
        Assert-True (@($r.Files | Where-Object { $_.path -ceq 'Directory.Packages.props' -and $_.category -ceq 'PackageVersion' -and $_.outcome -ceq 'routed' }).Count -eq 1) `
            'the routed package bump was not recorded as a package-version decision'
        # Routable but with no route supplied (a caller that forgot it): unmapped, so it escalates.
        $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props' = @(1, 2) }
        Assert-True $r.Escalate 'a test-route package verdict without its route did not escalate'

        $script:PackagePropsVerdict = Get-TestPackageVerdict (New-PackageProps (New-Bump @{ 'Newtonsoft.Json' = '13.0.4' }))
        Assert-True ((Get-ChangedPathImpact -Path 'Directory.Packages.props') -eq [ChangedPathImpact]::FullValidation) `
            'a full-validation package verdict did not keep the props file full validation'
    } finally {
        $script:PackagePropsVerdict = $null
    }
    Assert-True ((Get-ChangedPathImpact -Path 'Directory.Packages.props') -eq [ChangedPathImpact]::FullValidation) `
        'without a package verdict the props file is not full validation'

    # A build-only directory is trusted only while nothing outside it consumes it. Proved against
    # throwaway repositories: unreferenced, referenced by a test project, named by the shard
    # manifest, read by a test at run time (a Path.Combine split across lines or after a nested call
    # included), linked by a project in the other build-only directory, and merely mentioned in a
    # comment or a non-path string literal.
    $savedReferenceCache = $script:ToolReferenceCache
    $consumerRoot = Join-Path ([IO.Path]::GetTempPath()) ("buildonly-" + [Guid]::NewGuid().ToString('n'))
    try {
        foreach ($case in @(
            @{ Name = 'unreferenced'; File = 'tests/Proj/Notes.cs'; Text = '// measured by benchmarks/AiDotNet.Benchmarks (see the report)'; Consumed = $false },
            @{ Name = 'project reference'; File = 'tests/Proj/Proj.csproj'; Text = '<ProjectReference Include="..\..\benchmarks\AiDotNet.Benchmarks\AiDotNet.Benchmarks.csproj" />'; Consumed = $true },
            @{ Name = 'shard manifest'; File = '.github/test-shards.yml'; Text = '    project: benchmarks/AiDotNet.Benchmarks/AiDotNet.Benchmarks.csproj'; Consumed = $true },
            @{ Name = 'run-time read'; File = 'tests/Proj/ReadsData.cs'; Text = 'var text = File.ReadAllText("../../benchmarks/data/expected.json");'; Consumed = $true },
            @{ Name = 'run-time join'; File = 'tests/Proj/JoinsData.cs'; Text = 'var path = Path.Combine(root, "benchmarks", "data");'; Consumed = $true },
            @{ Name = 'run-time join across lines'; File = 'tests/Proj/JoinsSplit.cs'; Text = "var path = Path.Combine(root,`n    `"benchmarks`", `"data`", `"expected.json`");"; Consumed = $true },
            @{ Name = 'run-time join after a call'; File = 'tests/Proj/JoinsCall.cs'; Text = "var path = Path.Join(`n    GetRoot(), 'benchmarks');"; Consumed = $true },
            @{ Name = 'literal outside a join'; File = 'tests/Proj/Trait.cs'; Text = "[Trait(`"Category`",`n    `"Benchmarks`")] public void Measures() { var p = Path.Combine(a, b); }"; Consumed = $false },
            @{ Name = 'linked by another build-only directory'; File = 'AiDotNetBenchmarkTests/Proj/Proj.csproj'; Text = '<Compile Include="..\..\benchmarks\Shared\Harness.cs" />'; Consumed = $true }
        )) {
            $caseRoot = Join-Path $consumerRoot ([Guid]::NewGuid().ToString('n'))
            [void] (New-Item -ItemType Directory -Path (Join-Path $caseRoot (Split-Path -Parent $case.File)) -Force)
            [void] (New-Item -ItemType Directory -Path (Join-Path $caseRoot 'benchmarks/AiDotNet.Benchmarks') -Force)
            Set-Content -LiteralPath (Join-Path $caseRoot $case.File) -Value $case.Text
            # A benchmark project referencing itself or the product is not a consumer.
            Set-Content -LiteralPath (Join-Path $caseRoot 'benchmarks/AiDotNet.Benchmarks/AiDotNet.Benchmarks.csproj') `
                -Value '<Compile Include="..\..\benchmarks\Shared\Harness.cs" />'
            Push-Location $caseRoot
            try {
                $null = & git init -q 2>&1
                $null = & git add -A 2>&1
                $null = & git -c user.email=fixture@example.invalid -c user.name=fixture -c commit.gpgSign=false -c core.hooksPath=.no-hooks commit -q -m fixture 2>&1
                if ($LASTEXITCODE -ne 0) { [void] $failures.Add("build-only consumer fixture '$($case.Name)' could not be committed"); continue }
                $script:ToolReferenceCache = @{}
                $impact = Get-ChangedPathImpact -Path 'benchmarks/AiDotNet.Benchmarks/Program.cs'
                $expectedImpact = if ($case.Consumed) { [ChangedPathImpact]::MapCandidate } else { [ChangedPathImpact]::BuildOnly }
                Assert-True ($impact -eq $expectedImpact) `
                    "build-only directory, $($case.Name): expected $expectedImpact but got $impact"
            }
            finally { Pop-Location }
        }
        # Outside any repository the search itself fails, which must read as consumed.
        [void] (New-Item -ItemType Directory -Path (Join-Path $consumerRoot 'no-repository') -Force)
        Push-Location (Join-Path $consumerRoot 'no-repository')
        try {
            $savedCeiling = $env:GIT_CEILING_DIRECTORIES
            $env:GIT_CEILING_DIRECTORIES = $consumerRoot
            $script:ToolReferenceCache = @{}
            Assert-True ((Get-ChangedPathImpact -Path 'benchmarks/AiDotNet.Benchmarks/Program.cs') -eq [ChangedPathImpact]::MapCandidate) `
                'a failed consumer search must not classify a benchmark as build-only'
        }
        finally {
            $env:GIT_CEILING_DIRECTORIES = $savedCeiling
            Pop-Location
        }
    }
    finally {
        $script:ToolReferenceCache = $savedReferenceCache
        $global:LASTEXITCODE = 0
        Remove-Item -LiteralPath $consumerRoot -Recurse -Force -ErrorAction SilentlyContinue
    }

    # What the new classes do to a selection. #2578's exact path set: the benchmark no longer
    # escalates, and the mapped source still selects its own shards.
    $r = Select-ImpactedShards -Map $map -Changed @{
        'AiDotNetBenchmarkTests/BenchmarkTests/GaussianProcessBenchmarks.cs' = @(1, 2)
        'src/Covered.cs' = @(12, 14)
    }
    Assert-True (-not $r.Escalate -and $r.RequiresValidation -and ($r.Shards -join ',') -ceq 'Alpha,HeavyNoCoverage') `
        "a benchmark beside a mapped source widened or suppressed the selection: $($r.Shards -join ',')"
    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/PublicAPI.Unshipped.txt' = @(1, 2)
        'src/ElementWiseBaseline.txt' = @(1, 2)
        'src/CompatibilitySuppressions.xml' = @(1, 2)
        'tools/reference-data/vallex_phone_reference.py' = @(1, 2)
        'vercel.json' = @(1, 2)
        'src/Covered.cs' = @(105, 106)
    }
    Assert-True (-not $r.Escalate -and $r.RequiresValidation -and ($r.Shards -join ',') -ceq 'Beta,HeavyNoCoverage') `
        "baseline, reference-data or deployment files beside a mapped source widened or suppressed the selection: $($r.Shards -join ',')"
    # Alone, an analyzer or pack input still has to build, and still selects no shard; the
    # workflow keeps the build through requiresValidation (see buildOnlyChange below).
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/PublicAPI.Unshipped.txt' = @(1, 2) }
    Assert-True (-not $r.Escalate -and $r.Shards.Count -eq 0) 'a public API baseline alone selected shards or escalated'
    # An unlisted text file beside them still escalates.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Resources/Vocabulary.txt' = @(1, 2); 'src/Covered.cs' = @(12, 14) }
    Assert-True $r.Escalate 'an unlisted data file must still escalate'

    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14) }
    Assert-True (-not $r.Escalate) 'a mapped, fully covered change must not escalate'
    Assert-True $r.RequiresValidation 'a mapped change must require validation'
    Assert-True ($r.Shards -contains 'Alpha') 'a change on Alpha lines must select Alpha'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'a change outside Beta lines must not select Beta'
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'always-run shards must always be selected'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tools/TestImpact/Select-Shards.ps1' = @(1, 3); 'src/Covered.cs' = @(12, 14) } `
        -ReviewedControlPaths @('tools/TestImpact/Select-Shards.ps1') -RequiredShards @('Beta')
    Assert-True (-not $r.Escalate -and ($r.Shards -join ',') -ceq 'Alpha,Beta,HeavyNoCoverage') `
        'Reviewed policy impact lost either runtime coverage or explicitly affected workloads.'
    $r = Select-ImpactedShards -Map $map -Changed @{ '.github/test-shards.yml' = @(1, 3) } `
        -ReviewedControlPaths @('.github/test-shards.yml') -RequiredShards @('Beta')
    Assert-True (-not $r.Escalate -and $r.RequiresValidation -and ($r.Shards -join ',') -ceq 'Beta,HeavyNoCoverage') `
        'Manifest-only changes did not run the changed shard and mandatory workloads.'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tools/TestImpact/Select-Shards.ps1' = @(1, 3) } `
        -ReviewedControlPaths @('tools/TestImpact/Select-Shards.ps1')
    Assert-True (-not $r.Escalate -and -not $r.RequiresValidation -and $r.Shards.Count -eq 0) `
        'Tooling-only changes unnecessarily required model execution.'
    Assert-Throws { Select-ImpactedShards -Map $map -Changed @{ '.github/test-shards.yml' = @(1, 3) } -RequiredShards @('Unknown') } `
        'A required workload absent from the map was silently dropped.'

    # An unexecuted range in a mapped file routes to every shard that executes the file. Before this
    # it escalated, and #2100's field declarations and between-method insertions - lines coverage
    # never records - sent a four-file pull request to all 116 shards.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14, 50, 60) }
    Assert-True (-not $r.Escalate) 'an unexecuted range in a mapped file escalated instead of routing to its owners'
    Assert-True ($r.Shards -contains 'Alpha' -and $r.Shards -contains 'Beta') `
        'an unexecuted range was not routed to every shard executing the file'
    Assert-True (@($r.Routes | Where-Object { $_ -like 'Beta <= executes src/Covered.cs, whose changed lines 50-60*' }).Count -eq 1) `
        'the owner route did not record which lines no shard executes'
    Assert-True (@($r.Routes | Where-Object { $_ -like 'Alpha <= executes changed lines 12-14 of src/Covered.cs' }).Count -eq 1) `
        'a covered hit did not record the lines that selected it'

    foreach ($change in @(
        @{ Path = 'src/Unmapped.cs'; Why = 'an unmapped file must escalate' },
        @{ Path = 'Directory.Packages.props'; Why = 'dependency infrastructure must escalate' },
        @{ Path = '.github/workflows/sonarcloud.yml'; Why = 'the validation workflow must escalate' },
        @{ Path = '.github/workflows/test-impact-map.yml'; Why = 'the map workflow must escalate' },
        @{ Path = '.github/workflows/ci-shard-closure-policy.yml'; Why = 'the closure policy must escalate' },
        @{ Path = '.github/test-shards.yml'; Why = 'the shard manifest must escalate' },
        @{ Path = '.github/test-shard-changes.json'; Why = 'the shard history must escalate' },
        @{ Path = '.github/scripts/analyze-test-results.ps1'; Why = 'CI analysis scripts must escalate' },
        @{ Path = '.github/actions/local/action.yml'; Why = 'local actions must escalate' },
        @{ Path = 'tools/TestImpact/Select-Shards.ps1'; Why = 'impact tooling must escalate' },
        @{ Path = 'tools/TestImpact/New-CiValidationCertificate.ps1'; Why = 'certificate policy must escalate' },
        @{ Path = 'tools/TestImpact/Unknown-Helper.ps1'; Why = 'unreviewed tooling must escalate' },
        @{ Path = 'tools/TestImpact/Receive-RequiredArtifact.ps1.backup'; Why = 'transport lookalikes must escalate' },
        @{ Path = 'src/AiDotNet.Generators/TestScaffoldGenerator.cs'; Why = 'build-time source generators must escalate' },
        @{ Path = '.github/workflows/release-please.yml.backup'; Why = 'workflow lookalikes must escalate' },
        @{ Path = '.github/workflows/new-unknown.yml'; Why = 'unknown workflows must escalate' }
    )) {
        $r = Select-ImpactedShards -Map $map -Changed @{ $change.Path = @(1, 2) }
        Assert-True $r.Escalate $change.Why
        Assert-True $r.RequiresValidation "$($change.Why) and require validation"
    }

    foreach ($transportPath in @($script:IndependentToolPaths) + @(
        'TOOLS/TESTIMPACT/RECEIVE-REQUIREDARTIFACT.PS1',
        'tools\TestImpact\Test-RequiredArtifactResume.ps1'
    )) {
        $r = Select-ImpactedShards -Map $map -Changed @{ $transportPath = @(1, 2) }
        Assert-True (-not $r.Escalate -and -not $r.RequiresValidation -and $r.Shards.Count -eq 0) `
            "independently tested tooling launched runtime shards: $transportPath"
        $r = Select-ImpactedShards -Map $map -Changed @{
            $transportPath = @(1, 2)
            'src/Covered.cs' = @(12, 14)
        }
        Assert-True (-not $r.Escalate -and $r.RequiresValidation -and
            ($r.Shards -join ',') -ceq 'Alpha,HeavyNoCoverage') `
            "transport change widened or suppressed mapped runtime coverage: $transportPath"
        $r = Select-ImpactedShards -Map $map -Changed @{
            $transportPath = @(1, 2)
            'tools/TestImpact/Select-Shards.ps1' = @(1, 2)
        }
        Assert-True ($r.Escalate -and $r.RequiresValidation) `
            "independent tooling hid a genuine selection-policy change: $transportPath"
    }

    # The exact counterexample that exposed the original defect: two GitHub-hosted documentation
    # files plus an independent release workflow must not instantiate the model/test graph.
    $r = Select-ImpactedShards -Map $map -Changed @{
        '.github/AUTOMATED_RELEASE_SETUP.md' = @(1, 2)
        '.github/VERSIONING.md' = @(1, 2)
        '.github/workflows/release-please.yml' = @(1, 2)
    }
    Assert-True (-not $r.Escalate) 'the PR #2118 path set must not escalate'
    Assert-True (-not $r.RequiresValidation) 'the PR #2118 path set must suppress runtime validation'
    Assert-True (@($r.Shards).Count -eq 0) 'the PR #2118 path set must select zero shards'

    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/Covered.cs' = @(12, 14)
        '.github/workflows/release-please.yml' = @(1, 2)
        'docs/usage.md' = @(1, 2)
    }
    Assert-True (-not $r.Escalate) 'non-runtime files mixed with a covered edit must stay reducible'
    Assert-True $r.RequiresValidation 'a mixed change containing source must require validation'
    Assert-True ($r.Shards -contains 'Alpha') 'a mixed change must retain the mapped shard'
    Assert-True ($r.Shards -contains 'HeavyNoCoverage') 'a mixed change must retain always-run shards'

    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/Covered.cs' = @(12, 14)
        '.github/workflows/sonarcloud.yml' = @(1, 2)
    } -CurrentPaths @('src/Covered.cs')
    Assert-True (-not $r.Escalate) `
        'a historical selector-control edit permanently escalated a later mapped PR'
    Assert-True ($r.Shards -contains 'Alpha') `
        'ignoring historical control churn dropped the mapped runtime shard'

    $r = Select-ImpactedShards -Map $map -Changed @{
        'src/Covered.cs' = @(12, 14)
        '.github/workflows/sonarcloud.yml' = @(1, 2)
    } -CurrentPaths @('.github/workflows/sonarcloud.yml', 'src/Covered.cs')
    Assert-True $r.Escalate `
        'a selector-control edit in the current PR did not force complete validation'

    $r = Select-ImpactedShards -Map $map -Changed @{}
    Assert-True $r.Escalate 'an empty changed path set must fail closed'
    Assert-True $r.RequiresValidation 'an empty changed path set must require validation'

    $r = Select-ImpactedShards -Map $map -Changed @{} -AuditUnchangedMap
    Assert-True (-not $r.Escalate) 'an unchanged map audit must exercise reduced selection'
    Assert-True $r.RequiresValidation 'an unchanged map audit must still represent runtime validation'
    Assert-True (($r.Shards -join ',') -eq 'HeavyNoCoverage') `
        'an unchanged map audit must select exactly the always-run shards'

    $fullyMapped = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $fullyMapped.alwaysRun = @()
    $r = Select-ImpactedShards -Map $fullyMapped -Changed @{} -AuditUnchangedMap
    Assert-True $r.Escalate `
        'a vacuous unchanged audit with no always-run shards must not authorize certification'

    $docsOnly = @{
        '.github/AUTOMATED_RELEASE_SETUP.md' = @(1, 2)
        '.github/VERSIONING.md' = @(1, 2)
        '.github/workflows/release-please.yml' = @(1, 2)
    }
    $r = Select-ImpactedShards -Map $map -Changed $docsOnly -CurrentPaths @() -AuditUnchangedMap
    Assert-True (-not $r.Escalate -and $r.RequiresValidation -and
        ($r.Shards -join ',') -ceq 'HeavyNoCoverage') `
        'a non-runtime historical audit dropped the mandatory shard set'
    Assert-True ($r.Routes.Count -eq 1) 'a mandatory audit shard lacked its route explanation'
    $r = Select-ImpactedShards -Map $map -Changed $docsOnly
    Assert-True (-not $r.Escalate -and -not $r.RequiresValidation -and $r.Shards.Count -eq 0) `
        'an ordinary docs-only PR was forced to run audit-only validation'
    $r = Select-ImpactedShards -Map $fullyMapped -Changed $docsOnly -CurrentPaths @() -AuditUnchangedMap
    Assert-True $r.Escalate 'a non-runtime audit with no mandatory shards authorized vacuous certification'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props' = @(1, 2) } -AuditUnchangedMap
    Assert-True $r.Escalate 'audit-only mandatory routing suppressed a broad dependency change'

    foreach ($edge in @(10, 20)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True ($r.Shards -contains 'Alpha') "executed boundary line $edge must select Alpha"
    }
    foreach ($edge in @(9, 21)) {
        $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @($edge, $edge) }
        Assert-True (-not $r.Escalate -and $r.Shards -contains 'Alpha' -and $r.Shards -contains 'Beta') `
            "unexecuted boundary line $edge was not routed to the file's owners"
        Assert-True (@($r.Routes | Where-Object { $_ -like 'Alpha <= executes changed lines*' }).Count -eq 0) `
            "unexecuted boundary line $edge was reported as executed"
    }

    # ---- A pull request BEHIND its base branch (the #2100 shape). -------------------------------
    # The map-to-HEAD delta holds master's own later commits (a selector-control edit and an
    # unmapped source file) as well as the pull request's one covered edit.
    $behind = @{
        'src/Covered.cs' = @(12, 14)
        'tools/TestImpact/Select-Shards.ps1' = @(1, 2)
        'src/Finance/MasterOnly.cs' = @(1, 5)
        '.github/workflows/sonarcloud.yml' = @(1, 2)
    }
    # Reproduction: the stale event base.sha made every path current, and it escalated.
    $r = Select-ImpactedShards -Map $map -Changed $behind -CurrentPaths @($behind.Keys)
    Assert-True $r.Escalate 'fixture error: the stale-base reproduction did not escalate'
    # Fixed: the merge commit's first parent limits the change to the pull request's own path.
    $r = Select-ImpactedShards -Map $map -Changed $behind -CurrentPaths @('src/Covered.cs') -ScopeToCurrentPaths
    Assert-True (-not $r.Escalate) 'a pull request behind master escalated on master''s own later commits'
    Assert-True ((@($r.Shards) -join ',') -eq 'Alpha,HeavyNoCoverage') `
        'a pull request behind master did not select exactly its own covering shard plus always-run'
    Assert-True (-not (@($r.Routes) -match 'MasterOnly')) 'master''s later source file influenced the pull request'

    # Scoping must still escalate a control edit that IS in the pull request.
    $r = Select-ImpactedShards -Map $map -Changed $behind `
        -CurrentPaths @('src/Covered.cs', 'tools/TestImpact/Select-Shards.ps1') -ScopeToCurrentPaths
    Assert-True $r.Escalate 'a selector edit inside a behind pull request did not escalate'

    # An empty pull-request path set must never read as a non-runtime change.
    $r = Select-ImpactedShards -Map $map -Changed $behind -CurrentPaths @() -ScopeToCurrentPaths
    Assert-True ($r.Escalate -and $r.RequiresValidation) 'an empty pull-request path set suppressed validation'

    # A pull-request path with no map-to-HEAD hunks cannot be placed in map coordinates.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @(12, 14) } `
        -CurrentPaths @('src/Covered.cs', 'src/SingleRange.cs') -ScopeToCurrentPaths
    Assert-True $r.Escalate 'a pull-request path absent from the map delta did not escalate'

    # ---- Unmapped sources. ------------------------------------------------------------------------
    $deepMap = @{
        schemaVersion = 1; sha = '0123456789abcdef0123456789abcdef01234567'
        knownShards = @('Alpha', 'Beta', 'Gamma'); alwaysRun = @()
        files = @{
            'src/Finance/Agents/Dqn.cs' = @( @{ s = 0; r = @(1, 9) } )
            'src/Finance/Agents/Sac.cs' = @( @{ s = 1; r = @(1, 9) } )
            'src/Finance/Data/Feed.cs' = @( @{ s = 2; r = @(1, 9) } )
        }
    } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/NewAgent.cs' = @(1, 40) }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'Alpha,Beta') `
        'a new source file was not routed to exactly the owners of its own directory'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Brand/New/Deep.cs' = @(1, 4) }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'Alpha,Beta,Gamma') `
        'a new source file with no mapped sibling did not climb to its nearest mapped ancestor'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Orphan/Thing.cs' = @(1, 4) }
    Assert-True $r.Escalate 'a new source file with no mapped neighbour below the root escalated nothing'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/agents.json' = @(1, 4) }
    Assert-True $r.Escalate 'an unmapped non-C# file inside a mapped directory was routed instead of escalating'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/Finance.csproj' = @(1, 4) }
    Assert-True $r.Escalate 'an unmapped project file was routed instead of escalating'

    # Count full-map accesses rather than using a timing threshold. Three unmapped paths
    # need three direct file lookups and one shared directory projection, independent of depth.
    $countedMap = [pscustomobject]@{
        knownShards = $deepMap.knownShards; alwaysRun = @()
        BackingFiles = $deepMap.files; FileReads = 0
    }
    $countedMap | Add-Member -MemberType ScriptProperty -Name files -Value {
        $this.FileReads++
        return $this.BackingFiles
    }
    $r = Select-ImpactedShards -Map $countedMap -Changed @{
        'src/Finance/Agents/NewOne.cs' = @(1, 4)
        'src/Finance/Agents/NewTwo.cs' = @(1, 4)
        'src/Finance/Brand/New/Deep.cs' = @(1, 4)
    }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -ceq 'Alpha,Beta,Gamma') `
        'a shared directory projection changed the selected owners'
    Assert-True ($countedMap.FileReads -eq 4) `
        "directory ownership repeatedly enumerated the map ($($countedMap.FileReads) accesses instead of 4)"
    $countedMap.FileReads = 0
    $r = Select-ImpactedShards -Map $countedMap -Changed @{ 'src/Finance/Agents/Dqn.cs' = @(1, 2) }
    Assert-True ($countedMap.FileReads -eq 1) `
        'a directly mapped change unnecessarily built the directory projection'

    $otherMap = $deepMap | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $otherMap.knownShards = @('OtherAlpha', 'OtherBeta', 'OtherGamma')
    $r = Select-ImpactedShards -Map $otherMap -Changed @{ 'src/Finance/Agents/NewAgent.cs' = @(1, 4) }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -ceq 'OtherAlpha,OtherBeta') `
        'directory ownership leaked between maps selected in the same process'
    $r = Select-ImpactedShards -Map $deepMap -Changed @{ 'src/Finance/Agents/NewAgent.cs' = @(1, 4) }
    Assert-True ((@($r.Shards) -join ',') -ceq 'Alpha,Beta') `
        'returning to the original map retained another map''s directory owners'

    # ---- Test sources, routed through precomputed manifest routes. --------------------------------
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/FooTests.cs' = @(1, 30) } -TestRoutes @{
        'tests/P/FooTests.cs' = [pscustomobject]@{ Routable = $true; Shards = @('Beta'); Why = 'filter selects tests in this source' }
    }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'Beta,HeavyNoCoverage') `
        'a routable test source did not select exactly its owning shard plus always-run'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/Helpers.cs' = @(1, 30) } -TestRoutes @{
        'tests/P/Helpers.cs' = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'declares extension methods' }
    }
    Assert-True ($r.Escalate -and ($r.Reasons -join ';') -like '*declares extension methods*') `
        'an unroutable test source did not escalate with its reason'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/NewTests.cs' = @(1, 30) } -TestRoutes @{
        'tests/P/NewTests.cs' = [pscustomobject]@{ Routable = $false; Unassigned = $true; Shards = @(); Why = 'no shard filter selects any test this source affects' }
    }
    Assert-True (-not $r.Escalate -and (@($r.Shards) -join ',') -eq 'HeavyNoCoverage' -and
        @($r.Files | Where-Object { $_.path -eq 'tests/P/NewTests.cs' -and $_.outcome -eq 'unassigned' }).Count -eq 1) `
        'a test source no shard selects escalated, though the full matrix would not run it either'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'tests/P/FooTests.cs' = @(1, 30) }
    Assert-True $r.Escalate 'a test source with no route (no manifest) did not fail closed'

    # ---- VSTest filter grammar and three-valued evaluation. ---------------------------------------
    function New-Candidate([string] $Fqn, [bool] $PrefixOnly = $false, $Categories = @(), [bool] $AnySuffix = $false) {
        [pscustomobject]@{ Fqn = $Fqn; PrefixOnly = $PrefixOnly; AnySuffix = $AnySuffix; Categories = $Categories }
    }
    # Exercise the shipping catch-all filter, not a hand-copied approximation. Both
    # namespace spellings exist in this project; a full run cannot repair an omitted route.
    $manifestText = Get-Content (Join-Path $PSScriptRoot '../../.github/test-shards.yml') -Raw
    $remaining = [regex]::Match($manifestText,
        '(?m)^  - name: Unit - 13 Remaining[^\r\n]*\r?\n(?:    [^\r\n]*\r?\n)*?    filter: >-\r?\n(?<filter>(?:      [^\r\n]*\r?\n)+)')
    Assert-True $remaining.Success 'Shipping remaining-unit filter was not found.'
    $remainingFilter = ConvertTo-TestFilter $remaining.Groups['filter'].Value
    foreach ($root in @('AiDotNet.Tests', 'AiDotNetTests')) {
        Assert-True ((Test-TestFilter $remainingFilter (New-Candidate "$root.UnitTests.DistributedTraining.DistributedTrainingValidationTests.ShardingConfiguration_Constructor_ThrowsOnNullBackend")) -eq $script:FilterTrue) `
            "Remaining-unit filter omits the distributed tests under $root."
        Assert-True ((Test-TestFilter $remainingFilter (New-Candidate "$root.UnitTests.Diffusion.Models.DDPMModelTests.Test")) -eq $script:FilterFalse) `
            "Remaining-unit filter duplicates partitioned diffusion tests under $root."
    }
    Assert-True ((Test-TestFilter $remainingFilter (New-Candidate 'AiDotNet.Tests.IntegrationTests.DistributedTraining.Test')) -eq $script:FilterFalse) `
        'Remaining-unit filter captured integration tests.'
    $f = ConvertTo-TestFilter "Category!=GPU&Category!=Stress& `n (FullyQualifiedName~UnitTests.Alpha|`n FullyQualifiedName~UnitTests.Beta)"
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Beta.C.M')) -eq $script:FilterTrue) `
        'a folded multi-line filter did not match its second alternative'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Gamma.C.M')) -eq $script:FilterFalse) `
        'a filter matched a test outside every alternative'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Alpha.C.M' $false @('GPU'))) -eq $script:FilterUnknown) `
        'a category present somewhere in the file was treated as definitely on this test'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.UnitTests.Alpha.C.M' $false $null)) -eq $script:FilterUnknown) `
        'unknown categories were treated as known'
    $f = ConvertTo-TestFilter 'A=1|B=2&FullyQualifiedName~Nope'
    Assert-True ($f.Kind -eq 'Or' -and $f.Items[1].Kind -eq 'And') "'&' did not bind tighter than '|'"
    $f = ConvertTo-TestFilter 'FullyQualifiedName~Contracts.A&FullyQualifiedName!~Skip'
    Assert-True ((Test-TestFilter $f (New-Candidate 'N.Contracts.' $true)) -eq $script:FilterUnknown) `
        'an inherited test with an unknown method name was excluded by a method-level filter'
    Assert-True ((Test-TestFilter $f (New-Candidate 'N.Contracts.Skip' $true)) -eq $script:FilterFalse) `
        'a known prefix containing an excluded term was not excluded'
    # The unknown part of an inherited test is ONE method identifier, so it cannot supply a
    # namespace term the prefix lacks - but it can finish a term the prefix ends in.
    $f = ConvertTo-TestFilter 'FullyQualifiedName~P.Alpha'
    Assert-True ((Test-TestFilter $f (New-Candidate 'P.Beta.BTests.' $true)) -eq $script:FilterFalse) `
        'an inherited test was treated as able to acquire a namespace from its method name'
    Assert-True ((Test-TestFilter $f (New-Candidate 'X.P.' $true)) -eq $script:FilterUnknown) `
        'a method name completing a term the prefix ends in was ruled out'
    Assert-True ((Test-TestFilter $f (New-Candidate 'N.' $true @() $true)) -eq $script:FilterUnknown) `
        'a test whose class could not be read was ruled out by a namespace term'
    Assert-True ((Test-TestFilter (ConvertTo-TestFilter 'FullyQualifiedName~alpha') (New-Candidate 'X.Alpha.M')) -eq $script:FilterUnknown) `
        'a case-only match was decided instead of left unknown'
    Assert-True ((Test-TestFilter (ConvertTo-TestFilter 'Category=Playground') (New-Candidate 'X.Y.M')) -eq $script:FilterFalse) `
        'a positive category filter matched a file with no categories'
    foreach ($bad in @('FullyQualifiedName~A&', '(FullyQualifiedName~A', 'FullyQualifiedName', 'A=\(x\)')) {
        Assert-Throws { ConvertTo-TestFilter $bad } "malformed filter '$bad' was accepted"
    }

    # ---- C# test-shape parsing. -------------------------------------------------------------------
    $source = @'
using Xunit;
namespace AiDotNet.Tests.IntegrationTests.Finance;

// public class CommentedOut { [Fact] public void Ghost() {} }
[Trait("Category", "Slow")]
public class TradingTests : IClassFixture<Fixture>
{
    private const string Braces = "{ class Fake { ";
    private readonly char _brace = '}';

    [Fact]
    public void Learns() { var s = $"{1}"; }

    [Theory]
    [InlineData(new[] { 1, 2 })]
    public async Task Converges<T>(int[] x) { await Task.Yield(); }

    public class Nested
    {
        [Fact, Trait("Category", "Nested")]
        public void Inner() { }
    }
}

internal sealed class Helper { }
file class Private { }
'@
    $shape = Get-CSharpTestShape -Text $source
    Assert-True ($null -eq $shape.ParseError) "valid C# was reported unparseable: $($shape.ParseError)"

    # An interpolation hole is CODE, so it may hold a comment or a nested literal whose braces and
    # quotes are not the enclosing literal's. Getting either wrong reports 'unbalanced braces',
    # which fails Get-TestFileRoutes closed and escalates to the full 164-shard matrix. Neither
    # form occurs in the repo today (0 of 11,928 sources), so these guard the scanner, not a
    # current failure - the nested-quote form did occur, and did exactly that.
    $q = [string][char]34
    $holeCases = @(
        @{ Name = 'nested string in a hole'
           Body = 'var s = $"a{string.Join(", ", xs)}b";' },
        @{ Name = 'block comment in a hole'
           Body = 'var s = $"a{/* } ' + $q + ' */ 0}b";' },
        @{ Name = 'line comment in a hole'
           Body = "var s = `$`"a{ 0 // } $q`n }b`";" },
        @{ Name = 'raw interpolated hole containing a fence'
           Body = 'var s = $' + ($q * 3) + 'a{ ' + ($q * 3) + 'x' + ($q * 3) + ' }b' + ($q * 3) + ';' },
        @{ Name = 'doubled braces are literal, not a hole'
           Body = 'var s = $"{{ not a hole }}";' }
    )
    foreach ($case in $holeCases) {
        $text = "namespace N { public class C { public void M() { $($case.Body) } } }"
        $masked = ConvertTo-CodeOnlyCSharp -Text $text
        $open = ([regex]::Matches($masked, '\{')).Count
        $close = ([regex]::Matches($masked, '\}')).Count
        Assert-True ($open -eq $close) `
            "masking left braces unbalanced ($open open, $close close) for: $($case.Name)"
        Assert-True ($null -eq (Get-CSharpTestShape -Text $text).ParseError) `
            "a literal was misparsed and reported unbalanced braces for: $($case.Name)"
    }

    # .editorconfig must compile but must not drag in the shard matrix. It cannot change runtime
    # behaviour - nothing under src/AiDotNet.Generators reads AnalyzerConfigOptions - but it can
    # raise an analyzer to error, so it is BuildOnly rather than NonRuntime. Measured on PR #2112,
    # this was one of the two files escalating it to 130 shards.
    Assert-True ((Get-ChangedPathImpact -Path '.editorconfig') -eq [ChangedPathImpact]::BuildOnly) `
        '.editorconfig is not classified BuildOnly'
    Assert-True ((Get-ChangedPathImpact -Path 'src/Nested/.editorconfig') -eq [ChangedPathImpact]::BuildOnly) `
        'a nested .editorconfig is not classified BuildOnly'
    # tools/: a standalone tool nothing here builds, tests or invokes cannot move any shard (10 of 22 escalated PR
    # runs in late September 2026 escalated on such a file). A referenced one keeps escalating, and TestImpact stays
    # selection control. These read the real tree, so they fail loudly if a tool gains or loses a reference.
    if (Test-Path -LiteralPath 'tools/CoverageReportReview') {
        Assert-True ((Get-ChangedPathImpact -Path 'tools/CoverageReportReview/Program.cs') -eq [ChangedPathImpact]::NonRuntime) `
            'an unreferenced tool (CoverageReportReview) is not NonRuntime'
    }
    if (Test-Path -LiteralPath 'tools/AiDotNet.Evolve.Cli') {
        Assert-True ((Get-ChangedPathImpact -Path 'tools/AiDotNet.Evolve.Cli/Program.cs') -eq [ChangedPathImpact]::MapCandidate) `
            'a tool the test project references (AiDotNet.Evolve.Cli) was downgraded'
    }
    Assert-True ((Get-ChangedPathImpact -Path 'tools/TestImpact/Unknown-Helper.ps1') -eq [ChangedPathImpact]::SelectionControl) `
        'tools/TestImpact lost its selection-control classification'
    Assert-True ((Get-ChangedPathImpact -Path 'tools/loose-file.txt') -eq [ChangedPathImpact]::MapCandidate) `
        'a file directly under tools/ has no directory to scope by and must keep the old behaviour'
    # Other top-level trees, by the same rule (api/ escalated PR #2275 on a package-lock.json). A route string
    # such as [Route("api/...")] in product code is not a reference; a solution entry or a test's
    # Path.Combine(..., "samples", ...) is. These read the real tree too.
    if (Test-Path -LiteralPath 'api/package.json') {
        Assert-True ((Get-ChangedPathImpact -Path 'api/package-lock.json') -eq [ChangedPathImpact]::NonRuntime) `
            'an unreferenced top-level tree (api/) is not NonRuntime'
    }
    if (Test-Path -LiteralPath 'benchmarks') {
        # The solution builds benchmarks/, so it must never be NonRuntime (that would skip the compile). It is
        # build-only while no test or product project consumes it, and a map candidate again the day one does.
        Assert-True ((Get-ChangedPathImpact -Path 'benchmarks/AiDotNet.Benchmarks/Program.cs') -eq [ChangedPathImpact]::BuildOnly) `
            'a tree the solution builds and no test consumes (benchmarks/) is not build-only'
        $savedBuildOnlyDirectories = $script:BuildOnlyDirectories
        try {
            $script:BuildOnlyDirectories = @()
            Assert-True ((Get-ChangedPathImpact -Path 'benchmarks/AiDotNet.Benchmarks/Program.cs') -eq [ChangedPathImpact]::MapCandidate) `
                'a tree the solution builds (benchmarks/) was downgraded to non-runtime by the unreferenced-tree rule'
        }
        finally { $script:BuildOnlyDirectories = $savedBuildOnlyDirectories }
    }
    if (Test-Path -LiteralPath 'samples') {
        Assert-True ((Get-ChangedPathImpact -Path 'samples/x/y.txt') -eq [ChangedPathImpact]::MapCandidate) `
            'a tree a test reads through Path.Combine (samples/) was downgraded'
    }
    Assert-True ((Get-ChangedPathImpact -Path 'root-file.txt') -eq [ChangedPathImpact]::MapCandidate) `
        'a root-level file has no tree to scope by and must keep the old behaviour'
    # R9: .github/ files nothing reads are non-runtime; one a workflow invokes stays full validation.
    if (Test-Path -LiteralPath '.github/CODEOWNERS') {
        Assert-True ((Get-ChangedPathImpact -Path '.github/CODEOWNERS') -eq [ChangedPathImpact]::NonRuntime) `
            'CODEOWNERS, which no workflow reads, is not NonRuntime'
    }
    if (Test-Path -LiteralPath '.github/dependabot.yml') {
        Assert-True ((Get-ChangedPathImpact -Path '.github/dependabot.yml') -eq [ChangedPathImpact]::NonRuntime) `
            'dependabot.yml, which no workflow reads, is not NonRuntime'
    }
    if (Test-Path -LiteralPath '.github/scripts/Invoke-Shard.ps1') {
        Assert-True ((Get-ChangedPathImpact -Path '.github/scripts/Compare-IsaVerdicts.ps1') -eq [ChangedPathImpact]::NonRuntime) `
        'a script only the cross-ISA nightly runs was treated as validation infrastructure'
    Assert-True ((Get-ChangedPathImpact -Path '.github/scripts/find-pr-new-failures.ps1') -eq [ChangedPathImpact]::FullValidation) `
        'a script this validation workflow runs was spared'
    Assert-True (-not (Test-ScriptUsedOnlyByIndependentWorkflows -Path '.github/scripts/no-such-script-anywhere.ps1')) `
        'an unreferenced script was spared without review'
    Assert-True ((Get-ChangedPathImpact -Path '.github/scripts/Invoke-Shard.ps1') -eq [ChangedPathImpact]::FullValidation) `
            'a script the test workflow invokes (Invoke-Shard.ps1) was downgraded'
    }
    Assert-True ((Get-ChangedPathImpact -Path '.github/a.yml') -eq [ChangedPathImpact]::FullValidation) `
        'a short, generic .github file name must not be trusted as unreferenced'
    # The selection report lists every changed path, so a build-only file must be recorded as a decision even
    # beside runtime changes (it used to fall through the switch and vanish from the report).
    $reportMap = [pscustomobject]@{ knownShards = @('Alpha'); alwaysRun = @(); files = [pscustomobject]@{
        'src/Covered.cs' = @([pscustomobject]@{ s = 0; r = @(10, 20) }) } }
    $withConfig = Select-ImpactedShards -Map $reportMap -Changed @{ '.editorconfig' = @(1, 2); 'src/Covered.cs' = @(12, 14) }
    $configDecision = @($withConfig.Files | Where-Object { $_.path -eq '.editorconfig' })
    Assert-True ($configDecision.Count -eq 1 -and $configDecision[0].category -eq 'BuildOnly' -and $configDecision[0].outcome -eq 'skipped') `
        '.editorconfig beside a runtime change is missing from the per-file selection decisions'
    Assert-True (@($withConfig.Files | Where-Object { $_.path -eq 'src/Covered.cs' -and $_.outcome -eq 'routed' }).Count -eq 1) `
        'the runtime file beside .editorconfig lost its routed decision'
    # The neighbours it used to sit beside must keep escalating: they really can change compilation
    # output, not merely diagnostics.
    foreach ($shared in 'Directory.Build.props', 'Directory.Packages.props', 'global.json', 'nuget.config') {
        Assert-True ((Get-ChangedPathImpact -Path $shared) -eq [ChangedPathImpact]::FullValidation) `
            "$shared stopped requiring full validation"
    }
    # BuildOnly must never be mistaken for NonRuntime: that shares a gate with the build jobs, so
    # an analyzer promoted to error would ship without ever being compiled.
    Assert-True ((Get-ChangedPathImpact -Path '.editorconfig') -ne [ChangedPathImpact]::NonRuntime) `
        '.editorconfig was downgraded to NonRuntime, which would skip the build that catches it'
    Assert-True ((Get-ChangedPathImpact -Path 'src/AiDotNet.Generators/TestScaffoldGenerator.cs') -eq [ChangedPathImpact]::FullValidation) `
        'a generator edit stopped requiring full validation'
    $fqns = @($shape.Tests | Where-Object { -not $_.PrefixOnly } | ForEach-Object Fqn | Sort-Object)
    $expectedFqns = @('AiDotNet.Tests.IntegrationTests.Finance.TradingTests+Nested.Inner',
        'AiDotNet.Tests.IntegrationTests.Finance.TradingTests.Converges',
        'AiDotNet.Tests.IntegrationTests.Finance.TradingTests.Learns') | Sort-Object
    Assert-True (($fqns -join ',') -eq ($expectedFqns -join ',')) "test names were misread: $($fqns -join ',')"
    Assert-True (@($shape.Tests | Where-Object PrefixOnly).Count -eq 0) 'an interface base was treated as an inherited class'
    Assert-True ((@($shape.Categories | Sort-Object) -join ',') -eq 'Nested,Slow') 'literal categories were not collected'
    $topLevel = @($shape.Types | Where-Object { $null -eq $_.Parent -and -not $_.FileLocal } | ForEach-Object Name | Sort-Object)
    Assert-True (($topLevel -join ',') -eq 'Helper,TradingTests') `
        "referenceable top-level types were misread: $($topLevel -join ',')"

    foreach ($kind in @('class', 'record', 'record class')) {
        $shape = Get-CSharpTestShape -Text "namespace N; [Obsolete] public abstract`n$kind Base { }"
        Assert-True (@($shape.Types | Where-Object IsAbstract).Count -eq 1 -and $shape.Hazard -like '*abstract class*') `
            "multiline abstract $kind lost the generated-base hazard"
        $shape = Get-CSharpTestShape -Text "namespace N; file`n$kind Local { }"
        Assert-True (@($shape.Types | Where-Object FileLocal).Count -eq 1) `
            "multiline file $kind was considered externally referenceable"
    }
    $shape = Get-CSharpTestShape -Text @'
namespace N;
public abstract class Earlier { }
file class EarlierLocal { }
[Trait("abstract", "file")]
public /* abstract file */
class Current { }
'@
    $current = @($shape.Types | Where-Object Name -EQ 'Current')
    Assert-True ($current.Count -eq 1 -and -not $current[0].IsAbstract -and -not $current[0].FileLocal) `
        'a declaration inherited modifiers from earlier types, an attribute, or a comment'

    $shape = Get-CSharpTestShape -Text "namespace N { public class D : ModelContractBase<int> { } }"
    Assert-True (@($shape.Tests | Where-Object { $_.PrefixOnly -and $_.Fqn -eq 'N.D.' }).Count -eq 1) `
        'a class with a class base did not expose its inherited tests as open-ended'
    Assert-True ($null -eq $shape.Categories) 'categories inherited from a base class were treated as known'

    $shape = Get-CSharpTestShape -Text "namespace N { public static class X { public static int Twice(this int v) => v * 2; } }"
    Assert-True ($shape.Hazard -eq 'declares extension methods') 'extension methods were not flagged'
    $shape = Get-CSharpTestShape -Text "namespace N { public class X { "
    Assert-True ([bool] $shape.ParseError) 'unbalanced braces were not reported'

    # Modifiers on the line ABOVE the keyword belong to the same declaration.
    $shape = Get-CSharpTestShape -Text "namespace N;`n[Serializable]`npublic`n    abstract`nclass Base { [Fact] public void A() { } }`nfile`nclass Hidden { }"
    Assert-True (@($shape.Types | Where-Object { $_.Name -eq 'Base' -and $_.IsAbstract }).Count -eq 1) `
        'an abstract modifier on the line above the class keyword was missed'
    Assert-True ([bool] $shape.Hazard) 'a multi-line abstract base escaped the abstract-base hazard'
    Assert-True (@($shape.Types | Where-Object { $_.Name -eq 'Hidden' -and $_.FileLocal }).Count -eq 1) `
        'a file modifier on the line above the class keyword was missed'
    $shape = Get-CSharpTestShape -Text "namespace N;`npublic abstract int Unrelated;`npublic class Concrete { }"
    Assert-True (@($shape.Types | Where-Object { $_.Name -eq 'Concrete' -and -not $_.IsAbstract }).Count -eq 1) `
        'a modifier from the PREVIOUS declaration leaked onto the next type'

    # ---- Routing with injected file reads and reference search. -----------------------------------
    $manifest = @(
        [pscustomobject]@{ name = 'Alpha'; project = 'tests/P/P.csproj'; filter = 'FullyQualifiedName~P.Alpha' },
        [pscustomobject]@{ name = 'Beta'; project = 'tests/P/P.csproj'; filter = 'FullyQualifiedName~P.Beta' },
        [pscustomobject]@{ name = 'Other'; project = 'tests/Q/Q.csproj'; filter = 'FullyQualifiedName~P' }
    )
    $files = @{
        'tests/P/Alpha/ATests.cs' = 'namespace P.Alpha; public class ATests { [Fact] public void A() { } }'
        'tests/P/Shared/Base.cs'  = 'namespace P.Shared; public class Base { [Fact] public void Common() { } }'
        'tests/P/Shared/Abstract.cs' = 'namespace P.Shared; public abstract class ShapeBase { [Fact] public void Common() { } }'
        'tests/P/Alpha/MultilineAbstract.cs' = "namespace P.Alpha; public abstract`nclass MultilineBase { [Fact] public void Common() { } }"
        'tests/P/Shared/GenBase.cs' = 'namespace P.Shared; public class LayerHarness { }'
        'tests/P/Beta/BTests.cs'  = 'namespace P.Beta; public class BTests : Base { }'
        'tests/P/Shared/Ext.cs'   = 'namespace P.Shared; public static class Ext { public static int X(this int v) => v; }'
        'tests/P/Shared/Unused.cs' = 'namespace P.Shared; public class Unused { }'
    }
    $referrers = @{ 'Base' = @('tests/P/Beta/BTests.cs'); 'ATests' = @(); 'BTests' = @(); 'Unused' = @() }
    $read = { param($path) $files[$path] }
    $find = { param($names, $directory) @($names | ForEach-Object { $referrers[$_] } | Where-Object { $_ }) }
    $generatorNames = @('LayerHarness')
    $findGenerated = { param($names) @($names | Where-Object { $generatorNames -contains $_ }) }
    $routed = Get-TestFileRoutes -Paths @('tests/P/Alpha/ATests.cs', 'tests/P/Shared/Base.cs', 'tests/P/Shared/Ext.cs',
        'tests/P/Shared/Unused.cs', 'tests/Z/Stray.cs', 'tests/P/Shared/Abstract.cs', 'tests/P/Shared/GenBase.cs',
        'tests/P/Alpha/MultilineAbstract.cs') `
        -Manifest $manifest -ReadFile $read -FindReferrers $find -FindBuildTimeReferences $findGenerated
    Assert-True ($routed['tests/P/Alpha/ATests.cs'].Routable -and
        (@($routed['tests/P/Alpha/ATests.cs'].Shards) -join ',') -eq 'Alpha') `
        'a self-contained test file was not routed to exactly its own shard'
    Assert-True ($routed['tests/P/Shared/Base.cs'].Routable -and
        (@($routed['tests/P/Shared/Base.cs'].Shards) -join ',') -eq 'Beta') `
        'a base class was not routed to the shard running its derived tests (and not its own namespace)'
    Assert-True (-not $routed['tests/P/Shared/Ext.cs'].Routable) 'an extension-method helper was routed'
    Assert-True (-not $routed['tests/P/Shared/Unused.cs'].Routable) 'a support file with no tests and no consumers was routed'
    Assert-True (-not $routed['tests/Z/Stray.cs'].Routable) 'a test file outside every shard project was routed'
    Assert-True (-not $routed['tests/P/Shared/Abstract.cs'].Routable) `
        'an abstract test base was routed although build-time generated classes may derive from it'
    Assert-True (-not $routed['tests/P/Alpha/MultilineAbstract.cs'].Routable -and
        $routed['tests/P/Alpha/MultilineAbstract.cs'].Why -like '*abstract class*') `
        'a multiline abstract base was restricted to its namespace instead of escalating for generated descendants'
    Assert-True (-not $routed['tests/P/Shared/GenBase.cs'].Routable -and
        $routed['tests/P/Shared/GenBase.cs'].Why -like '*source generator references*') `
        'a test type a source generator references was routed by reading only the tree'
    $wide = Get-TestFileRoutes -Paths @('tests/P/Shared/Base.cs') -Manifest $manifest -ReadFile $read `
        -FindReferrers $find -FindBuildTimeReferences $findGenerated -MaximumClosure 1
    Assert-True (-not $wide['tests/P/Shared/Base.cs'].Routable) 'a closure wider than the limit was routed'

    # Reviewed generator inputs bound the name-based graph, without allowing known
    # generated bases or collections through. Unknown build inputs still fail closed.
    $files['tests/P/Beta/Inherited.cs'] = 'namespace P.Beta; public class Inherited : ShapeBase { }'
    $referrers['ShapeBase'] = @('tests/P/Beta/Inherited.cs')
    $files['tests/P/Shared/Collection.cs'] = 'namespace P.Shared; [Xunit.CollectionDefinition("Shared")] public class Fixture { }'
    $files['tests/P/Alpha/Consumer.cs'] = 'namespace P.Alpha; [Collection("Shared")] public class Consumer { [Fact] public void Test() { } }'
    $files['tests/P/Beta/ConstantConsumer.cs'] = 'namespace P.Beta; [Collection(Names.Shared)] public class ConstantConsumer { [Fact] public void Test() { } }'
    $collections = { param($names, $directory) @($files.Keys | Where-Object {
        $_.StartsWith($directory) -and (Test-CSharpCollectionConsumer -Text $files[$_] -Names $names)
    }) }
    $routingArgs = @{ Manifest = $manifest; ReadFile = $read; FindReferrers = $find;
        FindBuildTimeReferences = $findGenerated; FindCollectionReferrers = $collections;
        GeneratedDependenciesKnown = $true; GeneratedDependencyProjects = @('tests/P/') }
    $bounded = Get-TestFileRoutes -Paths @('tests/P/Shared/Abstract.cs', 'tests/P/Shared/Collection.cs', 'tests/P/Shared/GenBase.cs') @routingArgs
    Assert-True ($bounded['tests/P/Shared/Abstract.cs'].Routable -and
        ($bounded['tests/P/Shared/Abstract.cs'].Shards -join ',') -eq 'Beta') 'reviewed source-only abstract descendants were not routed'
    Assert-True ($bounded['tests/P/Shared/Collection.cs'].Routable -and
        ($bounded['tests/P/Shared/Collection.cs'].Shards -join ',') -eq 'Alpha,Beta') 'literal or unresolved collection consumer was omitted'
    Assert-True (-not $bounded['tests/P/Shared/GenBase.cs'].Routable) 'known generated base escaped the fallback'
    $generatorNames += 'Shared'
    $generatedCollection = Get-TestFileRoutes -Paths @('tests/P/Shared/Collection.cs') @routingArgs
    Assert-True (-not $generatedCollection['tests/P/Shared/Collection.cs'].Routable) 'generated collection consumers were omitted'
    $generatorNames = @('LayerHarness')
    $routingArgs.GeneratedDependenciesKnown = $false
    $unknown = Get-TestFileRoutes -Paths @('tests/P/Shared/Abstract.cs', 'tests/P/Shared/Collection.cs') @routingArgs
    Assert-True (-not $unknown['tests/P/Shared/Abstract.cs'].Routable -and -not $unknown['tests/P/Shared/Collection.cs'].Routable) 'unknown generator contract escaped the fallback'
    $routingArgs.GeneratedDependenciesKnown = $true
    $routingArgs.FindCollectionReferrers = $null
    $missingDiscovery = Get-TestFileRoutes -Paths @('tests/P/Shared/Collection.cs') @routingArgs
    Assert-True (-not $missingDiscovery['tests/P/Shared/Collection.cs'].Routable) 'missing collection discovery silently skipped tests'
    $shape = Get-CSharpTestShape -Text '[CollectionDefinition(Name)] public class Fixture { }' -GeneratedDependenciesKnown $true
    Assert-True ([bool] $shape.Hazard) 'unresolved collection definition was guessed'
    $shape = Get-CSharpTestShape -Text '[CollectionDefinition(nameof(N.Fixture), DisableParallelization = true)] public class Fixture { }' -GeneratedDependenciesKnown $true
    Assert-True (-not $shape.Hazard -and ($shape.Collections -join ',') -eq 'Fixture') 'nameof collection definition was not resolved'
    Assert-True (-not (Test-CSharpCollectionConsumer -Text '[Collection("Different")] class T { }' -Names @('Shared'))) 'different literal collection was included'
    Assert-True (Test-CSharpCollectionConsumer -Text '[Collection("shared")] class T { }' -Names @('Shared')) 'case-variant collection was unsafely excluded'
    Assert-True (Test-CSharpCollectionConsumer -Text '[Collection("Sha" + "red")] class T { }' -Names @('Shared')) 'computed collection name was excluded'
    Assert-True (Test-CSharpCollectionConsumer -Text 'using Alias = Xunit.CollectionAttribute; [Alias("Shared")] class T { }' -Names @('Shared')) 'aliased collection consumer was excluded'
    Assert-True (-not (Test-CSharpCollectionConsumer -Text '// [Collection("Shared")]' -Names @('Shared'))) 'comment was treated as a collection consumer'
    Assert-True (Test-ReviewedGeneratedDependencies -ChangedPaths @()) 'an empty change was treated as a generator change'
    Assert-True (Test-ReviewedGeneratedDependencies -ChangedPaths @('src/Models/Model.cs', 'tests/P/Alpha/AlphaTests.cs', 'docs/README.md')) 'a change outside the generator inputs was treated as one'
    foreach ($inputPath in @('src/AiDotNet.Generators/TestScaffoldGenerator.cs', 'Directory.Packages.props', 'Directory.Build.targets',
            'tests/AiDotNet.Tests/AiDotNetTests.csproj', 'global.json', 'src/sub/global.json')) {
        Assert-True (-not (Test-ReviewedGeneratedDependencies -ChangedPaths @('src/Models/Model.cs', $inputPath))) "a change to $inputPath kept the generated-dependency contract"
    }
    $emitter = ConvertTo-CodeOnlyCSharp -Text '// FakeBase appears only in a comment
var source = "class Test : RealBase { } // string contents survive"; /* AnotherFakeBase */' -PreserveStrings
    Assert-True ($emitter -notmatch 'FakeBase' -and $emitter -match 'RealBase' -and $emitter -match 'string contents survive') 'generator comment filtering removed emitted dependencies or retained comments'
    $files['tests/P/Shared/Collection.cs'] = 'namespace P.Shared; [CollectionDefinition("Renamed")] public class Fixture { }'
    $files['tests/P/Alpha/Consumer.cs'] = 'namespace P.Alpha; [Collection("Shared")] public class Consumer { [Fact] public void Test() { } }'
    $files['tests/P/Beta/ConstantConsumer.cs'] = 'namespace P.Beta; [Collection("Renamed")] public class ConstantConsumer { [Fact] public void Test() { } }'
    $routingArgs.FindCollectionReferrers = $collections
    $routingArgs.ReadPriorFile = { param($path) if ($path -eq 'tests/P/Shared/Collection.cs') { 'namespace P.Shared; [CollectionDefinition("Shared")] public class Fixture { }' } }
    $renamed = Get-TestFileRoutes -Paths @('tests/P/Shared/Collection.cs') @routingArgs
    Assert-True ($renamed['tests/P/Shared/Collection.cs'].Routable -and
        ($renamed['tests/P/Shared/Collection.cs'].Shards -join ',') -eq 'Alpha,Beta') 'collection rename omitted old or new consumers'

    # ---- Carrying a change's ranges back to the map's numbering. -------------------------------
    # Hand-checked cases first. map -> base: line 4 replaced (4 -> 4), 2 lines inserted after 7,
    # line 12 deleted.
    $hunks = @([int[]] @(4, 1, 4, 1), [int[]] @(7, 0, 8, 2), [int[]] @(12, 1, 13, 0))
    Assert-True (((Convert-RangesThroughHunks -Ranges @(2, 2) -Hunks $hunks) -join ',') -eq '2,2') 'an unchanged early line moved'
    Assert-True (((Convert-RangesThroughHunks -Ranges @(10, 10) -Hunks $hunks) -join ',') -eq '8,8') 'a line after an insertion did not shift back'
    Assert-True (((Convert-RangesThroughHunks -Ranges @(8, 8) -Hunks $hunks) -join ',') -eq '7,8') 'an inserted line did not map to its insertion point'
    Assert-True (((Convert-RangesThroughHunks -Ranges @(4, 4) -Hunks $hunks) -join ',') -eq '4,4') 'a replaced line did not map to what it replaced'
    # Map line 12 was deleted from between base lines 13 and 14: both neighbours take it, and the
    # line before them (base 12 = map 10) does not. An off-by-one here once shifted the pair to 12/13.
    Assert-True (((Convert-RangesThroughHunks -Ranges @(13, 13) -Hunks $hunks) -join ',') -match '(^|,)12,12(,|$)') 'a change before a deletion lost the deleted line'
    Assert-True (((Convert-RangesThroughHunks -Ranges @(14, 14) -Hunks $hunks) -join ',') -match '(^|,)12,12(,|$)') 'a change after a deletion lost the deleted line'
    Assert-True (((Convert-RangesThroughHunks -Ranges @(12, 12) -Hunks $hunks) -join ',') -eq '10,10') 'a change two lines from a deletion took the deleted line'
    Assert-True (((Convert-RangesThroughHunks -Ranges @(20, 20) -Hunks $hunks) -join ',') -eq '19,19') 'a line after a deletion did not shift forward'
    Assert-True (Test-RangesTouchIntroducedLines -Ranges @(9, 9) -Hunks $hunks) 'a change on an introduced line was not flagged'
    Assert-True (-not (Test-RangesTouchIntroducedLines -Ranges @(2, 2) -Hunks $hunks)) 'a change on a mapped line was flagged'

    # Then a property check against real git diffs. Every line is a unique token, so "the same
    # code" is unambiguous: whenever a change touches a base line whose token exists in the map, the
    # map line holding that token must be inside the carried-back ranges. Edits include MOVES,
    # which git shows as delete + insert - the case the introduced-line sweep exists for.
    $propertyRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("carry-ranges-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $propertyRoot -Force | Out-Null
    try {
        $random = [System.Random]::new(20260911)
        function New-EditedLines([string[]] $Lines, [string] $Tag, [int] $Edits) {
            $list = [System.Collections.Generic.List[string]]::new([string[]] $Lines)
            for ($e = 0; $e -lt $Edits; $e++) {
                $at = $random.Next([Math]::Max(1, $list.Count))
                switch ($random.Next(4)) {
                    0 { if ($list.Count -gt 3) { $list.RemoveAt($at) } }
                    1 { $list.Insert($at, "$Tag$e-$($random.Next(100000))") }
                    2 { if ($list.Count -gt 0) { $list[$at] = "$Tag$e-$($random.Next(100000))" } }
                    3 { if ($list.Count -gt 3) { $moved = $list[$at]; $list.RemoveAt($at); $list.Insert($random.Next($list.Count + 1), $moved) } }
                }
            }
            return , ([string[]] $list.ToArray())
        }
        function Get-FileHunks([string] $From, [string] $To) {
            $diff = @(& git diff --no-index --no-renames --no-ext-diff -U0 -- $From $To 2>$null)
            $parsed = ConvertTo-DiffHunks -DiffLines $diff
            if ($parsed.Count -eq 0) { return }
            # Emits each hunk (an int[4]) separately; callers collect them with @().
            return @($parsed.Values)[0]
        }

        $violations = 0
        for ($trial = 0; $trial -lt 120; $trial++) {
            $mapLines = [string[]] @(1..25 | ForEach-Object { "m$_" })
            $baseLines = New-EditedLines $mapLines 'b' ($random.Next(1, 5))
            $headLines = New-EditedLines $baseLines 'h' ($random.Next(1, 4))
            $mapFile = Join-Path $propertyRoot "map.txt"; $baseFile = Join-Path $propertyRoot "base.txt"; $headFile = Join-Path $propertyRoot "head.txt"
            [System.IO.File]::WriteAllText($mapFile, ($mapLines -join "`n") + "`n")
            [System.IO.File]::WriteAllText($baseFile, ($baseLines -join "`n") + "`n")
            [System.IO.File]::WriteAllText($headFile, ($headLines -join "`n") + "`n")

            $own = @(Get-FileHunks $baseFile $headFile)
            if ($own.Count -eq 0) { continue }
            $ownRanges = [System.Collections.Generic.List[int]]::new()
            foreach ($h in $own) {
                if ($h[1] -gt 0) { $ownRanges.Add($h[0]); $ownRanges.Add($h[0] + $h[1] - 1) }
                else { $ownRanges.Add([Math]::Max(1, $h[0])); $ownRanges.Add([Math]::Max(1, $h[0] + 1)) }
            }
            $mapToBase = @(Get-FileHunks $mapFile $baseFile)
            $carried = [System.Collections.Generic.List[int]]::new([int[]] (Convert-RangesThroughHunks -Ranges $ownRanges.ToArray() -Hunks $mapToBase))
            if (Test-RangesTouchIntroducedLines -Ranges $ownRanges.ToArray() -Hunks $mapToBase) {
                foreach ($h in @(Get-FileHunks $mapFile $headFile)) {
                    if ($h[1] -gt 0) { $carried.Add($h[0]); $carried.Add($h[0] + $h[1] - 1) }
                    else { $carried.Add([Math]::Max(1, $h[0])); $carried.Add([Math]::Max(1, $h[0] + 1)) }
                }
            }

            foreach ($h in $own) {
                $touched = if ($h[1] -gt 0) { @($h[0]..($h[0] + $h[1] - 1)) } else { @($h[0], ($h[0] + 1)) }
                foreach ($baseLine in $touched) {
                    if ($baseLine -lt 1 -or $baseLine -gt $baseLines.Count) { continue }
                    $mapLine = [Array]::IndexOf($mapLines, $baseLines[$baseLine - 1]) + 1
                    if ($mapLine -lt 1) { continue }
                    $inside = $false
                    for ($c = 0; $c + 1 -lt $carried.Count; $c += 2) {
                        if ($mapLine -ge $carried[$c] -and $mapLine -le $carried[$c + 1]) { $inside = $true; break }
                    }
                    if (-not $inside) { $violations++ }
                }
            }
        }
        Assert-True ($violations -eq 0) "carrying ranges back to the map dropped $violations mapped line(s) a change touched"
    }
    finally { Remove-Item -LiteralPath $propertyRoot -Recurse -Force -ErrorAction SilentlyContinue }

    # An empty build-time pathspec would make git grep search the whole repository.
    $configuredBuildTime = $script:BuildTimeDirectories
    try {
        $script:BuildTimeDirectories = @()
        Assert-Throws { Find-GitBuildTimeReferences -Names @('Anything') } `
            'an empty build-time directory list searched the whole repository instead of failing'
    }
    finally { $script:BuildTimeDirectories = $configuredBuildTime }

    # Faithful to real git: every hunk header is followed by its body lines. An earlier revision
    # used header-only fixtures, which real git never emits - and which masked the forged-header
    # parse bug that body-line counting exists to prevent (see check 15).
    $diff = @('--- a/src/Covered.cs', '+++ b/src/Covered.cs', '@@ -1,0 +1,50 @@') +
            @(1..50 | ForEach-Object { "+inserted $_" }) +
            @('@@ -500 +550 @@', '-old text', '+new text')
    $parsed = ConvertTo-ChangedRanges -DiffLines $diff
    $ranges = @($parsed['src/Covered.cs'])
    Assert-True ($ranges -contains 500) 'the parser must report old-side line numbers'
    Assert-True (-not ($ranges -contains 550)) 'the parser must not report new-side line numbers'

    $diff = @('--- a/src/Covered.cs', '+++ b/src/Covered.cs', '@@ -100 +100 @@', '-x', '+y',
              '--- a/src/Deleted.cs', '+++ /dev/null', '@@ -1,200 +0,0 @@') +
            @(1..200 | ForEach-Object { "-gone $_" })
    $parsed = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($parsed.ContainsKey('src/Deleted.cs')) 'a deleted file must be reported under its own path'
    Assert-True (@($parsed['src/Covered.cs']).Count -eq 2) 'a deletion must not contaminate the prior file'

    $parsed = ConvertTo-ChangedRanges -DiffLines @() -ChangedFiles @('src/Renamed.cs', 'assets/blob.bin')
    Assert-True ($parsed.ContainsKey('src/Renamed.cs')) 'a rename-only file must be present without a hunk'
    Assert-True ($parsed.ContainsKey('assets/blob.bin')) 'a binary file must be present without a hunk'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Covered.cs' = @() }
    Assert-True $r.Escalate 'a mapped file without hunks must escalate'

    $badRange = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badRange.files.'src/Covered.cs'[0].r = @(10)
    Assert-Throws { Assert-ShardMap -Map $badRange -Expected $expected } 'an odd range list must be rejected'

    $badIndex = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badIndex.files.'src/Covered.cs'[0].s = 99
    Assert-Throws { Assert-ShardMap -Map $badIndex -Expected $expected } 'an out-of-range shard index must be rejected'

    $badSchema = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badSchema.schemaVersion = 2
    Assert-Throws { Assert-ShardMap -Map $badSchema -Expected $expected } 'an unknown map schema must be rejected'

    $overlap = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $overlap.alwaysRun = @('Alpha')
    Assert-Throws { Assert-ShardMap -Map $overlap -Expected @('Alpha', 'Beta') } `
        'known and always-run shard sets must be disjoint'

    $badLine = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $badLine.files.'src/Covered.cs'[0].r = @(0, 10)
    Assert-Throws { Assert-ShardMap -Map $badLine -Expected $expected } 'non-positive map lines must be rejected'

    # A map naming a shard the manifest dropped (a removal or rename) is still refused.
    Assert-Throws { Assert-ShardMap -Map $map -Expected @('Alpha', 'HeavyNoCoverage') } `
        'a map naming a shard the manifest no longer has must be rejected'

    # A shard added after the map was built is not a reason to distrust the map: it runs on every
    # selection until a map includes it, and every mapped shard stays selective.
    $grown = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $grownExpected = @('Alpha', 'Beta', 'HeavyNoCoverage', 'NewShard')
    try { Assert-ShardMap -Map $grown -Expected $grownExpected } catch { [void] $failures.Add("a map missing only a newly added shard was rejected: $_") }
    $added = @(Add-UnmappedShardsAsAlwaysRun -Map $grown -Expected $grownExpected)
    Assert-True (($added -join ',') -ceq 'NewShard') "the unmapped shard was not identified (got '$($added -join ',')')"
    $r = Select-ImpactedShards -Map $grown -Changed @{ 'src/Covered.cs' = @(12, 14) }
    Assert-True ($r.Shards -contains 'NewShard') 'an unmapped shard was not run'
    Assert-True ($r.Shards -contains 'Alpha') 'a mapped shard reached by the change was not selected'
    Assert-True (-not ($r.Shards -contains 'Beta')) 'an unmapped shard made the mapped selection unselective'
    Assert-True (@($r.Routes | Where-Object { $_ -like 'NewShard <= is not in the coverage map yet*' }).Count -eq 1) `
        'the unmapped shard is not reported as unmapped'
    $r = Select-ImpactedShards -Map $grown -Changed @{ 'docs/readme.md' = @(1, 1) }
    Assert-True (-not ($r.Shards -contains 'NewShard')) 'an unmapped shard ran for a change that needs no validation'
    $again = @(Add-UnmappedShardsAsAlwaysRun -Map $grown -Expected $grownExpected)
    Assert-True ($again.Count -eq 0) 'a second pass re-added an unmapped shard'

    # Position-sliced inventory shards are recognised however the manifest was loaded, and the real
    # manifest's conformance windows and parameter-count shards are all recognised.
    Assert-True (Test-InventoryWindow ([pscustomobject]@{ name = 'W'; env = [pscustomobject]@{ ADNSHAPE_CONF_OFFSET = '5' } })) `
        'a conformance window was not recognised as inventory-sliced'
    Assert-True (Test-InventoryWindow @{ name = 'P'; env = @{ AIDOTNET_PARAMETER_COUNT_SHARD = '0' } }) `
        'a parameter-count shard was not recognised as inventory-sliced'
    Assert-True (Test-InventoryWindow ([pscustomobject]@{ name = 'L'; workload = 'ModelShape' })) `
        'a legacy auxiliary workload was not recognised as inventory-sliced'
    Assert-True (-not (Test-InventoryWindow ([pscustomobject]@{ name = 'O'; env = [pscustomobject]@{ ADNSHAPE_WORKERS = '4' } }))) `
        'an unsliced sweep was treated as inventory-sliced'
    Assert-True (-not (Test-InventoryWindow ([pscustomobject]@{ name = 'T'; filter = 'x' }))) `
        'an ordinary shard was treated as inventory-sliced'
    $realManifest = Join-Path $PSScriptRoot '../../.github/test-shards.yml'
    if ((Test-Path -LiteralPath $realManifest) -and (Get-Command yq -ErrorAction SilentlyContinue)) {
        $realShards = @(& yq -o=json -I=0 '.shard' $realManifest | ConvertFrom-Json)
        $slicedFilters = @($realShards | Where-Object {
            [string] $_.filter -match 'ModelContractConformanceTests|ParameterCountContractTests' })
        $unrecognised = @($slicedFilters | Where-Object { -not (Test-InventoryWindow $_) } | ForEach-Object name)
        Assert-True ($slicedFilters.Count -gt 0 -and $unrecognised.Count -eq 0) `
            "position-sliced manifest shards are not recognised: $($unrecognised -join ', ')"
    }

    # Definition drift: the same content loaded two ways is unchanged; any content edit is not.
    $measuredManifest = @(
        @{ name = 'Alpha'; filter = 'FullyQualifiedName~A'; env = @{ X = '1'; Y = '2' }; timeout = 30 },
        @{ name = 'Beta'; filter = 'FullyQualifiedName~B' }
    )
    $sameManifest = @($measuredManifest | ConvertTo-Json -Depth 5 | ConvertFrom-Json)
    [array]::Reverse($sameManifest)
    Assert-True ((ConvertTo-CanonicalShardJson $measuredManifest[0]) -ceq (ConvertTo-CanonicalShardJson $sameManifest[1])) `
        'a hashtable and its JSON round trip canonicalised differently'
    $redefined = @(Get-RedefinedShards -MapManifest $measuredManifest -Manifest $sameManifest -IndexedShards @('Alpha', 'Beta'))
    Assert-True ($redefined.Count -eq 0) "an unchanged manifest reported redefined shards: $($redefined -join ',')"
    foreach ($edit in @(
        @{ What = 'filter'; Entry = @{ name = 'Alpha'; filter = 'FullyQualifiedName~A|FullyQualifiedName~C'; env = @{ X = '1'; Y = '2' }; timeout = 30 } },
        @{ What = 'nested value'; Entry = @{ name = 'Alpha'; filter = 'FullyQualifiedName~A'; env = @{ X = '1'; Y = '3' }; timeout = 30 } },
        @{ What = 'number'; Entry = @{ name = 'Alpha'; filter = 'FullyQualifiedName~A'; env = @{ X = '1'; Y = '2' }; timeout = 31 } },
        @{ What = 'added key'; Entry = @{ name = 'Alpha'; filter = 'FullyQualifiedName~A'; env = @{ X = '1'; Y = '2' }; timeout = 30; heavy = $true } },
        @{ What = 'case'; Entry = @{ name = 'Alpha'; filter = 'FullyQualifiedName~a'; env = @{ X = '1'; Y = '2' }; timeout = 30 } }
    )) {
        $changed = @(Get-RedefinedShards -MapManifest $measuredManifest -Manifest @($edit.Entry, $measuredManifest[1]) -IndexedShards @('Alpha', 'Beta'))
        Assert-True (($changed -join ',') -ceq 'Alpha') "a $($edit.What) edit was not detected as a redefinition"
    }
    $renamedCase = @(Get-RedefinedShards -MapManifest $measuredManifest `
        -Manifest @(@{ name = 'alpha'; filter = 'FullyQualifiedName~A' }) -IndexedShards @('alpha'))
    Assert-True (($renamedCase -join ',') -ceq 'alpha') 'shard names were matched case-insensitively'
    $unindexed = @(Get-RedefinedShards -MapManifest $measuredManifest `
        -Manifest @(@{ name = 'Beta'; filter = 'changed' }) -IndexedShards @('Alpha'))
    Assert-True ($unindexed.Count -eq 0) 'a shard with no index was reported as redefined'
    $unknownHistory = @(Get-RedefinedShards -MapManifest @() -Manifest $sameManifest -IndexedShards @('Alpha', 'Beta'))
    Assert-True (($unknownHistory -join ',') -ceq 'Alpha,Beta') 'an unknown measured manifest did not redefine every indexed shard'

    $redefinedMap = $map | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $redefinedMap | Add-Member -NotePropertyName redefinedShards -NotePropertyValue @('Beta')
    $r = Select-ImpactedShards -Map $redefinedMap -Changed @{ 'src/Covered.cs' = @(12, 14) }
    Assert-True ((@($r.Shards) -join ',') -ceq 'Alpha,Beta,HeavyNoCoverage') 'a redefined shard was not made mandatory'
    Assert-True (@($r.Routes | Where-Object { $_ -like 'Beta <= changed its manifest definition*' }).Count -eq 1) `
        'a redefined shard is not reported as redefined'
    $r = Select-ImpactedShards -Map $redefinedMap -Changed @{ 'docs/readme.md' = @(1, 1) }
    Assert-True (@($r.Shards).Count -eq 0 -and -not $r.RequiresValidation) 'a redefined shard ran for a change that needs no validation'

    $commaMap = @{
        schemaVersion = 1; sha = 'abcdefabcdefabcdefabcdefabcdefabcdefabcd'; knownShards = @('Comma, Shard'); alwaysRun = @()
        files = @{ 'src/Comma.cs' = @( @{ s = 0; r = @(1, 1) } ) }
    } | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    try { Assert-ShardMap -Map $commaMap -Expected @('Comma, Shard') } catch {
        [void] $failures.Add("a comma-containing shard name was rejected: $_")
    }
    $r = Select-ImpactedShards -Map $commaMap -Changed @{ 'src/Comma.cs' = @(1, 1) }
    Assert-True ($r.Shards -contains 'Comma, Shard') 'a comma-containing shard name must stay intact'

    # 14. Shared-infrastructure matching. File entries match by BASENAME at any depth - MSBuild
    #     and NuGet apply these files per-directory, and src/Directory.Build.props exists in this
    #     repo - while look-alike names must not match, and directories stay prefix-matched.
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/Directory.Build.props' = @(1, 2) }
    Assert-True $r.Escalate 'a NESTED Directory.Build.props must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'src/sub/nuget.config' = @(1, 2) }
    Assert-True $r.Escalate 'a nested nuget.config must escalate'
    $r = Select-ImpactedShards -Map $map -Changed @{ 'Directory.Packages.props.backup' = @(1, 2) }
    Assert-True (-not ($r.Reasons -join ';').Contains('shared infrastructure')) `
        'a look-alike suffix must not match shared infrastructure'
    $r = Select-ImpactedShards -Map $map -Changed @{ '.github/workflows/docs.yml' = @(1, 2) }
    Assert-True (-not $r.Escalate -and -not $r.RequiresValidation) `
        'an independent YAML workflow must be classified as non-runtime'

    # 15. Hunk BODY content must be inert. A removed line whose text begins with '-- ' renders as
    #     '--- ...', byte-identical to an old-file header; the pre-fix parser nulled $current on it
    #     and silently dropped every later hunk of the file - under-selection with no escalation.
    #     This diff is verbatim real-git output for: delete the line '-- remove me', edit line 81.
    $diff = @(
        'diff --git a/F.cs b/F.cs',
        'index 9329992..2c35db3 100644',
        '--- a/F.cs',
        '+++ b/F.cs',
        '@@ -50 +49,0 @@ line 49',
        '--- remove me',
        '@@ -81 +80 @@ line 80',
        '-line 81',
        '+line 80 EDITED'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True ($r.ContainsKey('F.cs')) 'the file must be reported'
    Assert-True (@($r['F.cs']) -contains 81) 'the hunk AFTER the forged header line must survive'
    Assert-True (-not $r.ContainsKey('remove me')) 'body content must never become a path'

    # 16. And a forged header inside an ADDED body line must not smuggle a file in.
    $diff = @(
        '--- a/G.cs',
        '+++ b/G.cs',
        '@@ -5,0 +6,2 @@',
        '+--- a/EVIL.cs',
        '++++ b/EVIL.cs',
        '@@ -30 +32 @@',
        '-x',
        '+y'
    )
    $r = ConvertTo-ChangedRanges -DiffLines $diff
    Assert-True (-not $r.ContainsKey('EVIL.cs')) 'added body content must never become a path'
    Assert-True (@($r['G.cs']) -contains 30) 'the following real hunk still lands on the right file'

    # 17. PRECISE FALLBACKS. A changed range no shard executes used to select every shard executing the
    #     file: on #2553 (LayerBase, NeuralNetworkBase, one new layer) 37 of 114 shards came from
    #     nothing else. Each case below edits one construct of a fixture in which every member is
    #     executed by its own shard and holds a line no shard executes, so a selection wider than
    #     the rule allows names the extra shards. The hunks are real 'git diff -U0' output.
    $preciseLines = [string[]] @(
        'using System;',
        'namespace Fixture',
        '{',
        '    /// <summary>Counts.</summary>',
        '    public class Hub : HubBase',
        '    {',
        '        private int _count;',
        '        private int _seed = 3;',
        '        public const int Limit = 1;',
        '        public int Auto { get; set; }',
        '',
        '        // Runs first.',
        '        public Hub(int count) : base(count)',
        '        {',
        '            _count = count;',
        '            if (count < 0) { _count = 0; }',
        '        }',
        '',
        '        public int Value',
        '        {',
        '            get',
        '            {',
        '                if (_count < 0) { return 0; }',
        '                return _count;',
        '            }',
        '        }',
        '',
        '        public int Twice =>',
        '            _count < 0 ? 0',
        '            : _count * 2;',
        '',
        '        public int Run(int x)',
        '        {',
        '            Func<int, int> next = y =>',
        '            {',
        '                if (y < 0) { return 0; }',
        '                return y + 1;',
        '            };',
        '            int Local(int z)',
        '            {',
        '                if (z < 0) { return 0; }',
        '                return z;',
        '            }',
        '            if (x < 0) { return -1; }',
        '            return next(Local(x));',
        '        }',
        '',
        '        public int Unused(int x)',
        '        {',
        '            return x;',
        '        }',
        '',
        '        public static Hub operator +(Hub a, Hub b)',
        '        {',
        '            if (a == null) { return b; }',
        '            return a;',
        '        }',
        '',
        '        public int this[int i]',
        '        {',
        '            get',
        '            {',
        '                if (i < 0) { return 0; }',
        '                return i;',
        '            }',
        '        }',
        '#if FIXTURE_SYMBOL',
        '        private void Disabled() { }',
        '#endif',
        '    }',
        '',
        '    public class Side',
        '    {',
        '        private int _side;',
        '        public int Get() { return _side; }',
        '    }',
        '',
        '    public class Idle',
        '    {',
        '        private int _idle;',
        '    }',
        '',
        '    public enum Mode { One, Two }',
        '}'
    )
    # Shard -> the lines it executes. Nothing executes Unused, Idle, or any 'if (...)' guard line.
    $preciseOwners = [ordered]@{
        Ctor   = @('_count = count;')
        Access = @('return _count;')
        Arrow  = @(': _count * 2;')
        Run    = @('return y + 1;', 'return z;', 'return next(Local(x));')
        Oper   = @('return a;')
        Index  = @('return i;')
        Side   = @('public int Get() { return _side; }')
    }
    $precisePath = 'src/Area/Hub.cs'
    $hubShards = 'Access,Arrow,Ctor,Index,Oper,Run'
    $fileShards = 'Access,Arrow,Ctor,Index,Oper,Run,Side'
    function Get-PreciseLine([string[]] $Lines, [string] $Text) {
        $found = @(for ($i = 0; $i -lt $Lines.Count; $i++) { if ($Lines[$i].Contains($Text)) { $i + 1 } })
        if ($found.Count -ne 1) { throw "fixture line '$Text' matched $($found.Count) line(s)" }
        return [int] $found[0]
    }
    function New-PreciseMap([string[]] $Lines) {
        $occurrences = @(for ($s = 0; $s -lt $preciseOwners.Count; $s++) {
            $executed = @(foreach ($text in @($preciseOwners[$s])) { $line = Get-PreciseLine $Lines $text; $line; $line })
            @{ s = $s; r = $executed }
        })
        return @{
            schemaVersion = 1; sha = '0123456789abcdef0123456789abcdef01234567'
            knownShards = @($preciseOwners.Keys); alwaysRun = @('HeavyNoCoverage')
            files = @{ $precisePath = $occurrences }
        } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
    }
    function Edit-PreciseLine([string] $Find, [string] $From, [string] $To, [string[]] $Lines = $preciseLines) {
        $copy = [string[]] $Lines.Clone()
        $index = (Get-PreciseLine $copy $Find) - 1
        if (-not $copy[$index].Contains($From)) { throw "fixture line '$Find' does not contain '$From'" }
        $copy[$index] = $copy[$index].Replace($From, $To)
        return , $copy
    }
    function Add-PreciseLines([string] $Before, [string[]] $Added, [string[]] $Lines = $preciseLines) {
        $index = (Get-PreciseLine $Lines $Before) - 1
        return , ([string[]] (@($Lines[0..($index - 1)]) + @($Added) + @($Lines[$index..($Lines.Count - 1)])))
    }
    $preciseMap = New-PreciseMap $preciseLines
    $preciseRoot = Join-Path ([IO.Path]::GetTempPath()) ("select-shards-precise-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $preciseRoot -Force | Out-Null
    function Get-PreciseHunks([string[]] $Old, [string[]] $New) {
        [IO.File]::WriteAllLines((Join-Path $preciseRoot 'old.cs'), $Old)
        [IO.File]::WriteAllLines((Join-Path $preciseRoot 'new.cs'), $New)
        Push-Location $preciseRoot
        try {
            $diff = @(& git -c core.quotepath=false diff --no-index --no-renames --no-ext-diff -U0 -- old.cs new.cs)
            if ($LASTEXITCODE -ne 1) { throw "the fixture edit produced no diff (git exited $LASTEXITCODE)" }
        }
        finally { Pop-Location }
        $parsed = ConvertTo-DiffHunks -DiffLines $diff
        if ($parsed.Count -ne 1) { throw "the fixture diff named $($parsed.Count) file(s)" }
        return , @(@($parsed.Values)[0])
    }
    function Invoke-PreciseCase {
        # Old is the file before the change, New after it, and MapLines the file as the map measured it:
        # Old itself unless the base branch has edited the file since the map. The coverage map is built
        # from MapLines. Returns the selection with Indexed, the shards other than always-run.
        param(
            [string[]] $New, [string[]] $Old = $preciseLines, [string[]] $MapLines = $Old,
            [bool] $Enabled = $true, [bool] $Swept = $false, [switch] $Unreadable
        )
        $hunks = Get-PreciseHunks $Old $New
        $mapHunks = @()
        if (($MapLines -join "`n") -cne ($Old -join "`n")) { $mapHunks = Get-PreciseHunks $MapLines $Old }
        $ranges = [System.Collections.Generic.List[int]]::new()
        foreach ($hunk in $hunks) { $ranges.AddRange([int[]] (Convert-RangesThroughHunks -Ranges (ConvertTo-OldSideRange -Hunk $hunk) -Hunks $mapHunks)) }
        $oldText = $Old -join "`n"
        $mapText = $MapLines -join "`n"
        $newText = if ($Unreadable) { $null } else { $New -join "`n" }
        $result = Select-ImpactedShards -Map (New-PreciseMap $MapLines) -Changed @{ $precisePath = $ranges } -PreciseFallbacks $Enabled `
            -ChangeDetail @{ $precisePath = (New-ChangeDetail -Base 'base' -MapRevision 'map' -Hunks $hunks -MapHunks $mapHunks -Swept $Swept) } `
            -ReadSource { param($revision, $path) if ($revision -ceq 'HEAD') { $newText } elseif ($revision -ceq 'map') { $mapText } else { $oldText } }.GetNewClosure()
        $result | Add-Member -NotePropertyName Indexed -NotePropertyValue ((@($result.Shards | Where-Object { $_ -cne 'HeavyNoCoverage' }) | Sort-Object) -join ',')
        return $result
    }
    try {
        Assert-True (Initialize-CSharpSyntax) "the C# parser PowerShell ships did not load, so no precise fallback can apply: $script:CSharpSyntaxProblem"

        # 17a. A body: only the shards executing that member, whatever else runs in the file.
        foreach ($case in @(
            @{ Kind = 'constructor'; Find = 'if (count < 0)'; Shard = 'Ctor'; Label = 'Hub.constructor' },
            @{ Kind = 'property accessor'; Find = 'if (_count < 0)'; Shard = 'Access'; Label = 'Hub.Value' },
            @{ Kind = 'expression-bodied property'; Find = '_count < 0 ? 0'; Shard = 'Arrow'; Label = 'Hub.Twice' },
            @{ Kind = 'lambda'; Find = 'if (y < 0)'; Shard = 'Run'; Label = 'Hub.Run' },
            @{ Kind = 'local function'; Find = 'if (z < 0)'; Shard = 'Run'; Label = 'Hub.Run' },
            @{ Kind = 'method'; Find = 'if (x < 0)'; Shard = 'Run'; Label = 'Hub.Run' },
            @{ Kind = 'operator'; Find = 'if (a == null)'; Shard = 'Oper'; Label = 'Hub.operator +' },
            @{ Kind = 'indexer'; Find = 'if (i < 0)'; Shard = 'Index'; Label = 'Hub.this[]' }
        )) {
            $from = if ($case.Find.Contains('null')) { 'null' } else { '< 0' }
            $to = if ($case.Find.Contains('null')) { 'b' } else { '< 1' }
            $r = Invoke-PreciseCase -New (Edit-PreciseLine $case.Find $from $to)
            Assert-True (-not $r.Escalate -and $r.Indexed -ceq $case.Shard) `
                "an unexecuted line in a $($case.Kind) body selected '$($r.Indexed)', not only the shard executing that member ($($case.Shard))"
            Assert-True (@($r.Routes | Where-Object { $_ -like "$($case.Shard) <= executes $([WildcardPattern]::Escape($case.Label)) (lines *) of $precisePath, which holds changed lines * that no shard executes" }).Count -eq 1) `
                "the $($case.Kind) route did not name the member and the changed lines"
            Assert-True ($r.Shards -contains 'HeavyNoCoverage' -and @($r.Untested).Count -eq 0) `
                "the $($case.Kind) case dropped the always-run shard or reported tested code as untested"
            $r = Invoke-PreciseCase -New (Edit-PreciseLine $case.Find $from $to) -Enabled $false
            Assert-True ($r.Indexed -ceq $fileShards) "with the precise fallbacks off, the $($case.Kind) case did not select every shard executing the file"
        }

        # 17b. Comments and spacing select nothing; the always-run shard still runs, so nothing escalates.
        foreach ($case in @(
            @{ What = 'a comment edit'; New = (Edit-PreciseLine '// Runs first.' 'first' 'before anything else') },
            @{ What = 'respacing a statement and adding a trailing comment'; New = (Edit-PreciseLine 'if (x < 0)' 'if (x < 0) { return -1; }' 'if (x < 0)   {  return -1;  } // guard') },
            @{ What = 'blank lines and a comment between members'; New = (Add-PreciseLines 'public int Unused(int x)' @('', '        // Not called.', '')) },
            @{ What = 'removing a comment'; New = [string[]] @($preciseLines | Where-Object { $_ -notlike '*// Runs first.*' }) }
        )) {
            $r = Invoke-PreciseCase -New $case.New
            Assert-True (-not $r.Escalate -and $r.Indexed -ceq '' -and @($r.Untested).Count -eq 0) "$($case.What) selected '$($r.Indexed)'"
            Assert-True (@($r.Files | Where-Object { $_.why -like '*changed only comments or whitespace*' }).Count -eq 1) "$($case.What) was not recorded as inert"
        }

        # 17c. A declaration: every shard executing the enclosing type, and no shard that only
        #      executes another type of the same file.
        foreach ($case in @(
            @{ What = 'an added override'; New = (Edit-PreciseLine 'public int Unused(int x)' 'public int' 'public override int') },
            @{ What = 'an added virtual'; New = (Edit-PreciseLine 'public int Run(int x)' 'public int' 'public virtual int') },
            @{ What = 'an added abstract member'; New = (Add-PreciseLines 'public int Unused(int x)' @('        public abstract int Later(int x);', '')) },
            @{ What = 'an explicit interface signature'; New = (Edit-PreciseLine 'public int Unused(int x)' 'public int Unused' 'int IHub.Unused') },
            @{ What = 'a changed parameter list'; New = (Edit-PreciseLine 'public int Run(int x)' '(int x)' '(int x, int scale = 1)') },
            @{ What = 'a field without an initializer'; New = (Edit-PreciseLine 'private int _count;' 'int' 'long') },
            @{ What = 'a field initializer'; New = (Edit-PreciseLine 'private int _seed = 3;' '3' '4') },
            @{ What = 'a property initializer'; New = (Edit-PreciseLine 'public int Auto { get; set; }' '{ get; set; }' '{ get; set; } = 2;') },
            @{ What = 'an auto-property accessor'; New = (Edit-PreciseLine 'public int Auto { get; set; }' 'set;' 'init;') },
            @{ What = 'an added attribute'; New = (Add-PreciseLines 'public int Run(int x)' @('        [Obsolete]')) },
            @{ What = 'a changed base list'; New = (Edit-PreciseLine 'public class Hub : HubBase' 'HubBase' 'HubBase, IHub') },
            @{ What = 'a constructor initializer'; New = (Edit-PreciseLine 'public Hub(int count) : base(count)' 'base(count)' 'base(count + 1)') },
            @{ What = 'an added method'; New = (Add-PreciseLines 'public int Unused(int x)' @('        public int Added(int x)', '        {', '            return x;', '        }', '')) },
            @{ What = 'a removed method'; New = [string[]] @($preciseLines[0..((Get-PreciseLine $preciseLines 'public int Unused(int x)') - 2)] + $preciseLines[((Get-PreciseLine $preciseLines 'public static Hub operator') - 1)..($preciseLines.Count - 1)]) }
        )) {
            $r = Invoke-PreciseCase -New $case.New
            Assert-True (-not $r.Escalate -and $r.Indexed -ceq $hubShards) `
                "$($case.What) selected '$($r.Indexed)', not exactly the shards executing the enclosing type ($hubShards)"
        }
        $r = Invoke-PreciseCase -New (Edit-PreciseLine 'public class Side' 'public class' 'public sealed class')
        Assert-True ($r.Indexed -ceq 'Side') "a type declaration line selected '$($r.Indexed)', not the shards executing that type"
        Assert-True (@($r.Routes | Where-Object { $_ -like "Side <= executes Side (lines *) of $precisePath, which holds changed lines *" }).Count -eq 1) `
            'the type route did not name the type'

        # 17d. Everything the index cannot bound keeps the file-wide rule: every shard executing the file.
        $partialLines = Edit-PreciseLine 'public class Hub : HubBase' 'public class' 'public partial class'
        $excludedLines = Add-PreciseLines 'public int Run(int x)' @('        [ExcludeFromCodeCoverage]')
        foreach ($case in @(
            @{ What = 'a using'; New = (Edit-PreciseLine 'using System;' 'System' 'System.Linq') },
            @{ What = 'a preprocessor directive'; New = (Edit-PreciseLine '#if FIXTURE_SYMBOL' 'FIXTURE_SYMBOL' 'OTHER_SYMBOL') },
            @{ What = 'code a directive disables'; New = (Edit-PreciseLine 'private void Disabled() { }' '{ }' '{ return; }') },
            @{ What = 'an XML documentation comment'; New = (Edit-PreciseLine '/// <summary>Counts.</summary>' 'Counts' 'Counts things') },
            @{ What = 'a constant'; New = (Edit-PreciseLine 'public const int Limit = 1;' '1' '2') },
            @{ What = 'an enum member'; New = (Edit-PreciseLine 'public enum Mode { One, Two }' 'Two' 'Two, Three') },
            @{ What = 'a type added at namespace level'; New = (Add-PreciseLines 'public class Side' @('    public class Fresh { }', '')) },
            @{ What = 'a namespace-level delegate'; New = (Add-PreciseLines 'public class Side' @('    public delegate int Step(int x);', '')) },
            @{ What = 'a field of a type no shard executes'; New = (Edit-PreciseLine 'private int _idle;' 'int' 'long') },
            @{ What = 'a field of a partial type'; Old = $partialLines; New = (Edit-PreciseLine 'private int _count;' 'int' 'long' $partialLines) },
            @{ What = 'a change that does not parse'; New = (Edit-PreciseLine 'if (x < 0)' 'return -1;' 'return -;') },
            @{ What = 'a body excluded from coverage'; Old = $excludedLines; New = (Edit-PreciseLine 'if (x < 0)' '< 0' '< 1' $excludedLines) },
            @{ What = 'a file that cannot be read at HEAD'; New = (Edit-PreciseLine 'if (x < 0)' '< 0' '< 1'); Unreadable = $true },
            @{ What = "lines the base branch also changed since the map"; New = (Edit-PreciseLine 'if (x < 0)' '< 0' '< 1'); Swept = $true }
        )) {
            $arguments = @{ New = $case.New }
            if ($case.ContainsKey('Old')) { $arguments['Old'] = $case.Old }
            if ($case.ContainsKey('Unreadable')) { $arguments['Unreadable'] = $true }
            if ($case.ContainsKey('Swept')) { $arguments['Swept'] = $true }
            $r = Invoke-PreciseCase @arguments
            Assert-True (-not $r.Escalate -and $r.Indexed -ceq $fileShards -and @($r.Untested).Count -eq 0) `
                "$($case.What) selected '$($r.Indexed)', not every shard executing the file"
            Assert-True (@($r.Routes | Where-Object { $_ -like "Side <= executes $precisePath, whose changed lines * no shard executes" }).Count -eq 1) `
                "$($case.What) did not record the file-wide route"
        }
        # A partial type still bounds its own member bodies.
        $r = Invoke-PreciseCase -Old $partialLines -New (Edit-PreciseLine 'if (x < 0)' '< 0' '< 1' $partialLines)
        Assert-True ($r.Indexed -ceq 'Run') "a body in a partial type selected '$($r.Indexed)', not the member's shards"
        # No parser at all: the same file-wide rule, never an empty or narrower selection.
        $script:CSharpSyntaxAvailable = $false
        try {
            $r = Invoke-PreciseCase -New (Edit-PreciseLine 'if (x < 0)' '< 0' '< 1')
            Assert-True ($r.Indexed -ceq $fileShards) "without a parser an unexecuted range selected '$($r.Indexed)', not every shard executing the file"
        }
        finally { $script:CSharpSyntaxAvailable = $null }
        # No per-hunk detail (a caller that predates it): unchanged behaviour.
        $r = Select-ImpactedShards -Map $preciseMap -Changed @{ $precisePath = @((Get-PreciseLine $preciseLines 'if (x < 0)'), (Get-PreciseLine $preciseLines 'if (x < 0)')) }
        Assert-True ((@($r.Shards | Where-Object { $_ -cne 'HeavyNoCoverage' }) -join ',') -ceq $fileShards) 'a selection without change detail did not keep the file-wide rule'

        # 17e. A member no shard executes is reported, not fanned out.
        $r = Invoke-PreciseCase -New (Edit-PreciseLine 'return x;' 'x;' 'x + 1;')
        Assert-True (-not $r.Escalate -and $r.Indexed -ceq '') "a change inside a member no shard executes selected '$($r.Indexed)'"
        Assert-True (@($r.Untested).Count -eq 1 -and $r.Untested[0].path -ceq $precisePath -and $r.Untested[0].member -ceq 'Hub.Unused' -and
            $r.Untested[0].lines -ceq "$(Get-PreciseLine $preciseLines 'return x;')-$(Get-PreciseLine $preciseLines 'return x;')") `
            'the untested member was not reported with its path, name and changed lines'
        Assert-True (@($r.Files | Where-Object { $_.why -like '*untested code changed*' }).Count -eq 1) 'the file decision did not say untested code changed'
        # Two changed ranges of one untested member are one record naming both.
        $twice = Add-PreciseLines 'return x;' @('            x = x + 0;')
        $twice = [string[]] (@($twice[0..((Get-PreciseLine $twice 'return x;') - 1)]) + @('            // end of Unused', '            x = x + 1;') + @($twice[(Get-PreciseLine $twice 'return x;')..($twice.Count - 1)]))
        $r = Invoke-PreciseCase -New $twice
        Assert-True ($r.Indexed -ceq '' -and @($r.Untested).Count -eq 1 -and $r.Untested[0].member -ceq 'Hub.Unused' -and $r.Untested[0].lines -like '*-*, *-*') `
            "two changed ranges of one untested member were reported as $(@($r.Untested).Count) record(s) with lines '$(@($r.Untested | ForEach-Object { $_.lines }) -join ' | ')'"
        # Beside a covered change the covered one still selects, and the untested one is still reported.
        $both = Edit-PreciseLine 'return _count;' '_count;' '_count + 1;' (Edit-PreciseLine 'return x;' 'x;' 'x + 1;')
        $r = Invoke-PreciseCase -New $both
        Assert-True ($r.Indexed -ceq 'Access' -and @($r.Untested).Count -eq 1) "a covered change beside an untested one selected '$($r.Indexed)' with $(@($r.Untested).Count) untested"

        # 17f. A map older than the pull request's base. The declaration is found in the map's own
        #      version of the file by name, so lines the base branch added, edits it made to the
        #      member, and moving the member all leave the member's shard selected, and the route
        #      reports the member's lines as the map numbers them.
        $mapRunLine = Get-PreciseLine $preciseLines 'public int Run(int x)'
        $unusedFrom = (Get-PreciseLine $preciseLines 'public int Unused(int x)') - 1
        $movedUnused = [string[]] $preciseLines[$unusedFrom..($unusedFrom + 4)]
        $withoutUnused = [string[]] @($preciseLines[0..($unusedFrom - 1)] + $preciseLines[($unusedFrom + 5)..($preciseLines.Count - 1)])
        $addedMember = @('        public int Late(int x)', '        {', '            if (x < 9) { return 9; }', '            return x;', '        }', '')
        foreach ($case in @(
            @{ What = 'lines added above the type'; Base = (Add-PreciseLines 'namespace Fixture' @('// added on the base branch', '// since the map was built')) },
            @{ What = 'an edit to the same member'; Base = (Edit-PreciseLine 'return next(Local(x));' 'Local(x))' 'Local(x)) + 0') },
            @{ What = 'a member moved above it'; Base = (Add-PreciseLines 'public int Value' $movedUnused $withoutUnused) }
        )) {
            $r = Invoke-PreciseCase -MapLines $preciseLines -Old $case.Base -New (Edit-PreciseLine 'if (x < 0)' '< 0' '< 1' $case.Base)
            Assert-True ($r.Indexed -ceq 'Run') "behind a base with $($case.What), an unexecuted line of Run selected '$($r.Indexed)'"
            Assert-True (@($r.Routes | Where-Object { $_ -like "Run <= executes Hub.Run (lines $mapRunLine-*) of *" }).Count -eq 1) `
                "behind a base with $($case.What), the member was not reported in the map's line numbers"
        }
        # A declaration behind a base that only edited a body: still the type's shards.
        $editedBase = Edit-PreciseLine 'return next(Local(x));' 'Local(x))' 'Local(x)) + 0'
        $r = Invoke-PreciseCase -MapLines $preciseLines -Old $editedBase -New (Edit-PreciseLine 'private int _count;' 'int' 'long' $editedBase)
        Assert-True ($r.Indexed -ceq $hubShards) "behind a base that edited a body, a changed field selected '$($r.Indexed)', not the type's shards"
        # What the map's version does not have keeps the file-wide rule: a member the base branch added
        # (its body has no lines in the map), and a type that gained a member (code may have moved in).
        $addedBase = Add-PreciseLines 'public int Unused(int x)' $addedMember
        $r = Invoke-PreciseCase -MapLines $preciseLines -Old $addedBase -New (Edit-PreciseLine 'if (x < 9)' '< 9' '< 8' $addedBase)
        Assert-True ($r.Indexed -ceq $fileShards) "a body the base branch added since the map selected '$($r.Indexed)', not every shard executing the file"
        $r = Invoke-PreciseCase -MapLines $preciseLines -Old $addedBase -New (Edit-PreciseLine 'private int _count;' 'int' 'long' $addedBase)
        Assert-True ($r.Indexed -ceq $fileShards) "a field of a type that gained a member since the map selected '$($r.Indexed)', not every shard executing the file"

        # 17g. The kill switch, and a new source file's path for the type-level stage.
        $savedSwitch = $env:TEST_IMPACT_PRECISE_FALLBACKS
        try {
            foreach ($value in @('0', 'false', 'OFF', 'no', ' disabled ')) {
                $env:TEST_IMPACT_PRECISE_FALLBACKS = $value
                Assert-True (-not (Test-PreciseFallbacksEnabled)) "TEST_IMPACT_PRECISE_FALLBACKS='$value' did not turn the precise fallbacks off"
            }
            foreach ($value in @('', '1', 'true', 'on')) {
                $env:TEST_IMPACT_PRECISE_FALLBACKS = $value
                Assert-True (Test-PreciseFallbacksEnabled) "TEST_IMPACT_PRECISE_FALLBACKS='$value' turned the precise fallbacks off"
            }
        }
        finally { $env:TEST_IMPACT_PRECISE_FALLBACKS = $savedSwitch }
        $r = Select-ImpactedShards -Map $preciseMap -Changed @{ 'src/Area/Brand.cs' = @(1, 9); $precisePath = @(1, 1) }
        Assert-True ((@($r.NewSourcePaths) -join ',') -ceq 'src/Area/Brand.cs') 'an unmapped source routed to its directory was not reported as a new source path'
        Assert-True ((@($r.Shards | Where-Object { $_ -cne 'HeavyNoCoverage' }) -join ',') -ceq $fileShards) 'a new source file no longer selects its directory''s owners'
    }
    catch { [void] $failures.Add("the precise-fallback cases could not run: $($_.Exception.Message) at line $($_.InvocationInfo.ScriptLineNumber)") }
    finally { Remove-Item -LiteralPath $preciseRoot -Recurse -Force -ErrorAction SilentlyContinue }

    # 18. ROLLING MAP. One ordinary map per measured commit; each layer is selected for with the change in
    # that layer's own line numbers, and only shard names are merged. Test-TestImpactEndToEnd.ps1 drives the
    # same cases through real commits, where the translation itself is under test.
    try {
        function New-TestLayer([string] $Sha, [string[]] $Known, [hashtable] $Files) {
            return (@{ sha = $Sha; map = @{ schemaVersion = 1; sha = $Sha; knownShards = @($Known); alwaysRun = @(); files = $Files } } |
                ConvertTo-Json -Depth 9 -Compress | ConvertFrom-Json)
        }
        function Invoke-RollingCase {
            # The orchestration's per-layer loop without git: Changed and Settled are given per layer commit.
            param([object[]] $Layers, [hashtable] $ChangedByLayer, [hashtable] $SettledByLayer = @{}, [string[]] $AlwaysRun = @(),
                  [hashtable] $TestRoutes = @{}, [hashtable] $RedefinedByLayer = @{}, [switch] $Scoped, [string[]] $CurrentPaths = @())
            $context = New-RollingSelectionContext -Layers $Layers
            $allKnown = @($Layers | ForEach-Object { @($_.map.knownShards) } | ForEach-Object { [string] $_ })
            $results = foreach ($layer in $Layers) {
                $layerView = New-RollingLayerView -Layer $layer -AllKnown $allKnown -AlwaysRun $AlwaysRun
                if ($RedefinedByLayer.ContainsKey([string] $layer.sha)) {
                    $layerView.View | Add-Member -NotePropertyName redefinedShards -NotePropertyValue @($RedefinedByLayer[[string] $layer.sha])
                }
                $settled = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
                foreach ($path in @($SettledByLayer[[string] $layer.sha])) { if ($path) { [void] $settled.Add($path) } }
                $layerContext = [pscustomobject]@{
                    Sha = [string] $layer.sha; Home = $layerView.Home; SettledPaths = $settled; MappedPaths = $context.MappedPaths
                    DirectoryOwners = $context.DirectoryOwners; DirectoryFileCount = $context.DirectoryFileCount
                }
                $changed = $ChangedByLayer[[string] $layer.sha]
                $current = if ($Scoped) { $CurrentPaths } else { @($changed.Keys) }
                [pscustomobject]@{
                    Sha = [string] $layer.sha; Home = $layerView.Home; Index = $layer.map.files
                    Selection = Select-ImpactedShards -Map $layerView.View -Changed $changed -CurrentPaths $current `
                        -ScopeToCurrentPaths:$Scoped -TestRoutes $TestRoutes -PreciseFallbacks $false -Rolling $layerContext
                }
            }
            return Merge-RollingLayerSelections -Layers @($results) -Mandatory $AlwaysRun
        }
        $shaOne = '1' * 40
        $shaTwo = '2' * 40

        # 18a. A rolling map with ONE layer is the map it was built from: same shards, same escalation, same
        # reasons and routes, for every regime the single-map selection has.
        $single = New-TestLayer $map.sha @($map.knownShards) @{
            'src/Covered.cs' = @(@{ s = 0; r = @(10, 20) }, @{ s = 1; r = @(100, 110) })
            'src/SingleRange.cs' = @(, @{ s = 0; r = @(5, 6) })
        }
        $bare = @{ schemaVersion = 1; sha = $map.sha; knownShards = @('Alpha', 'Beta'); alwaysRun = @(); files = @{
            'src/Covered.cs' = @(@{ s = 0; r = @(10, 20) }, @{ s = 1; r = @(100, 110) }) } } | ConvertTo-Json -Depth 8 -Compress | ConvertFrom-Json
        $bareLayer = New-TestLayer $map.sha @('Alpha', 'Beta') @{ 'src/Covered.cs' = @(@{ s = 0; r = @(10, 20) }, @{ s = 1; r = @(100, 110) }) }
        foreach ($case in @(
            @{ What = 'a mapped hit'; Changed = @{ 'src/Covered.cs' = @(12, 12) } },
            @{ What = 'a range no shard executes'; Changed = @{ 'src/Covered.cs' = @(50, 51) } },
            @{ What = 'two files'; Changed = @{ 'src/Covered.cs' = @(105, 105); 'src/SingleRange.cs' = @(5, 5) } },
            @{ What = 'an unmapped source with mapped neighbours'; Changed = @{ 'src/Area/Sub/New.cs' = @(1, 4); 'src/Covered.cs' = @(1, 1) } },
            @{ What = 'an orphan source'; Changed = @{ 'src/Nowhere/Deep/New.cs' = @(1, 4) } },
            @{ What = 'an unmapped project file'; Changed = @{ 'src/Thing.csproj' = @(1, 1) } },
            @{ What = 'no change at all'; Changed = @{} },
            @{ What = 'documentation only'; Changed = @{ 'docs/Guide.md' = @(1, 1) } },
            @{ What = 'validation infrastructure'; Changed = @{ 'Directory.Build.props' = @(1, 1); 'src/Covered.cs' = @(12, 12) } },
            @{ What = 'a routed test source'; Changed = @{ 'tests/T/FooTests.cs' = @(3, 3) }
               Routes = @{ 'tests/T/FooTests.cs' = [pscustomobject]@{ Routable = $true; Shards = @('Beta'); Why = 'its class matches the filter' } } },
            @{ What = 'an unroutable test source'; Changed = @{ 'tests/T/FooTests.cs' = @(3, 3) }
               Routes = @{ 'tests/T/FooTests.cs' = [pscustomobject]@{ Routable = $false; Shards = @(); Why = 'no filter decides it' } } },
            @{ What = 'no always-run shard and no hit'; Changed = @{ 'src/Covered.cs' = @(50, 51) }; Bare = $true },
            @{ What = 'no always-run shard and an empty selection'; Changed = @{ 'tests/T/FooTests.cs' = @(3, 3) }; Bare = $true
               Routes = @{ 'tests/T/FooTests.cs' = [pscustomobject]@{ Routable = $false; Unassigned = $true; Shards = @(); Why = 'no shard runs it' } } }
        )) {
            $routesGiven = if ($case.ContainsKey('Routes')) { $case.Routes } else { @{} }
            $useBare = $case.ContainsKey('Bare')
            $one = Select-ImpactedShards -Map $(if ($useBare) { $bare } else { $map }) -Changed $case.Changed -TestRoutes $routesGiven -PreciseFallbacks $false
            $rolled = Invoke-RollingCase -Layers @($(if ($useBare) { $bareLayer } else { $single })) -ChangedByLayer @{ $map.sha = $case.Changed } `
                -AlwaysRun $(if ($useBare) { @() } else { @($map.alwaysRun) }) -TestRoutes $routesGiven
            Assert-True ([bool] $one.Escalate -eq [bool] $rolled.Escalate) "one-layer rolling map, $($case.What): escalate $($rolled.Escalate), the single map says $($one.Escalate)"
            Assert-True ([bool] $one.RequiresValidation -eq [bool] $rolled.RequiresValidation) "one-layer rolling map, $($case.What): requiresValidation differs from the single map"
            Assert-True ((@($one.Shards) -join '|') -ceq (@($rolled.Shards) -join '|')) "one-layer rolling map, $($case.What): selected '$(@($rolled.Shards) -join ',')', the single map selects '$(@($one.Shards) -join ',')'"
            Assert-True ((@($one.Reasons) -join '|') -ceq (@($rolled.Reasons) -join '|')) "one-layer rolling map, $($case.What): reasons '$(@($rolled.Reasons) -join '|')' differ from the single map's '$(@($one.Reasons) -join '|')'"
            Assert-True ((@($one.Routes | Sort-Object) -join '|') -ceq (@($rolled.Routes | Sort-Object) -join '|')) "one-layer rolling map, $($case.What): routes differ from the single map's"
        }

        # 18b. THE MIXED-COORDINATE CASE. F.cs is 100 lines at commit 1. Commit 2 inserts 20 lines at its top.
        #   measured at 1:  Early executes 40-42,  Shifted executes 20-22        (commit 1's numbering)
        #   measured at 2:  Late  executes 40-42,  Moved   executes 60-62        (commit 2's numbering)
        # so Shifted and Late execute the same code, as do Early and Moved. A pull request on commit 2 edits
        # line 41 there, which is line 21 at commit 1. Exactly Shifted and Late execute it.
        $layerOne = New-TestLayer $shaOne @('Early', 'Shifted') @{ 'src/F.cs' = @(@{ s = 0; r = @(40, 42) }, @{ s = 1; r = @(20, 22) }) }
        $layerTwo = New-TestLayer $shaTwo @('Late', 'Moved') @{ 'src/F.cs' = @(@{ s = 0; r = @(40, 42) }, @{ s = 1; r = @(60, 62) }) }
        $insertion = ConvertTo-DiffHunks -DiffLines @('diff --git a/src/F.cs b/src/F.cs', '--- a/src/F.cs', '+++ b/src/F.cs', '@@ -0,0 +1,20 @@')
        $atTwo = [int[]] @(41, 41)
        $atOne = [int[]] (Convert-RangesThroughHunks -Ranges $atTwo -Hunks @($insertion['src/F.cs']))
        Assert-True (($atOne -join ',') -ceq '21,21') "line 41 after a 20-line insertion carried back to '$($atOne -join ',')', not line 21"
        $right = Invoke-RollingCase -Layers @($layerOne, $layerTwo) -ChangedByLayer @{ $shaOne = @{ 'src/F.cs' = $atOne }; $shaTwo = @{ 'src/F.cs' = $atTwo } }
        Assert-True ((@($right.Shards) -join ',') -ceq 'Late,Shifted') "translated per measured commit, the edit selected '$(@($right.Shards) -join ',')', not exactly Late and Shifted"
        Assert-True (-not $right.Escalate) 'the two-commit case escalated'
        # The mistake, both ways round. With the newer numbering against the older layer the edit lands on
        # Early's lines and Shifted - which really executes it - is never selected: a missed shard. This is
        # why no range is ever read against a commit it was not translated to.
        $newerOnly = Invoke-RollingCase -Layers @($layerOne, $layerTwo) -ChangedByLayer @{ $shaOne = @{ 'src/F.cs' = $atTwo }; $shaTwo = @{ 'src/F.cs' = $atTwo } }
        Assert-True ((@($newerOnly.Shards) -join ',') -ceq 'Early,Late') "the newer commit's numbers against both layers selected '$(@($newerOnly.Shards) -join ',')' (expected the wrong answer Early,Late)"
        Assert-True (@($newerOnly.Shards) -cnotcontains 'Shifted') 'the mixed-coordinate mistake no longer misses a shard; this case has stopped guarding it'
        $olderOnly = Invoke-RollingCase -Layers @($layerOne, $layerTwo) -ChangedByLayer @{ $shaOne = @{ 'src/F.cs' = $atOne }; $shaTwo = @{ 'src/F.cs' = $atOne } }
        Assert-True ((@($olderOnly.Shards) -join ',') -cne 'Late,Shifted') "the older commit's numbers against both layers still selected exactly the right shards; this case has stopped guarding it"
        Assert-True (@($right.Layers).Count -eq 2 -and $right.Layers[0].selected -eq 1 -and $right.Layers[1].selected -eq 1) 'the per-commit summary does not show one shard selected at each commit'

        # 18c. A file added AFTER a layer's commit. Shards measured before it existed cannot be said not to
        # execute it, so the unmapped-file rule applies to them until they are measured again; shards measured
        # with it present are selected by their own coverage. A file that already existed there, unchanged up
        # to the change's base, is different: those shards ran with exactly that content and executed none of it.
        $before = New-TestLayer $shaOne @('OldA', 'OldB', 'OldFar') @{
            'src/Area/Old.cs' = @(@{ s = 0; r = @(1, 9) }, @{ s = 1; r = @(3, 4) })
            'src/Elsewhere/Far.cs' = @(, @{ s = 2; r = @(1, 9) })
        }
        $after = New-TestLayer $shaTwo @('NewHit', 'NewMiss') @{
            'src/Area/New.cs' = @(, @{ s = 0; r = @(1, 10) })
            'src/Area/Old.cs' = @(, @{ s = 1; r = @(1, 9) })
        }
        $edit = @{ $shaOne = @{ 'src/Area/New.cs' = @(5, 5) }; $shaTwo = @{ 'src/Area/New.cs' = @(5, 5) } }
        $added = Invoke-RollingCase -Layers @($before, $after) -ChangedByLayer $edit
        Assert-True ((@($added.Shards) -join ',') -ceq 'NewHit,OldA,OldB') "a file added after one layer's commit selected '$(@($added.Shards) -join ',')', not its measured shard plus the older layer's directory owners"
        Assert-True (-not $added.Escalate) 'a file added after one layer''s commit escalated although it has mapped neighbours'
        $present = Invoke-RollingCase -Layers @($before, $after) -ChangedByLayer $edit -SettledByLayer @{ $shaOne = @('src/Area/New.cs') }
        Assert-True ((@($present.Shards) -join ',') -ceq 'NewHit') "a file that existed unchanged, unexecuted, at the older layer's commit selected '$(@($present.Shards) -join ',')', not only the shard that executes it"
        # No mapped neighbour at all for the older layer's purposes: that layer cannot bound the change, so
        # exactly its shards run, and the layer measured with the file stays selective.
        $lonely = New-TestLayer $shaTwo @('NewHit', 'NewMiss') @{
            'src/Fresh/Deep/New.cs' = @(, @{ s = 0; r = @(1, 10) })
            'src/Area/Old.cs' = @(, @{ s = 1; r = @(1, 9) })
        }
        $partial = Invoke-RollingCase -Layers @($before, $lonely) -ChangedByLayer @{ $shaOne = @{ 'src/Fresh/Deep/New.cs' = @(5, 5) }; $shaTwo = @{ 'src/Fresh/Deep/New.cs' = @(5, 5) } }
        Assert-True ((@($partial.Shards) -join ',') -ceq 'NewHit,OldA,OldB,OldFar') "an unbounded change for one layer selected '$(@($partial.Shards) -join ',')', not that layer's every shard plus the other layer's own selection"
        Assert-True (-not $partial.Escalate) 'one unbounded layer escalated the whole matrix'
        Assert-True (@($partial.Routes | Where-Object { $_ -like 'OldFar <= was measured at 11111111*' }).Count -eq 1) 'a shard run because its layer could not bound the change has no route saying so'
        # Unbounded for every layer is the whole matrix, as with a single map.
        $everywhere = Invoke-RollingCase -Layers @($before, $after) -ChangedByLayer @{ $shaOne = @{ 'src/Nowhere/Deep/X.cs' = @(1, 1) }; $shaTwo = @{ 'src/Nowhere/Deep/X.cs' = @(1, 1) } }
        Assert-True ([bool] $everywhere.Escalate) 'a change no layer can bound did not escalate'

        # 18d. A shard redefined since ITS layer was measured runs; the other layers are unaffected by that.
        $redefinedCase = Invoke-RollingCase -Layers @($layerOne, $layerTwo) -RedefinedByLayer @{ $shaTwo = @('Moved') } `
            -ChangedByLayer @{ $shaOne = @{ 'src/F.cs' = $atOne }; $shaTwo = @{ 'src/F.cs' = $atTwo } }
        Assert-True ((@($redefinedCase.Shards) -join ',') -ceq 'Late,Moved,Shifted') "a shard redefined after its digest selected '$(@($redefinedCase.Shards) -join ',')', not the redefined shard as well"

        # 18e. Always-run shards come through whichever layer names them; a layer's opinion about ANOTHER
        # layer's shard is not kept (a test route names shards of both layers and each layer keeps its own).
        $routed = @{ 'tests/T/FooTests.cs' = [pscustomobject]@{ Routable = $true; Shards = @('Early', 'Moved'); Why = 'its class matches the filter' } }
        $mixed = Invoke-RollingCase -Layers @($layerOne, $layerTwo) -AlwaysRun @('Heavy') -TestRoutes $routed `
            -ChangedByLayer @{ $shaOne = @{ 'tests/T/FooTests.cs' = @(1, 1) }; $shaTwo = @{ 'tests/T/FooTests.cs' = @(1, 1) } }
        Assert-True ((@($mixed.Shards) -join ',') -ceq 'Early,Heavy,Moved') "a test route across two layers selected '$(@($mixed.Shards) -join ',')'"
        $foreign = Merge-RollingLayerSelections -Mandatory @('Heavy') -Layers @(
            [pscustomobject]@{ Sha = $shaOne; Home = [System.Collections.Generic.HashSet[string]]::new([string[]] @('Early'), [StringComparer]::Ordinal)
                Selection = [pscustomobject]@{ Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @('Heavy', 'Late'); Routes = @('Late <= x', 'Heavy <= is always run') } },
            [pscustomobject]@{ Sha = $shaTwo; Home = [System.Collections.Generic.HashSet[string]]::new([string[]] @('Late'), [StringComparer]::Ordinal)
                Selection = [pscustomobject]@{ Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @('Heavy'); Routes = @('Heavy <= is always run') } })
        Assert-True ((@($foreign.Shards) -join ',') -ceq 'Heavy') "a layer's selection of another layer's shard was kept ('$(@($foreign.Shards) -join ',')')"
        Assert-True (@($foreign.Routes).Count -eq 1) 'a route for another layer''s shard was kept'

        # A member is reported untested only when no layer's shards execute it.
        $untestedHere = [pscustomobject]@{ path = 'src/F.cs'; member = 'F.M'; lines = '21-21'; memberLines = '21-21' }
        $bothIndex = Merge-RollingLayerSelections -Layers @(
            [pscustomobject]@{ Sha = $shaOne; Home = [System.Collections.Generic.HashSet[string]]::new([string[]] @('Early'), [StringComparer]::Ordinal); Index = $layerOne.map.files
                Selection = [pscustomobject]@{ Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @(); Routes = @(); Untested = @($untestedHere) } },
            [pscustomobject]@{ Sha = $shaTwo; Home = [System.Collections.Generic.HashSet[string]]::new([string[]] @('Late'), [StringComparer]::Ordinal); Index = $layerTwo.map.files
                Selection = [pscustomobject]@{ Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @('Late'); Routes = @('Late <= x'); Untested = @() } })
        Assert-True (@($bothIndex.Untested).Count -eq 0) 'a member one layer''s shards execute was reported as untested because another layer''s do not'
        $neither = Merge-RollingLayerSelections -Layers @(
            [pscustomobject]@{ Sha = $shaOne; Home = [System.Collections.Generic.HashSet[string]]::new([string[]] @('Early'), [StringComparer]::Ordinal); Index = $layerOne.map.files
                Selection = [pscustomobject]@{ Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @('Early'); Routes = @('Early <= x'); Untested = @($untestedHere) } },
            [pscustomobject]@{ Sha = $shaTwo; Home = [System.Collections.Generic.HashSet[string]]::new([string[]] @('Late'), [StringComparer]::Ordinal); Index = $layerTwo.map.files
                Selection = [pscustomobject]@{ Escalate = $false; RequiresValidation = $true; Reasons = @(); Shards = @(); Routes = @(); Untested = @([pscustomobject]@{ path = 'src/F.cs'; member = 'F.M'; lines = '41-41'; memberLines = '41-41' }) } })
        Assert-True (@($neither.Untested).Count -eq 1) 'a member no layer''s shards execute was not reported as untested exactly once'

        # 18f. Nothing selected by any layer for a change that needs validating: the single map's rule.
        $nothing = Invoke-RollingCase -Layers @($before, $after) -ChangedByLayer @{ $shaOne = @{}; $shaTwo = @{} }
        Assert-True ([bool] $nothing.Escalate -and (@($nothing.Reasons) -join '|') -ceq 'changed path set was empty') 'an empty change against every layer did not escalate as the single map does'
        $oneSided = Invoke-RollingCase -Layers @($before, $after) -ChangedByLayer @{ $shaOne = @{ 'src/Area/Old.cs' = @(3, 3) }; $shaTwo = @{} }
        Assert-True (-not $oneSided.Escalate -and (@($oneSided.Shards) -join ',') -ceq 'OldA,OldB') "a change only one layer sees selected '$(@($oneSided.Shards) -join ',')' or escalated"

        # 18g. The map kinds are told apart from the head of the file.
        $sniffRoot = Join-Path ([IO.Path]::GetTempPath()) ("rolling-sniff-" + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $sniffRoot | Out-Null
        try {
            '{"schemaVersion":1,"kind":"rolling","anchor":{}}' | Set-Content -LiteralPath (Join-Path $sniffRoot 'r.json') -Encoding utf8
            "{`n  `"schemaVersion`": 1,`n  `"kind`": `"rolling`"`n}" | Set-Content -LiteralPath (Join-Path $sniffRoot 'p.json') -Encoding utf8
            $map | ConvertTo-Json -Depth 8 -Compress | Set-Content -LiteralPath (Join-Path $sniffRoot 'm.json') -Encoding utf8
            Assert-True (Test-RollingMapFile -Path (Join-Path $sniffRoot 'r.json')) 'a rolling map was not recognised'
            Assert-True (Test-RollingMapFile -Path (Join-Path $sniffRoot 'p.json')) 'an indented rolling map was not recognised'
            Assert-True (-not (Test-RollingMapFile -Path (Join-Path $sniffRoot 'm.json'))) 'an ordinary map was read as a rolling map'
            # A missing map is the ordinary path's to report (it escalates); the sniff must not throw first.
            Assert-True (-not (Test-RollingMapFile -Path (Join-Path $sniffRoot 'absent.json'))) 'a missing map file was read as a rolling map'
            Assert-True (-not (Test-RollingMapFile -Path $sniffRoot)) 'a directory was read as a rolling map'
        }
        finally { Remove-Item -LiteralPath $sniffRoot -Recurse -Force -ErrorAction SilentlyContinue }
    }
    catch { [void] $failures.Add("the rolling-map cases could not run: $($_.Exception.Message) at line $($_.InvocationInfo.ScriptLineNumber)") }

    if ($failures.Count -gt 0) {
        Write-Host 'Select-Shards self-test FAILED:'
        foreach ($failure in $failures) { Write-Host "  - $failure" }
        exit 1
    }
    Write-Host 'Select-Shards self-test passed.'
    exit 0
}

# ---------------------------------------------------------------- selection

if ($ClassifyOnly) {
    $requiresValidation = $true
    $reason = 'classification-failed'
    $changedFiles = @()
    # Per path, so the Build job's test-level selection can leave out paths that cannot affect any test.
    $pathImpacts = [ordered]@{}
    try {
        if ($PullRequestHeadSha -and $BaseSha) { throw 'pass PullRequestHeadSha or BaseSha, not both' }
        if ($PullRequestHeadSha) { $BaseSha = Resolve-PullRequestBase -PullRequestHeadSha $PullRequestHeadSha }
        if (-not $BaseSha) { throw 'classification needs PullRequestHeadSha or BaseSha' }
        & git cat-file -e "$BaseSha^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) { throw "base commit '$BaseSha' is not present in this checkout" }
        # A source-to-Markdown rename must report both the deleted source path and added Markdown
        # path; otherwise looking only at the destination could incorrectly authorize no CI.
        $changedFiles = @(& git -c core.quotepath=false diff --no-renames --name-only $BaseSha HEAD --)
        if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$BaseSha' failed" }
        $changedFiles = @($changedFiles | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($changedFiles.Count -eq 0) {
            $reason = 'changed-path-set-empty'
        }
        else {
            # No shard manifest here, so only the build-only outcome can be established; a bump that reaches
            # test projects stays full validation for the test-level stage, which then narrows nothing.
            if ($changedFiles -ccontains $script:PackagePropsPath) {
                $script:PackagePropsVerdict = Resolve-PackagePropsVerdict -BaseSha $BaseSha
            }
            $reviewed = [pscustomobject]@{ Paths = @(); Shards = @() }
            if (Test-Path "$PSScriptRoot/CiPolicyImpact.ps1") {
                . "$PSScriptRoot/CiPolicyImpact.ps1"
                $reviewed = Get-ReviewedCiPolicyImpact -BaseSha $BaseSha -Paths $changedFiles
            }
            $requiresValidation = [bool] @(
                $changedFiles | Where-Object {
                    $_ -cnotin $reviewed.Paths -and
                    (Get-ChangedPathImpact -Path ([string] $_)) -ne [ChangedPathImpact]::NonRuntime
                }
            ).Count -or $reviewed.Shards.Count -gt 0
            $reason = $(if ($requiresValidation) { 'runtime-or-unknown' } else { 'non-runtime-only' })
            foreach ($path in $changedFiles) {
                $pathImpacts[[string] $path] = if ([string] $path -cin $reviewed.Paths) { 'ReviewedControl' }
                    else { [string] (Get-ChangedPathImpact -Path ([string] $path)) }
            }
        }
    }
    catch {
        Write-Host "::warning::path classification failed, so runtime validation remains required: $($_.Exception.Message)"
    }

    $result = [pscustomobject]@{
        requiresValidation = $requiresValidation
        reason = $reason
        baseSha = [string] $BaseSha
        changedPaths = @($changedFiles)
        pathImpacts = $pathImpacts
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

function Exit-Escalated {
    param([Parameter(Mandatory)] [string] $Reason, [string] $Message)

    if ($Message) { Write-Host "::warning::$Message" }
    # Same shape as a successful selection, routes included: consumers read it under StrictMode.
    # requiresShards is true here on purpose: an escalation is the fail-closed path, so it must
    # never be the thing that talks the workflow out of running the matrix.
    $result = [pscustomobject]@{
        escalate = $true; requiresValidation = $true; requiresShards = $true
        reason = $Reason; reasons = @(); routes = @(); shards = @()
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    exit 0
}

if (-not (Test-Path -LiteralPath $MapFile)) {
    Exit-Escalated -Reason 'map-missing' -Message "no shard map at $MapFile - running the full matrix"
}

function Invoke-RollingMapSelection {
    <#
    .SYNOPSIS
        Selection against a rolling map: the single-map flow below, once per measured commit.
    .DESCRIPTION
        Everything that does not depend on a map commit is established once (the base, the change's own
        paths and diff, package and CI-policy routing). Everything that does is done per layer, with that
        layer's commit as the map commit and nothing shared between layers: the change's ranges
        (Get-ScopedChangedRanges / Get-ChangedRanges, including the stale-map sweep), which files exist
        there unchanged, the manifest the layer was measured with, the auxiliary inventory check, and
        Select-ImpactedShards itself. Merge-RollingLayerSelections then keeps each layer's verdict for its own
        shards. A layer's ranges never meet another layer's diff.

        The orchestration is a copy of the single-map flow rather than a refactor of it, so that flow is
        untouched; a rolling map with one layer must select exactly what the map it was built from selects,
        and Test-TestImpactEndToEnd.ps1 holds that on real commits.

        Cost, measured per layer for a pull request: one diff from the layer's commit to the base limited to
        the pull request's paths, at most one sweep diff, one manifest read and one name-only diff, plus an
        ls-tree and a name-only diff when the change touches files only other layers index. The change's own
        diff, test routing, package routing and CI-policy review are taken once.
    #>
    $rolling = $null
    $layers = @()
    $universe = $null
    $manifest = @()
    $retiredIndexed = @()
    $unmapped = @()
    $allKnown = @()
    try {
        . "$PSScriptRoot/RollingShardMap.ps1"
        $rolling = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
        Assert-RollingShardMap -Map $rolling
        $layers = @($rolling.layers)
        # Each layer is an ordinary map over its own shards, held to the ordinary map's rules.
        foreach ($layer in $layers) { Assert-ShardMap -Map $layer.map -Expected @($layer.map.knownShards | ForEach-Object { [string] $_ }) }
        $allKnown =@($layers | ForEach-Object { @($_.map.knownShards) } | ForEach-Object { [string] $_ })
        $universe = [pscustomobject]@{
            schemaVersion = 1; sha = [string] $rolling.anchor.sha
            knownShards = @($allKnown); alwaysRun = @($rolling.alwaysRun | ForEach-Object { [string] $_ })
        }
        if ($ShardManifestFile) {
            $expectedNames = [System.Collections.Generic.HashSet[string]]::new([string[]] @($ExpectedShards), [StringComparer]::Ordinal)
            $retired = @(@($universe.knownShards) + @($universe.alwaysRun) | Where-Object { -not $expectedNames.Contains([string] $_) } | Sort-Object -Unique)
            if ($retired.Count -gt 0) {
                $retiredIndexed = @($retired | Where-Object { $_ -cin @($universe.knownShards) })
                $universe.alwaysRun = @(@($universe.alwaysRun) | Where-Object { $expectedNames.Contains([string] $_) })
                Write-Host "$($retired.Count) shard(s) in the map are no longer in the manifest: $($retired -join ', ')"
            }
            $manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
            if (@($manifest | Where-Object { $null -ne $_.PSObject.Properties['workload'] }).Count -gt 0) {
                . "$PSScriptRoot/CiWorkloadKinds.ps1"
                $extension = Complete-CiMapWorkloads -Map $universe -Manifest $manifest
                $universe = $extension.Map
                if ($extension.Added.Count -gt 0) {
                    Write-Host "$($extension.Added.Count) workload(s) are not in the coverage map yet and will run until a map includes them: $($extension.Added -join ', ')"
                }
            }
        }
        # Assert-ShardMap's universe rule: a map shard the manifest no longer has is a removal or a rename.
        $allowed = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($name in @($ExpectedShards) + @($retiredIndexed)) {
            if ([string]::IsNullOrWhiteSpace([string] $name)) { throw 'expected shard names must be nonempty' }
            [void] $allowed.Add([string] $name)
        }
        if ($allowed.Count -eq 0) { throw 'the expected shard manifest is empty' }
        $extra = @(@($universe.knownShards) + @($universe.alwaysRun) | Where-Object { -not $allowed.Contains([string] $_) } | Sort-Object -Unique)
        if ($extra.Count -gt 0) { throw "map names shard(s) the manifest does not have: $($extra -join ', ')" }
        $unmapped = @(Add-UnmappedShardsAsAlwaysRun -Map $universe -Expected $ExpectedShards)
        if ($unmapped.Count -gt 0) {
            Write-Host "$($unmapped.Count) shard(s) are not in the coverage map yet and will run until a map includes them: $($unmapped -join ', ')"
        }
    }
    catch {
        Exit-Escalated -Reason 'map-unreadable' `
            -Message "the rolling shard map at $MapFile could not be trusted - running the full matrix ($($_.Exception.Message))"
    }

    foreach ($layer in $layers) {
        & git cat-file -e "$($layer.sha)^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Exit-Escalated -Reason 'map-unresolvable' `
                -Message "the rolling map's commit $($layer.sha) is not present in this checkout - running the full matrix"
        }
    }

    try {
        $preciseFallbacks = Test-PreciseFallbacksEnabled
        $scopeToPullRequest = $false
        if (@(@($PullRequestHeadSha, $BaseSha, $DeltaFromTree) | Where-Object { $_ }).Count -gt 1) {
            throw 'pass at most one of PullRequestHeadSha, BaseSha and DeltaFromTree'
        }
        if ($PullRequestHeadSha) {
            $BaseSha = Resolve-PullRequestBase -PullRequestHeadSha $PullRequestHeadSha
            $scopeToPullRequest = $true
            Write-Host "pull request base (merge commit's first parent): $BaseSha"
            if ($MapLineageRef) {
                # Every layer's commit must be in the checkout's history, not only the anchor's.
                foreach ($layer in $layers) { $BaseSha = Resolve-MapLineageBase -LineageRef $MapLineageRef -MapSha ([string] $layer.sha) }
                Write-Host "the map is from '$MapLineageRef', not the pull request's base branch: selecting for everything the checkout adds to it (merge base $BaseSha)"
            }
        }
        elseif ($MapLineageRef) {
            throw 'MapLineageRef is only meaningful with PullRequestHeadSha'
        }
        if ($DeltaFromTree) {
            if ($DeltaFromTree -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
                throw "delta tree '$DeltaFromTree' is not a complete Git object id"
            }
            & git cat-file -e "$DeltaFromTree^{tree}" 2>$null
            if ($LASTEXITCODE -ne 0) { throw "delta tree '$DeltaFromTree' is not present in this checkout" }
            $BaseSha = $DeltaFromTree
            $scopeToPullRequest = $true
            Write-Host "delta from validated tree: $DeltaFromTree"
        }
        $currentPaths = $null
        if ($BaseSha) {
            $baseKind = if ($DeltaFromTree) { 'tree' } else { 'commit' }
            & git cat-file -e "$BaseSha^{$baseKind}"
            if ($LASTEXITCODE -ne 0) { throw "base $baseKind '$BaseSha' is not present in this checkout" }
            $currentPaths = @(& git -c core.quotepath=false diff --no-renames --name-only $BaseSha HEAD --)
            if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from current base '$BaseSha' failed" }
            $currentPaths = @($currentPaths | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        }
        if ($ShardManifestFile) {
            $manifestNames = @($manifest | ForEach-Object { [string] $_.name } | Sort-Object)
            if (($manifestNames -join "`n") -cne (@($ExpectedShards | Sort-Object) -join "`n")) {
                throw 'the shard manifest does not describe exactly the expected shards'
            }
        }

        # Commit-independent, so taken once: the change's own diff, package routing, CI-policy review.
        $ownDiff = @()
        if ($scopeToPullRequest) { $ownDiff = @(Get-ScopedChangeDiff -Base $BaseSha -Paths $currentPaths) }
        $packageRoute = $null
        $packageRouting = $null
        if ($BaseSha -and $currentPaths -ccontains $script:PackagePropsPath) {
            $verdict = Resolve-PackagePropsVerdict -BaseSha $BaseSha -Manifest $(if ($ShardManifestFile) { $manifest } else { $null })
            $script:PackagePropsVerdict = $verdict
            if ($verdict.Impact -ceq 'TestRoute') {
                $packageRoute = [pscustomobject]@{
                    Routable = $true; Shards = @($verdict.Shards); Why = $verdict.Why; Category = 'PackageVersion'
                }
            }
            Write-Host "package versions: $($verdict.Impact) - $($verdict.Why)"
            $packageRouting = [pscustomobject]@{ impact = $verdict.Impact; why = $verdict.Why; packages = @($verdict.Packages) }
        }
        $reviewed = [pscustomobject]@{ Paths = @(); Shards = @() }
        if ($BaseSha -and (Test-Path "$PSScriptRoot/CiPolicyImpact.ps1")) {
            try {
                . "$PSScriptRoot/CiPolicyImpact.ps1"
                $reviewed = Get-ReviewedCiPolicyImpact -BaseSha $BaseSha -Paths $currentPaths
                Write-Host "CI policy review: $($reviewed.Paths.Count) control paths handled; $($reviewed.Shards.Count) execution workloads required."
            }
            catch { Write-Warning "CI execution impact is unproven; retaining full-validation controls: $($_.Exception.Message)" }
        }

        $context = New-RollingSelectionContext -Layers $layers
        $definitionsCompared = [bool] $ShardManifestFile
        $scopedTestRoutes = $null
        $deltaImportReasons = [System.Collections.Generic.List[string]]::new()
        $layerResults = [System.Collections.Generic.List[object]]::new()
        $unionChanged = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        Write-Host "rolling map: $($layers.Count) measured commit(s), anchor $(([string] $rolling.anchor.sha).Substring(0, 8))"
        foreach ($layer in $layers) {
            # The ONE commit this iteration's line numbers are in. Every range below is in it.
            $mapSha = [string] $layer.sha
            $layerView = New-RollingLayerView -Layer $layer -AllKnown $allKnown -AlwaysRun @($universe.alwaysRun) -Unmapped $unmapped
            $view = $layerView.View
            $homeSet = $layerView.Home
            $homeNames = @($layer.map.knownShards | ForEach-Object { [string] $_ })
            $changeDetail = @{}
            if ($scopeToPullRequest) {
                $changed = if ($preciseFallbacks) {
                    Get-ScopedChangedRanges -MapSha $mapSha -Base $BaseSha -Paths $currentPaths -OwnDiff $ownDiff -Detail $changeDetail
                } else {
                    Get-ScopedChangedRanges -MapSha $mapSha -Base $BaseSha -Paths $currentPaths -OwnDiff $ownDiff
                }
            }
            else {
                $changed = if ($preciseFallbacks) { Get-ChangedRanges -MapSha $mapSha -Detail $changeDetail } else { Get-ChangedRanges -MapSha $mapSha }
            }
            # Two statements: an if-expression yielding @() unrolls to $null.
            $layerCurrent = @($changed.Keys)
            if ($null -ne $currentPaths) { $layerCurrent = @($currentPaths) }
            foreach ($key in $changed.Keys) { [void] $unionChanged.Add([string] $key) }
            $changedSinceMeasured = @(& git -c core.quotepath=false diff --no-renames --name-only $mapSha HEAD --).Count
            if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$mapSha' failed" }

            $testRoutes = @{}
            if ($ShardManifestFile) {
                # This layer was measured with the manifest committed at ITS commit.
                $manifestAtMap = Read-ShardManifestAtRevision -Revision $mapSha
                $manifestTracked = $null -ne $manifestAtMap
                if (-not $manifestTracked) {
                    & git cat-file -e 'HEAD:.github/test-shards.yml' 2>$null
                    $manifestTracked = $LASTEXITCODE -eq 0
                    $manifestAtMap = @()
                }
                if ($manifestTracked) {
                    $redefinedShards = @(Get-RedefinedShards -MapManifest $manifestAtMap -Manifest $manifest `
                        -IndexedShards @($homeNames | Where-Object { $_ -cnotin $retiredIndexed }))
                    if ($redefinedShards.Count -gt 0) {
                        $view | Add-Member -NotePropertyName redefinedShards -NotePropertyValue $redefinedShards -Force
                        Write-Host "$($redefinedShards.Count) shard(s) measured at $($mapSha.Substring(0, 8)) changed definition since and will run until measured again: $($redefinedShards -join ', ')"
                    }
                }
                else {
                    $definitionsCompared = $false
                    Write-Host "no committed shard manifest at $($mapSha.Substring(0, 8)) or HEAD; definition drift is not checked"
                }
                if ($scopeToPullRequest -and $null -ne $scopedTestRoutes) {
                    # A pull request's test-source routing reads the change and the manifest, not a map commit.
                    $testRoutes = $scopedTestRoutes
                }
                else {
                    $currentSet = [System.Collections.Generic.HashSet[string]]::new([string[]] $layerCurrent, [StringComparer]::OrdinalIgnoreCase)
                    $testPaths = @($changed.Keys | Where-Object {
                        $candidate = [string] $_
                        (-not $scopeToPullRequest -or $currentSet.Contains($candidate)) -and
                        $candidate.StartsWith('tests/', [StringComparison]::OrdinalIgnoreCase) -and
                        $candidate.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase) -and
                        (Get-ChangedPathImpact -Path $candidate) -eq [ChangedPathImpact]::MapCandidate -and
                        -not $context.MappedPaths.Contains($candidate)
                    })
                    $revisions = @('HEAD', $BaseSha, $mapSha)
                    $testRoutes = Get-TestFileRoutes -Paths $testPaths -Manifest $manifest `
                        -ReadFile { param($path) Get-GitFileText -Path $path -Revisions $revisions } `
                        -ReadPriorFile { param($path) Get-GitFileText -Path $path -Revisions @($BaseSha, $mapSha) } `
                        -FindReferrers { param($names, $directory) Find-GitReferrers -Names $names -Directory $directory } `
                        -FindBuildTimeReferences { param($names) Find-GitBuildTimeReferences -Names $names } `
                        -FindCollectionReferrers { param($names, $directory) Find-GitCollectionReferrers -Names $names -Directory $directory } `
                        -GeneratedDependenciesKnown (Test-ReviewedGeneratedDependencies -ChangedPaths $layerCurrent) `
                        -GeneratedDependencyProjects @('tests/AiDotNet.Tests/')
                    if ($scopeToPullRequest) { $scopedTestRoutes = $testRoutes }
                }
            }
            if ($null -ne $packageRoute) { $testRoutes[$script:PackagePropsPath] = $packageRoute }

            # Changed files another layer indexes and this one does not. "Settled" ones existed at this layer's
            # commit and are byte-identical from there to the change's base: this layer's shards were measured
            # with exactly the content the change edits. Only a scoped change has such a base; a replay of
            # everything since the layer's commit has, by definition, no file unchanged since then.
            $elsewhere = @($changed.Keys | Where-Object {
                $context.MappedPaths.Contains([string] $_) -and -not $layer.map.files.PSObject.Properties[[string] $_]
            })
            $settled = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
            for ($offset = 0; $scopeToPullRequest -and $offset -lt $elsewhere.Count; $offset += 200) {
                $batch = @($elsewhere[$offset..([Math]::Min($offset + 199, $elsewhere.Count - 1))])
                $listed = @(& git -c core.quotepath=false --literal-pathspecs ls-tree -r --name-only $mapSha -- @batch)
                if ($LASTEXITCODE -ne 0) { throw "git ls-tree at '$mapSha' failed" }
                $movedSince = @(& git -c core.quotepath=false --literal-pathspecs diff --no-renames --name-only $mapSha $BaseSha -- @batch)
                if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from '$mapSha' to '$BaseSha' failed" }
                $moved = [System.Collections.Generic.HashSet[string]]::new([string[]] @($movedSince | Where-Object { $_ }), [StringComparer]::OrdinalIgnoreCase)
                foreach ($path in $listed) { if ($path -and -not $moved.Contains([string] $path)) { [void] $settled.Add([string] $path) } }
            }
            $layerContext = [pscustomobject]@{
                Sha = $mapSha; Home = $homeSet; SettledPaths = $settled; MappedPaths = $context.MappedPaths
                DirectoryOwners = $context.DirectoryOwners; DirectoryFileCount = $context.DirectoryFileCount
            }

            $selection = Select-ImpactedShards -Map $view -Changed $changed -CurrentPaths $layerCurrent `
                -ScopeToCurrentPaths:$scopeToPullRequest -TestRoutes $testRoutes `
                -AuditUnchangedMap:$AuditUnchangedMap -ReviewedControlPaths $reviewed.Paths -RequiredShards $reviewed.Shards `
                -ChangeDetail $changeDetail -PreciseFallbacks $preciseFallbacks `
                -ReadSource { param($revision, $path) Get-GitSourceText -Path $path -Revision $revision } `
                -Rolling $layerContext

            if (-not $selection.Escalate -and $selection.RequiresValidation -and $ShardManifestFile -and
                @($manifest | Where-Object { Test-InventoryWindow $_ }).Count -gt 0) {
                . "$PSScriptRoot/AuxiliaryInventory.ps1"
                $auxiliary = @($manifest | Where-Object { Test-InventoryWindow $_ })
                $indexedAuxiliary = @($auxiliary | Where-Object { $homeSet.Contains([string] $_.name) })
                $windows = @($auxiliary | Where-Object { Test-NameKeyedWindow $_ })
                $workloads = @($auxiliary | Where-Object { -not (Test-NameKeyedWindow $_) })
                $invalidated = @()
                if (@($indexedAuxiliary | Where-Object { $_ -cin $workloads }).Count -gt 0 -and (Test-AuxiliaryInventoryChange -MapSha $mapSha)) {
                    $invalidated += @($workloads | ForEach-Object { [pscustomobject]@{ Name = $_.name; Why = 'reflection inventory changed or could not be established' } })
                }
                if (@($indexedAuxiliary | Where-Object { $_ -cin $windows }).Count -gt 0) {
                    $windowCheck = Get-AuxiliaryWindowInvalidation -MapSha $mapSha -BaseSha $BaseSha -Windows $windows
                    $invalidated += @($windowCheck.Shards | ForEach-Object { [pscustomobject]@{ Name = $_; Why = $windowCheck.Reason } })
                }
                if ($invalidated.Count -gt 0) {
                    if ($DeltaFromTree) {
                        # Imported window results refer to the old catalog, whichever layer noticed.
                        [void] $deltaImportReasons.Add('auxiliary inventory changed; delta imports are unsafe')
                    }
                    else {
                        $selection.Shards = @(@($selection.Shards) + @($invalidated.Name) | Sort-Object -Unique)
                        $selection.Routes = @($selection.Routes) + @($invalidated | ForEach-Object { "$($_.Name) <= $($_.Why)" })
                    }
                }
            }

            [void] $layerResults.Add([pscustomobject]@{
                Sha = $mapSha; Home = $homeSet; Selection = $selection; ChangedCount = $changedSinceMeasured; Index = $layer.map.files
            })
        }

        $selection = Merge-RollingLayerSelections -Layers @($layerResults) -Mandatory @(@($universe.alwaysRun) + @($reviewed.Shards))
        if ($deltaImportReasons.Count -gt 0) {
            $selection.Escalate = $true
            $selection.RequiresValidation = $true
            $selection.Reasons = @($selection.Reasons) + @($deltaImportReasons | Select-Object -Unique)
        }
        foreach ($summary in $selection.Layers) {
            Write-Host ("  measured at {0}: {1} shard(s), {2} selected{3}; {4} file(s) changed since" -f $summary.sha.Substring(0, 8),
                $summary.shards, $summary.selected, $(if ($summary.escalated) { ' (unbounded here)' } else { '' }), $summary.changedSinceMeasured)
        }
        if ($null -eq $currentPaths) { $currentPaths = @($unionChanged) }
        $changedSinceMapCount = [int] ($selection.Layers | Measure-Object -Property changedSinceMeasured -Maximum).Maximum
        Write-Host "changed files since the oldest measured commit: $changedSinceMapCount; changed by the current change: $(@($currentPaths).Count)"

        $untestedMembers = @(if ($selection.PSObject.Properties['Untested']) { @($selection.Untested) })
        $newTypePaths = @(if ($selection.PSObject.Properties['NewSourcePaths']) {
            $priorRevision = if ($BaseSha) { $BaseSha } else { [string] $rolling.anchor.sha }
            foreach ($candidate in @($selection.NewSourcePaths)) {
                & git cat-file -e "${priorRevision}:$candidate" 2>$null
                if ($LASTEXITCODE -ne 0) { $candidate }
            }
        })
        $preciseState = if (-not $preciseFallbacks) { 'off (TEST_IMPACT_PRECISE_FALLBACKS)' }
            elseif ($null -eq $script:CSharpSyntaxAvailable) { 'on (not needed by this change)' }
            elseif ($script:CSharpSyntaxAvailable) { 'on' }
            else { "unavailable, so every unexecuted range kept the file-wide rule ($script:CSharpSyntaxProblem)" }
        Write-Host "precise fallbacks: $preciseState; $($untestedMembers.Count) changed member(s) no shard executes; $($newTypePaths.Count) new source file(s)"

        if ($retiredIndexed.Count -gt 0) {
            $reachedRetired = @($selection.Shards | Where-Object { $_ -cin $retiredIndexed })
            if ($reachedRetired.Count -gt 0 -and -not $selection.Escalate -and -not $definitionsCompared) {
                $selection.Escalate = $true
                $selection.RequiresValidation = $true
                $selection.Reasons = @($selection.Reasons) + @($reachedRetired | ForEach-Object {
                    "the change reaches retired shard '$_', and no manifest comparison accounts for its tests"
                })
            }
            $selection.Shards = @($selection.Shards | Where-Object { $_ -cnotin $retiredIndexed })
        }

        if ($selection.Escalate) {
            Write-Host '::warning::selection escalated to the full matrix'
            foreach ($reason in $selection.Reasons) { Write-Host "  reason: $reason" }
        }
        else {
            Write-Host "selected $($selection.Shards.Count) of $($ExpectedShards.Count) shard(s)"
            foreach ($shard in $selection.Shards) { Write-Host "  $shard" }
        }

        $impacts = @($currentPaths | ForEach-Object { Get-ChangedPathImpact -Path ([string] $_) })
        $buildOnlyChange = $impacts.Count -gt 0 -and
            @($impacts | Where-Object { $_ -eq [ChangedPathImpact]::BuildOnly }).Count -gt 0 -and
            @($impacts | Where-Object {
                $_ -ne [ChangedPathImpact]::BuildOnly -and $_ -ne [ChangedPathImpact]::NonRuntime
            }).Count -eq 0
        if ($buildOnlyChange) {
            Write-Host 'build-only change: compiling it, but no test shard can be affected by it'
        }
        $emittedShards = @(if ($buildOnlyChange) { @() } else { $selection.Shards })

        $result = [pscustomobject]@{
            escalate          = [bool] $selection.Escalate
            requiresValidation = [bool] ($selection.RequiresValidation -or $buildOnlyChange)
            requiresShards    = [bool] ($selection.RequiresValidation -and -not $buildOnlyChange)
            reason            = $(if ($buildOnlyChange) { 'build-only' }
                                  elseif ($selection.Escalate) { 'impact-unknown' }
                                  elseif (-not $selection.RequiresValidation) { 'non-runtime-only' }
                                  else { 'selected' })
            reasons           = @($selection.Reasons)
            shards            = $emittedShards
            routes            = @($selection.Routes)
            wouldSelect       = @($selection.Shards)
            files             = @(if ($selection.PSObject.Properties['Files']) { @($selection.Files) })
            untested          = $untestedMembers
            newTypePaths      = $newTypePaths
            preciseFallbacks  = $preciseState
            packageRouting    = $packageRouting
            map               = [pscustomobject]@{
                sha = [string] $rolling.anchor.sha; changedSinceMap = $changedSinceMapCount; changedByThisChange = @($currentPaths).Count
            }
            # One entry per measured commit: how many shards it holds, how many of them this change selected,
            # and how far that commit is behind HEAD.
            rolling           = [pscustomobject]@{
                anchorSha = [string] $rolling.anchor.sha; anchorRunId = [long] $rolling.anchor.runId
                layers = @($selection.Layers)
            }
        }
        if ($OutFile) { $result | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
    }
    catch {
        Exit-Escalated -Reason 'selection-failed' `
            -Message "rolling shard selection failed ($($_.Exception.Message))"
    }
}

if (Test-RollingMapFile -Path $MapFile) {
    Invoke-RollingMapSelection
    exit 0
}

$map = $null
$retiredIndexed = @()
$definitionsCompared = $false
try {
    $map = Get-Content -LiteralPath $MapFile -Raw | ConvertFrom-Json
    if ($ShardManifestFile) {
        # Shards the manifest no longer has. Dropping an always-run entry loses nothing; an indexed
        # one keeps its position for the file index, and a change that reaches it escalates below.
        $expectedNames = [System.Collections.Generic.HashSet[string]]::new([string[]] @($ExpectedShards), [StringComparer]::Ordinal)
        $retired = @(@($map.knownShards) + @($map.alwaysRun) | Where-Object { -not $expectedNames.Contains([string] $_) } | Sort-Object -Unique)
        if ($retired.Count -gt 0) {
            $retiredIndexed = @($retired | Where-Object { $_ -cin @($map.knownShards) })
            $map.alwaysRun = @(@($map.alwaysRun) | Where-Object { $expectedNames.Contains([string] $_) })
            Write-Host "$($retired.Count) shard(s) in the map are no longer in the manifest: $($retired -join ', ')"
        }
        $manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
        if (@($manifest | Where-Object { $null -ne $_.PSObject.Properties['workload'] }).Count -gt 0) {
            # Keep certified routing usable while the manifest grows. Workloads the map has not
            # measured are ADDED as mandatory; no indexed coverage is invented for them.
            Assert-ShardMap -Map $map -Expected @(@($map.knownShards) + @($map.alwaysRun))
            . "$PSScriptRoot/CiWorkloadKinds.ps1"
            $extension = Complete-CiMapWorkloads -Map $map -Manifest $manifest
            $map = $extension.Map
            if ($extension.Added.Count -gt 0) {
                Write-Host "$($extension.Added.Count) workload(s) are not in the coverage map yet and will run until a map includes them: $($extension.Added -join ', ')"
            }
        }
    }
    Assert-ShardMap -Map $map -Expected @(@($ExpectedShards) + $retiredIndexed)
    $unmapped = @(Add-UnmappedShardsAsAlwaysRun -Map $map -Expected $ExpectedShards)
    if ($unmapped.Count -gt 0) {
        Write-Host "$($unmapped.Count) shard(s) are not in the coverage map yet and will run until a map includes them: $($unmapped -join ', ')"
    }
}
catch {
    Exit-Escalated -Reason 'map-unreadable' `
        -Message "the shard map at $MapFile could not be trusted - running the full matrix ($($_.Exception.Message))"
}

$mapSha = [string] $map.sha
& git cat-file -e "$mapSha^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Exit-Escalated -Reason 'map-unresolvable' `
        -Message "the map's commit $mapSha is not present in this checkout - running the full matrix"
}

try {
    # Per-hunk detail for the precise fallbacks; left empty, every unexecuted range keeps the file-wide rule.
    $preciseFallbacks = Test-PreciseFallbacksEnabled
    $changeDetail = @{}
    $changed = if ($preciseFallbacks) { Get-ChangedRanges -MapSha $mapSha -Detail $changeDetail } else { Get-ChangedRanges -MapSha $mapSha }
    $currentPaths = @($changed.Keys)
    $scopeToPullRequest = $false
    if (@(@($PullRequestHeadSha, $BaseSha, $DeltaFromTree) | Where-Object { $_ }).Count -gt 1) {
        throw 'pass at most one of PullRequestHeadSha, BaseSha and DeltaFromTree'
    }
    if ($PullRequestHeadSha) {
        $BaseSha = Resolve-PullRequestBase -PullRequestHeadSha $PullRequestHeadSha
        $scopeToPullRequest = $true
        Write-Host "pull request base (merge commit's first parent): $BaseSha"
        if ($MapLineageRef) {
            $BaseSha = Resolve-MapLineageBase -LineageRef $MapLineageRef -MapSha $mapSha
            Write-Host "the map is from '$MapLineageRef', not the pull request's base branch: selecting for everything the checkout adds to it (merge base $BaseSha)"
        }
    }
    elseif ($MapLineageRef) {
        throw 'MapLineageRef is only meaningful with PullRequestHeadSha'
    }
    if ($DeltaFromTree) {
        # A tree, not a commit: the validated pull-request merge commit may no longer be fetchable,
        # but its tree can be rebuilt and compared exactly; see Resolve-CiValidationReuse.ps1.
        if ($DeltaFromTree -notmatch '^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$') {
            throw "delta tree '$DeltaFromTree' is not a complete Git object id"
        }
        & git cat-file -e "$DeltaFromTree^{tree}" 2>$null
        if ($LASTEXITCODE -ne 0) { throw "delta tree '$DeltaFromTree' is not present in this checkout" }
        $BaseSha = $DeltaFromTree
        $scopeToPullRequest = $true
        Write-Host "delta from validated tree: $DeltaFromTree"
    }
    if ($BaseSha) {
        $baseKind = if ($DeltaFromTree) { 'tree' } else { 'commit' }
        & git cat-file -e "$BaseSha^{$baseKind}"
        if ($LASTEXITCODE -ne 0) { throw "base $baseKind '$BaseSha' is not present in this checkout" }
        $currentPaths = @(& git -c core.quotepath=false diff --no-renames --name-only $BaseSha HEAD --)
        if ($LASTEXITCODE -ne 0) { throw "git diff --name-only from current base '$BaseSha' failed" }
        $currentPaths = @($currentPaths | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }
    Write-Host "changed files since the map: $($changed.Count); changed by the current change: $($currentPaths.Count)"
    $changedSinceMapCount = $changed.Count
    if ($scopeToPullRequest) {
        # The change's own ranges, carried back to the map's numbering. The map-to-HEAD ranges above
        # would also sweep in every edit the base branch made to the same files since the map.
        $changeDetail = @{}
        $changed = if ($preciseFallbacks) {
            Get-ScopedChangedRanges -MapSha $mapSha -Base $BaseSha -Paths $currentPaths -Detail $changeDetail
        } else {
            Get-ScopedChangedRanges -MapSha $mapSha -Base $BaseSha -Paths $currentPaths
        }
    }

    # Test sources the map cannot index, among the paths selection will actually consider.
    $testRoutes = @{}
    if ($ShardManifestFile) {
        $manifest = @(Get-Content -LiteralPath $ShardManifestFile -Raw | ConvertFrom-Json)
        $manifestNames = @($manifest | ForEach-Object { [string] $_.name } | Sort-Object)
        if (($manifestNames -join "`n") -cne (@($ExpectedShards | Sort-Object) -join "`n")) {
            throw 'the shard manifest does not describe exactly the expected shards'
        }

        # The map was measured with the manifest committed at its own commit. An indexed shard
        # defined differently now may run other tests than its index records.
        $manifestAtMap = Read-ShardManifestAtRevision -Revision $mapSha
        $manifestTracked = $null -ne $manifestAtMap
        if (-not $manifestTracked) {
            & git cat-file -e 'HEAD:.github/test-shards.yml' 2>$null
            $manifestTracked = $LASTEXITCODE -eq 0
            $manifestAtMap = @()
        }
        if ($manifestTracked) {
            $definitionsCompared = $true
            $redefinedShards = @(Get-RedefinedShards -MapManifest $manifestAtMap -Manifest $manifest `
                -IndexedShards @($map.knownShards | Where-Object { $_ -cnotin $retiredIndexed }))
            if ($redefinedShards.Count -gt 0) {
                $map | Add-Member -NotePropertyName redefinedShards -NotePropertyValue $redefinedShards -Force
                Write-Host "$($redefinedShards.Count) mapped shard(s) changed definition since the map and will run until a map measures them: $($redefinedShards -join ', ')"
            }
        }
        else {
            Write-Host 'no committed shard manifest at the map commit or HEAD; definition drift is not checked'
        }
        $currentSet = [System.Collections.Generic.HashSet[string]]::new([string[]] $currentPaths, [StringComparer]::OrdinalIgnoreCase)
        $testPaths = @($changed.Keys | Where-Object {
            $candidate = [string] $_
            (-not $scopeToPullRequest -or $currentSet.Contains($candidate)) -and
            $candidate.StartsWith('tests/', [StringComparison]::OrdinalIgnoreCase) -and
            $candidate.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase) -and
            (Get-ChangedPathImpact -Path $candidate) -eq [ChangedPathImpact]::MapCandidate -and
            -not $map.files.PSObject.Properties[$candidate]
        })
        $revisions = @('HEAD', $BaseSha, $mapSha)
        $testRoutes = Get-TestFileRoutes -Paths $testPaths -Manifest $manifest `
            -ReadFile { param($path) Get-GitFileText -Path $path -Revisions $revisions } `
            -ReadPriorFile { param($path) Get-GitFileText -Path $path -Revisions @($BaseSha, $mapSha) } `
            -FindReferrers { param($names, $directory) Find-GitReferrers -Names $names -Directory $directory } `
            -FindBuildTimeReferences { param($names) Find-GitBuildTimeReferences -Names $names } `
            -FindCollectionReferrers { param($names, $directory) Find-GitCollectionReferrers -Names $names -Directory $directory } `
            -GeneratedDependenciesKnown (Test-ReviewedGeneratedDependencies -ChangedPaths $currentPaths) `
            -GeneratedDependencyProjects @('tests/AiDotNet.Tests/')
    }

    # Dependency bumps route by package. Only for the change being selected for: the same file in the older
    # map-to-HEAD delta keeps the shared-infrastructure rule.
    $packageRouting = $null
    if ($BaseSha -and $currentPaths -ccontains $script:PackagePropsPath) {
        $verdict = Resolve-PackagePropsVerdict -BaseSha $BaseSha -Manifest $(if ($ShardManifestFile) { $manifest } else { $null })
        $script:PackagePropsVerdict = $verdict
        if ($verdict.Impact -ceq 'TestRoute') {
            $testRoutes[$script:PackagePropsPath] = [pscustomobject]@{
                Routable = $true; Shards = @($verdict.Shards); Why = $verdict.Why; Category = 'PackageVersion'
            }
        }
        Write-Host "package versions: $($verdict.Impact) - $($verdict.Why)"
        $packageRouting = [pscustomobject]@{ impact = $verdict.Impact; why = $verdict.Why; packages = @($verdict.Packages) }
    }

    $reviewed = [pscustomobject]@{ Paths = @(); Shards = @() }
    if ($BaseSha -and (Test-Path "$PSScriptRoot/CiPolicyImpact.ps1")) {
        try {
            . "$PSScriptRoot/CiPolicyImpact.ps1"
            $reviewed = Get-ReviewedCiPolicyImpact -BaseSha $BaseSha -Paths $currentPaths
            Write-Host "CI policy review: $($reviewed.Paths.Count) control paths handled; $($reviewed.Shards.Count) execution workloads required."
        }
        catch { Write-Warning "CI execution impact is unproven; retaining full-validation controls: $($_.Exception.Message)" }
    }
    $selection = Select-ImpactedShards -Map $map -Changed $changed -CurrentPaths $currentPaths `
        -ScopeToCurrentPaths:$scopeToPullRequest -TestRoutes $testRoutes `
        -AuditUnchangedMap:$AuditUnchangedMap -ReviewedControlPaths $reviewed.Paths -RequiredShards $reviewed.Shards `
        -ChangeDetail $changeDetail -PreciseFallbacks $preciseFallbacks `
        -ReadSource { param($revision, $path) Get-GitSourceText -Path $path -Revision $revision }

    # Report-only: the changed members no shard executes, and the sources this change adds. A new
    # type's residual consumers are generated tests and reflection inventories, which only the
    # Build job's type-level selection can see; newTypePaths names the files for it.
    $untestedMembers = @(if ($selection.PSObject.Properties['Untested']) { @($selection.Untested) })
    $newTypePaths = @(if ($selection.PSObject.Properties['NewSourcePaths']) {
        $priorRevision = if ($BaseSha) { $BaseSha } else { $mapSha }
        foreach ($candidate in @($selection.NewSourcePaths)) {
            & git cat-file -e "${priorRevision}:$candidate" 2>$null
            if ($LASTEXITCODE -ne 0) { $candidate }
        }
    })
    $preciseState = if (-not $preciseFallbacks) { 'off (TEST_IMPACT_PRECISE_FALLBACKS)' }
        elseif ($null -eq $script:CSharpSyntaxAvailable) { 'on (not needed by this change)' }
        elseif ($script:CSharpSyntaxAvailable) { 'on' }
        else { "unavailable, so every unexecuted range kept the file-wide rule ($script:CSharpSyntaxProblem)" }
    Write-Host "precise fallbacks: $preciseState; $($untestedMembers.Count) changed member(s) no shard executes; $($newTypePaths.Count) new source file(s)"
    foreach ($entry in $untestedMembers) {
        Write-Host "  untested code changed: $($entry.member) in $($entry.path), lines $($entry.lines)"
    }

    if (-not $selection.Escalate -and $selection.RequiresValidation -and $ShardManifestFile -and
        @($manifest | Where-Object { Test-InventoryWindow $_ }).Count -gt 0) {
        . "$PSScriptRoot/AuxiliaryInventory.ps1"
        $auxiliary = @($manifest | Where-Object { Test-InventoryWindow $_ })
        $indexedAuxiliary = @($auxiliary | Where-Object { $_.name -cin $map.knownShards })
        # Windows keyed by model name (the committed conformance table, the hashed parameter-count split) are
        # invalidated only where a changed model identity lands. Workload sweeps keep the whole-catalog rule.
        $windows = @($auxiliary | Where-Object { Test-NameKeyedWindow $_ })
        $workloads = @($auxiliary | Where-Object { -not (Test-NameKeyedWindow $_) })
        $invalidated = @()
        if (@($indexedAuxiliary | Where-Object { $_ -cin $workloads }).Count -gt 0 -and (Test-AuxiliaryInventoryChange -MapSha $mapSha)) {
            $invalidated += @($workloads | ForEach-Object { [pscustomobject]@{ Name = $_.name; Why = 'reflection inventory changed or could not be established' } })
        }
        if (@($indexedAuxiliary | Where-Object { $_ -cin $windows }).Count -gt 0) {
            $windowCheck = Get-AuxiliaryWindowInvalidation -MapSha $mapSha -BaseSha $BaseSha -Windows $windows
            $invalidated += @($windowCheck.Shards | ForEach-Object { [pscustomobject]@{ Name = $_; Why = $windowCheck.Reason } })
        }
        if ($invalidated.Count -gt 0) {
            if ($DeltaFromTree) {
                # Imported window results refer to the old catalog. Refuse imports
                # rather than overwrite newly rerun results under the same window names.
                $selection.Escalate = $true
                $selection.Reasons = @($selection.Reasons) + @('auxiliary inventory changed; delta imports are unsafe')
            }
            else {
                $selection.Shards = @(@($selection.Shards) + @($invalidated.Name) | Sort-Object -Unique)
                $selection.Routes = @($selection.Routes) + @($invalidated | ForEach-Object { "$($_.Name) <= $($_.Why)" })
            }
        }
    }

    if ($retiredIndexed.Count -gt 0) {
        # With definitions compared, a retired shard's tests can only have moved to a shard added
        # or redefined since the map (both mandatory), been run already by an unchanged overlapping
        # shard (whose index then records them), or stopped running. Only without that comparison is
        # there nowhere to account for them.
        $reachedRetired = @($selection.Shards | Where-Object { $_ -cin $retiredIndexed })
        if ($reachedRetired.Count -gt 0 -and -not $selection.Escalate -and -not $definitionsCompared) {
            $selection.Escalate = $true
            $selection.RequiresValidation = $true
            $selection.Reasons = @($selection.Reasons) + @($reachedRetired | ForEach-Object {
                "the change reaches retired shard '$_', and no manifest comparison accounts for its tests"
            })
        }
        $selection.Shards = @($selection.Shards | Where-Object { $_ -cnotin $retiredIndexed })
    }

    if ($selection.Escalate) {
        Write-Host '::warning::selection escalated to the full matrix'
        foreach ($reason in $selection.Reasons) { Write-Host "  reason: $reason" }
    }
    else {
        Write-Host "selected $($selection.Shards.Count) of $($ExpectedShards.Count) shard(s)"
        foreach ($shard in $selection.Shards) {
            Write-Host "  $shard"
            $why = @($selection.Routes | Where-Object { $_.StartsWith("$shard <= ", [StringComparison]::Ordinal) } |
                ForEach-Object { $_.Substring($shard.Length + 4) })
            foreach ($line in @($why | Select-Object -First 5)) { Write-Host "      because it $line" }
            if ($why.Count -gt 5) { Write-Host "      ... and $($why.Count - 5) more reason(s)" }
        }
    }

    # A build-only change still has to COMPILE - an .editorconfig edit can raise an analyzer to
    # error - but it cannot alter runtime behaviour, so it must not drag in the shard matrix. The
    # build jobs and the shard matrix share requiresValidation today, so the distinction needs its
    # own flag rather than reusing that one.
    $impacts = @($currentPaths | ForEach-Object { Get-ChangedPathImpact -Path ([string] $_) })
    $buildOnlyChange = $impacts.Count -gt 0 -and
        @($impacts | Where-Object { $_ -eq [ChangedPathImpact]::BuildOnly }).Count -gt 0 -and
        @($impacts | Where-Object {
            $_ -ne [ChangedPathImpact]::BuildOnly -and $_ -ne [ChangedPathImpact]::NonRuntime
        }).Count -eq 0
    if ($buildOnlyChange) {
        Write-Host 'build-only change: compiling it, but no test shard can be affected by it'
    }
    $emittedShards = @(if ($buildOnlyChange) { @() } else { $selection.Shards })

    $result = [pscustomobject]@{
        escalate          = $selection.Escalate
        # A build-only change adds no mapped path, so Select-ImpactedShards reports no validation
        # needed. The workflow requires validation for anything that compiles and treats false as
        # the selector contradicting it - which sent every .editorconfig change to the full matrix.
        # It must compile, so validation stays required; only the shard matrix is skipped.
        requiresValidation = [bool] ($selection.RequiresValidation -or $buildOnlyChange)
        requiresShards    = [bool] ($selection.RequiresValidation -and -not $buildOnlyChange)
        reason            = $(if ($buildOnlyChange) { 'build-only' }
                              elseif ($selection.Escalate) { 'impact-unknown' }
                              elseif (-not $selection.RequiresValidation) { 'non-runtime-only' }
                              else { 'selected' })
        reasons           = $selection.Reasons
        # @() inside a $( ) subexpression unrolls to nothing, which serialises as null and makes
        # every consumer that binds this to a [string[]] fail. Build the array first.
        shards            = $emittedShards
        routes            = @($selection.Routes)
        # Report-only fields (Write-SelectionReport.ps1); the workflow's matrix decision never reads them.
        # wouldSelect is what coverage selected even when an escalation overrides it with the full matrix.
        wouldSelect       = @($selection.Shards)
        files             = @(if ($selection.PSObject.Properties['Files']) { @($selection.Files) })
        # Changed members no shard executes: nothing was selected for them, so they are named instead.
        untested          = $untestedMembers
        # C# sources this change adds, for the type-level selection in the Build job.
        newTypePaths      = $newTypePaths
        preciseFallbacks  = $preciseState
        packageRouting    = $packageRouting
        map               = [pscustomobject]@{
            sha = $mapSha; changedSinceMap = $changedSinceMapCount; changedByThisChange = @($currentPaths).Count
        }
    }
    if ($OutFile) { $result | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
}
catch {
    Exit-Escalated -Reason 'selection-failed' `
        -Message "shard selection failed, so the full matrix will run: $($_.Exception.Message)"
}

# Explicit, because falling off the end leaves the script's exit code as whatever the LAST native
# command returned. Test-source routing runs git grep, which exits 1 when nothing matches - for
# example the type names of a test file this pull request deletes - and callers read a nonzero exit
# as a selector failure and run the full matrix, discarding a valid selection.
exit 0
