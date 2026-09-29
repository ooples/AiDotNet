<#
.SYNOPSIS
    Decides which pull-request auxiliary workloads a change can affect: the model performance
    census, the samples run and the docs wiki gate.

.DESCRIPTION
    These three used to be separate workflows triggered by path filters as wide as 'src/**', so
    every source push took a runner for each of them - and 34 for the census - whatever it
    changed. They now run inside the validation pipeline, reuse its build, and run only when this
    selector says the change can affect them. Every rule fails closed: anything it cannot prove
    irrelevant runs.

      census     full      census tooling, its fixture base, the shard manifest or this selector
                           changed, or the shard selection escalated;
                 selected  the fixtures owned by the test shards the coverage selection chose
                           (Select-CensusFixtures.ps1 maps them);
                 none      nothing under the census's own paths changed, or the change is
                           non-runtime.
      samples    run when samples/ or root build configuration changed, a non-C# file under src/
                 changed, or a C# file's token stream changed. Comments and whitespace (XML docs
                 included) cannot change what a sample compiles or does.
      docsWiki   run when the docs content, the wiki/snippet tools or root build configuration
                 changed, the source generator changed, a non-C# file under src/ changed, or a C#
                 file's declarations or documentation comments changed. Method bodies cannot
                 change the generated wiki or whether a snippet compiles.

.PARAMETER BaseSha
    The base the pull request is validated against: the first parent of GitHub's merge commit.

.PARAMETER ValidationScope
    The shard selection's outcome: 'full' (escalated), 'selected' or 'none' (non-runtime change).

.PARAMETER SelectedShards
    The test shards the coverage selection chose when ValidationScope is 'selected'.

.PARAMETER SelfTest
    Runs the built-in adversarial checks against a scratch repository and exits.
#>
[CmdletBinding(DefaultParameterSetName = 'Select')]
param(
    [Parameter(Mandatory, ParameterSetName = 'Select')] [string] $BaseSha,
    [Parameter(ParameterSetName = 'Select')] [string] $HeadSha = 'HEAD',
    [Parameter(Mandatory, ParameterSetName = 'Select')]
    [ValidateSet('full', 'selected', 'none')] [string] $ValidationScope,
    [Parameter(ParameterSetName = 'Select')] [AllowEmptyCollection()] [string[]] $SelectedShards = @(),
    [Parameter(ParameterSetName = 'Select')] [string] $OutFile,
    [Parameter(Mandatory, ParameterSetName = 'SelfTest')] [switch] $SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. "$PSScriptRoot/AuxiliaryInventory.ps1"

# Paths whose change can alter how every census fixture is measured or judged, or how fixtures are
# mapped to shards. The fixture base holds the census test itself, which the correctness shards
# skip, so their coverage says nothing about it.
$script:CensusControlPaths = @(
    'tools/ModelPerfFixtureRunner/',
    'tools/ModelPerfProbe/',
    'tests/AiDotNet.Tests/ModelFamilyTests/Base/',
    '.github/workflows/model-performance-census.yml',
    '.github/model-performance-intent.json',
    '.github/test-shards.yml',
    'tools/TestImpact/Select-AuxiliaryWorkloads.ps1',
    'tools/TestImpact/Select-Shards.ps1',
    'tools/TestImpact/AuxiliaryInventory.ps1'
)
# The census's former pull_request path filter. A change outside it never ran the census.
$script:CensusScopePaths = @(
    'src/',
    'tests/AiDotNet.Tests/ModelFamilyTests/',
    'tools/ModelPerfFixtureRunner/',
    'tools/ModelPerfProbe/',
    '.github/workflows/model-performance-census.yml'
)
# Build configuration that applies to every project, samples and tools included.
$script:RootBuildPaths = @(
    'Directory.Build.props', 'Directory.Build.targets', 'Directory.Packages.props',
    'global.json', 'nuget.config', 'NuGet.config'
)
$script:SamplesPaths = @(
    'samples/',
    '.github/workflows/samples.yml',
    'tools/TestImpact/Select-AuxiliaryWorkloads.ps1'
)
$script:DocsWikiPaths = @(
    'website/src/content/docs/',
    'tools/WikiGenerator/',
    'tools/DocSnippetVerify/',
    'src/AiDotNet.Generators/',
    '.github/workflows/docs-wiki.yml',
    'tools/TestImpact/Select-AuxiliaryWorkloads.ps1',
    'tools/TestImpact/AuxiliaryInventory.ps1'
)

function Test-PathUnder {
    param([Parameter(Mandatory)] [string] $Path, [Parameter(Mandatory)] [string[]] $Prefixes)
    foreach ($prefix in $Prefixes) {
        if ($prefix.EndsWith('/')) {
            if ($Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
        elseif ($Path.Equals($prefix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Get-CSharpParse {
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Source)
    $tree = [Microsoft.CodeAnalysis.CSharp.CSharpSyntaxTree]::ParseText($Source)
    if (@($tree.GetDiagnostics() | Where-Object Severity -eq Error).Count -gt 0) {
        throw 'the file does not parse'
    }
    return $tree.GetRoot()
}

function Add-TokenText {
    param([Text.StringBuilder] $Builder, $Token)
    # Length-prefixed so adjacent tokens cannot run together into a different sequence.
    [void] $Builder.Append($Token.RawKind).Append(':').Append($Token.Text.Length).Append(':').Append($Token.Text).Append("`n")
}

function Get-CSharpRuntimeText {
    <#
        Everything that can change what the file compiles to: every token, string literals exactly,
        preprocessor directives and inactive (#if'd-out) text. Comments - documentation comments
        included - and whitespace are dropped.
    #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Source)
    $root = Get-CSharpParse $Source
    $text = [Text.StringBuilder]::new()
    foreach ($token in $root.DescendantTokens($null, $false)) {
        foreach ($trivia in $token.LeadingTrivia) {
            if ($trivia.IsDirective) {
                foreach ($directiveToken in $trivia.GetStructure().DescendantTokens()) { Add-TokenText $text $directiveToken }
            }
            elseif ($trivia.RawKind -eq [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::DisabledTextTrivia) {
                [void] $text.Append('disabled:').Append($trivia.ToFullString()).Append("`n")
            }
        }
        Add-TokenText $text $token
    }
    return $text.ToString()
}

function Get-CSharpDocumentationText {
    <# The documentation comments, in order: what the wiki generator and IntelliSense read. #>
    param([Parameter(Mandatory)] [AllowEmptyString()] [string] $Source)
    $root = Get-CSharpParse $Source
    $docKinds = @(
        [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::SingleLineDocumentationCommentTrivia,
        [int] [Microsoft.CodeAnalysis.CSharp.SyntaxKind]::MultiLineDocumentationCommentTrivia
    )
    $text = [Text.StringBuilder]::new()
    foreach ($trivia in $root.DescendantTrivia()) {
        if ($trivia.RawKind -in $docKinds) { [void] $text.Append($trivia.ToFullString()).Append("`n") }
    }
    return $text.ToString()
}

function Get-GitBlob {
    <# File text at a revision, or $null when the file does not exist there. #>
    param([Parameter(Mandatory)] [string] $Revision, [Parameter(Mandatory)] [string] $Path)
    $lines = @(& git show "${Revision}:$Path" 2>$null)
    if ($LASTEXITCODE -ne 0) { $global:LASTEXITCODE = 0; return $null }
    return ($lines -join "`n")
}

function Test-CSharpChange {
    <#
        Whether a changed C# file differs in the aspect a consumer depends on. Added, deleted and
        unparseable files are changes.
    #>
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [ValidateSet('Runtime', 'Declarations')] [string] $Aspect,
        [Parameter(Mandatory)] [string] $Base,
        [Parameter(Mandatory)] [string] $Head
    )
    $before = Get-GitBlob -Revision $Base -Path $Path
    $after = Get-GitBlob -Revision $Head -Path $Path
    if ($null -eq $before -or $null -eq $after) { return $true }
    try {
        if ($Aspect -eq 'Runtime') {
            return (Get-CSharpRuntimeText $before) -cne (Get-CSharpRuntimeText $after)
        }
        if ((Get-AuxiliaryInventorySignature $before) -cne (Get-AuxiliaryInventorySignature $after)) { return $true }
        return (Get-CSharpDocumentationText $before) -cne (Get-CSharpDocumentationText $after)
    }
    catch {
        Write-Host "  $Path cannot be compared ($($_.Exception.Message)); treated as changed"
        return $true
    }
}

function Get-SourceReason {
    <# First src/ change a consumer depends on, as a reason string, or $null. #>
    param([string[]] $Paths, [string] $Aspect, [string] $Base, [string] $Head)
    foreach ($path in $Paths) {
        if (-not $path.StartsWith('src/', [StringComparison]::OrdinalIgnoreCase)) { continue }
        if (-not $path.EndsWith('.cs', [StringComparison]::OrdinalIgnoreCase)) { return "$path changed (not C#)" }
        if (Test-CSharpChange -Path $path -Aspect $Aspect -Base $Base -Head $Head) {
            $what = if ($Aspect -eq 'Runtime') { 'code' } else { 'declarations or documentation' }
            return "$path changed its $what"
        }
    }
    return $null
}

function Get-AuxiliaryDecision {
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] [string[]] $Paths,
        [Parameter(Mandatory)] [string] $Scope,
        [AllowEmptyCollection()] [string[]] $Shards = @(),
        [Parameter(Mandatory)] [string] $Base,
        [Parameter(Mandatory)] [string] $Head
    )

    $control = @($Paths | Where-Object { Test-PathUnder $_ $script:CensusControlPaths })
    $inScope = @($Paths | Where-Object { Test-PathUnder $_ $script:CensusScopePaths })
    $census = if ($control.Count -gt 0) {
        [pscustomobject]@{ mode = 'full'; shards = @(); reason = "$($control[0]) changed" }
    }
    elseif ($inScope.Count -eq 0) {
        [pscustomobject]@{ mode = 'none'; shards = @(); reason = 'no model source, model test or census path changed' }
    }
    elseif ($Scope -eq 'full') {
        [pscustomobject]@{ mode = 'full'; shards = @(); reason = 'shard selection ran the full matrix' }
    }
    elseif ($Scope -eq 'none') {
        [pscustomobject]@{ mode = 'none'; shards = @(); reason = 'non-runtime change' }
    }
    elseif (@($Shards).Count -eq 0) {
        [pscustomobject]@{ mode = 'full'; shards = @(); reason = 'a runtime change selected no shards' }
    }
    else {
        [pscustomobject]@{ mode = 'selected'; shards = @($Shards | Sort-Object -Unique); reason = "fixtures of $(@($Shards).Count) selected shard(s)" }
    }

    $root = @($Paths | Where-Object { Test-PathUnder $_ $script:RootBuildPaths })
    $samplesPath = @($Paths | Where-Object { Test-PathUnder $_ $script:SamplesPaths })
    $samplesReason = if ($samplesPath.Count -gt 0) { "$($samplesPath[0]) changed" }
                     elseif ($root.Count -gt 0) { "$($root[0]) changed" }
                     else { Get-SourceReason -Paths $Paths -Aspect Runtime -Base $Base -Head $Head }
    $wikiPath = @($Paths | Where-Object { Test-PathUnder $_ $script:DocsWikiPaths })
    $wikiReason = if ($wikiPath.Count -gt 0) { "$($wikiPath[0]) changed" }
                  elseif ($root.Count -gt 0) { "$($root[0]) changed" }
                  else { Get-SourceReason -Paths $Paths -Aspect Declarations -Base $Base -Head $Head }

    return [pscustomobject]@{
        census = $census
        samples = [pscustomobject]@{ run = [bool] $samplesReason; reason = $(if ($samplesReason) { $samplesReason } else { 'no sample, build configuration or source code change' }) }
        docsWiki = [pscustomobject]@{ run = [bool] $wikiReason; reason = $(if ($wikiReason) { $wikiReason } else { 'no docs, tool, declaration or documentation change' }) }
    }
}

if ($SelfTest) {
    $failures = [Collections.Generic.List[string]]::new()
    function Assert-Decision([bool] $Condition, [string] $Message) { if (-not $Condition) { [void] $failures.Add($Message) } }

    $scratch = Join-Path ([IO.Path]::GetTempPath()) ("aux-select-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    Push-Location $scratch
    try {
        & git init -q .
        & git config user.email 'selftest@example.invalid'
        & git config user.name 'selftest'
        & git config core.autocrlf false
        $model = @'
namespace AiDotNet.Models;
/// <summary>Adds numbers.</summary>
public class Adder
{
    /// <summary>Sums.</summary>
    public int Add(int a, int b)
    {
        // plain comment
        return a + b;
    }
    private const string Label = "sum";
}
'@
        New-Item -ItemType Directory -Force -Path 'src/Models', 'samples/Basic', 'website/src/content/docs' | Out-Null
        Set-Content -LiteralPath 'src/Models/Adder.cs' -Value $model -NoNewline
        Set-Content -LiteralPath 'src/Models/Notes.md' -Value 'notes' -NoNewline
        Set-Content -LiteralPath 'samples/Basic/Program.cs' -Value 'System.Console.WriteLine(1);' -NoNewline
        Set-Content -LiteralPath 'website/src/content/docs/intro.md' -Value 'intro' -NoNewline
        & git add -A; & git commit -q -m base
        $base = (& git rev-parse HEAD).Trim()

        function Invoke-Case {
            param([string] $Name, [scriptblock] $Edit, [string] $Scope = 'selected', [string[]] $Shards = @('ModelFamily - NeuralNetworks A-F'))
            & git checkout -q $base
            & $Edit
            & git add -A
            & git commit -q -m $Name --allow-empty
            $head = (& git rev-parse HEAD).Trim()
            $paths = @(& git -c core.quotepath=false diff --no-renames --name-only $base $head)
            return Get-AuxiliaryDecision -Paths $paths -Scope $Scope -Shards $Shards -Base $base -Head $head
        }
        function Set-Model([string] $From, [string] $To) {
            Set-Content -LiteralPath 'src/Models/Adder.cs' -Value ($model.Replace($From, $To)) -NoNewline
        }

        $d = Invoke-Case 'body' { Set-Model 'return a + b;' 'return b + a;' }
        Assert-Decision ($d.samples.run) 'a method-body change did not run the samples'
        Assert-Decision (-not $d.docsWiki.run) 'a method-body change ran the docs wiki'
        Assert-Decision ($d.census.mode -ceq 'selected' -and @($d.census.shards).Count -eq 1) 'a mapped source change did not select census fixtures by shard'

        $d = Invoke-Case 'string' { Set-Model '"sum"' '"total"' }
        Assert-Decision ($d.samples.run) 'a string-literal change did not run the samples'

        $d = Invoke-Case 'comment' { Set-Model '// plain comment' "// reworded comment`n        // over two lines" }
        Assert-Decision (-not $d.samples.run) 'a comment-only change ran the samples'
        Assert-Decision (-not $d.docsWiki.run) 'a plain-comment change ran the docs wiki'

        $d = Invoke-Case 'doc' { Set-Model '<summary>Sums.</summary>' '<summary>Returns the sum.</summary>' }
        Assert-Decision (-not $d.samples.run) 'a documentation-only change ran the samples'
        Assert-Decision ($d.docsWiki.run) 'a documentation change did not run the docs wiki'

        $d = Invoke-Case 'signature' { Set-Model 'int Add(int a, int b)' 'long Add(int a, int b)' }
        Assert-Decision ($d.samples.run -and $d.docsWiki.run) 'a signature change did not run both samples and docs wiki'

        $d = Invoke-Case 'directive' { Set-Model 'namespace AiDotNet.Models;' "#define EXTRA`nnamespace AiDotNet.Models;" }
        Assert-Decision ($d.samples.run) 'a preprocessor change did not run the samples'

        $d = Invoke-Case 'whitespace' { Set-Model 'return a + b;' 'return   a+b ;' }
        Assert-Decision (-not $d.samples.run -and -not $d.docsWiki.run) 'a whitespace-only change ran samples or docs wiki'

        $d = Invoke-Case 'added' { Set-Content -LiteralPath 'src/Models/New.cs' -Value 'namespace A; class B { }' -NoNewline }
        Assert-Decision ($d.samples.run -and $d.docsWiki.run) 'an added source file did not run samples and docs wiki'

        $d = Invoke-Case 'deleted' { Remove-Item -LiteralPath 'src/Models/Adder.cs' }
        Assert-Decision ($d.samples.run -and $d.docsWiki.run) 'a deleted source file did not run samples and docs wiki'

        $d = Invoke-Case 'broken' { Set-Model 'return a + b;' 'return a + ;' }
        Assert-Decision ($d.samples.run -and $d.docsWiki.run) 'an unparseable source file was not treated as changed'

        $d = Invoke-Case 'non-cs' { Set-Content -LiteralPath 'src/Models/Notes.md' -Value 'changed' -NoNewline }
        Assert-Decision ($d.samples.run -and $d.docsWiki.run) 'a non-C# source file did not run samples and docs wiki'

        $d = Invoke-Case 'sample' { Set-Content -LiteralPath 'samples/Basic/Program.cs' -Value 'System.Console.WriteLine(2);' -NoNewline } -Scope 'none' -Shards @()
        Assert-Decision ($d.samples.run -and -not $d.docsWiki.run) 'a sample-only change did not run exactly the samples'
        Assert-Decision ($d.census.mode -ceq 'none') 'a sample-only change ran the census'

        $d = Invoke-Case 'docs' { Set-Content -LiteralPath 'website/src/content/docs/intro.md' -Value 'changed' -NoNewline } -Scope 'none' -Shards @()
        Assert-Decision ($d.docsWiki.run -and -not $d.samples.run -and $d.census.mode -ceq 'none') 'a docs-only change did not run exactly the docs wiki'

        $d = Invoke-Case 'root' { Set-Content -LiteralPath 'Directory.Build.props' -Value '<Project />' -NoNewline } -Scope 'full' -Shards @()
        Assert-Decision ($d.samples.run -and $d.docsWiki.run) 'a root build-configuration change did not run samples and docs wiki'

        $d = Invoke-Case 'escalated' { Set-Model 'return a + b;' 'return b + a;' } -Scope 'full' -Shards @()
        Assert-Decision ($d.census.mode -ceq 'full') 'an escalated selection did not run the full census'

        $d = Invoke-Case 'empty-selection' { Set-Model 'return a + b;' 'return b + a;' } -Scope 'selected' -Shards @()
        Assert-Decision ($d.census.mode -ceq 'full') 'a runtime change with no selected shard did not fail closed to the full census'

        $d = Invoke-Case 'non-runtime' { Set-Content -LiteralPath 'src/Models/Notes.md' -Value 'changed' -NoNewline } -Scope 'none' -Shards @()
        Assert-Decision ($d.census.mode -ceq 'none') 'a non-runtime change ran the census'

        foreach ($control in @('tools/ModelPerfProbe/Probe.cs', 'tests/AiDotNet.Tests/ModelFamilyTests/Base/Base.cs', '.github/test-shards.yml')) {
            $d = Invoke-Case "control $control" {
                New-Item -ItemType Directory -Force -Path (Split-Path $control) | Out-Null
                Set-Content -LiteralPath $control -Value 'x' -NoNewline
            }
            Assert-Decision ($d.census.mode -ceq 'full') "$control did not run the full census"
        }

        $d = Invoke-Case 'unrelated' { New-Item -ItemType Directory -Force -Path 'tools/Other' | Out-Null; Set-Content -LiteralPath 'tools/Other/x.cs' -Value 'class X { }' -NoNewline } -Scope 'selected' -Shards @('Unit - 01')
        Assert-Decision ($d.census.mode -ceq 'none' -and -not $d.samples.run -and -not $d.docsWiki.run) 'an unrelated tool change ran an auxiliary workload'
    }
    finally {
        Pop-Location
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($failures.Count -gt 0) {
        $failures | ForEach-Object { Write-Host "::error::$_" }
        exit 1
    }
    Write-Host 'Auxiliary workload selection passed: body, string, comment, documentation, signature, directive, whitespace, added, deleted, unparseable, non-C#, sample, docs, root-config, escalation, empty-selection, non-runtime, census-control and unrelated cases.'
    exit 0
}

$paths = @(& git -c core.quotepath=false diff --no-renames --name-only $BaseSha $HeadSha)
if ($LASTEXITCODE -ne 0) { throw "could not diff $BaseSha..$HeadSha" }
$decision = Get-AuxiliaryDecision -Paths $paths -Scope $ValidationScope -Shards $SelectedShards -Base $BaseSha -Head $HeadSha
Write-Host "census: $($decision.census.mode) - $($decision.census.reason)"
Write-Host "samples: $(if ($decision.samples.run) { 'run' } else { 'skip' }) - $($decision.samples.reason)"
Write-Host "docs wiki: $(if ($decision.docsWiki.run) { 'run' } else { 'skip' }) - $($decision.docsWiki.reason)"
if ($OutFile) { $decision | ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8 }
exit 0
