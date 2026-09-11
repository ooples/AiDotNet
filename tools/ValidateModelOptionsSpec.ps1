param(
    [string]$RepositoryRoot = (Join-Path $PSScriptRoot '..'),
    [ValidatePattern('^[0-9a-f]{7,40}$')]
    [string]$SpecRevision
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

enum ModelOptionsSpecCheck
{
    FamilyRoster
    ParentContract
    PropertyRoster
    RecordedBaseline
    PhaseContinuity
    CompletionTarget
}

$failures = [System.Collections.Generic.List[string]]::new()

function Add-Failure([ModelOptionsSpecCheck]$Check, [string]$Message)
{
    $failures.Add("${Check}: $Message")
}

function Get-GitSource([string]$Revision, [string]$Path)
{
    $source = & git -C $RepositoryRoot show ($Revision + ':' + $Path)
    if ($LASTEXITCODE -ne 0)
    {
        throw "Cannot read immutable source ${Revision}:$Path. Fetch the documented implementation history before validating."
    }
    return $source -join "`n"
}

$specPath = 'docs/superpowers/specs/2026-09-08-model-options-surface-design.md'
$spec = if ([string]::IsNullOrEmpty($SpecRevision))
{
    Get-Content -Raw -LiteralPath (Join-Path $RepositoryRoot $specPath)
}
else
{
    Get-GitSource $SpecRevision $specPath
}

$familySectionMatch = [regex]::Match($spec, '(?s)### 5\.2 Family base classes(?<body>.*?)### 5\.3 Constructors')
if (-not $familySectionMatch.Success)
{
    throw 'Cannot locate the authoritative family section.'
}
$familySection = $familySectionMatch.Groups['body'].Value
$parentContract = [regex]::Match($familySection, 'All seven derive\s+from `(?<parent>\w+)`')
if (-not $parentContract.Success -or $parentContract.Groups['parent'].Value -ne 'ModelHyperparameterOptions')
{
    Add-Failure ParentContract 'All seven families must identify their recorded common parent, ModelHyperparameterOptions.'
}
$familyPaths = @(
    'src/NeuralNetworks/Options/SequenceModelOptions.cs',
    'src/NeuralNetworks/Options/VisionLanguageModelOptions.cs',
    'src/NeuralNetworks/Options/GanOptions.cs',
    'src/NeuralNetworks/Options/EmbeddingModelOptions.cs',
    'src/Models/Options/DocumentNeuralNetworkOptions.cs',
    'src/Video/Options/VideoHyperparameterOptions.cs',
    'src/Models/Options/AudioHyperparameterOptions.cs'
)
$declaredFamilyRows = [regex]::Matches($familySection, '(?m)^\|\s*`\w+Options`\s*\|')
if ($declaredFamilyRows.Count -ne $familyPaths.Count)
{
    Add-Failure FamilyRoster "Expected exactly $($familyPaths.Count) family rows, found $($declaredFamilyRows.Count)."
}

foreach ($familyPath in $familyPaths)
{
    $familyName = [System.IO.Path]::GetFileNameWithoutExtension($familyPath)
    $source = Get-GitSource '944a1fd72' $familyPath
    $declaration = [regex]::Match($source, 'class\s+' + [regex]::Escape($familyName) + '\s*:\s*ModelHyperparameterOptions\b')
    if (-not $declaration.Success)
    {
        throw "Recorded family declaration does not match its documented parent: $familyName"
    }

    $row = [regex]::Match($familySection, '(?m)^\|\s*`' + [regex]::Escape($familyName) + '`\s*\|(?<body>[^\r\n]+)')
    if (-not $row.Success)
    {
        Add-Failure FamilyRoster "Missing authoritative row for $familyName."
        continue
    }
    if (-not $row.Value.Contains($familyPath))
    {
        Add-Failure FamilyRoster "Incorrect or missing source path for $familyName."
    }
    $properties = [regex]::Matches($source, 'public\s+[\w?]+\s+(?<name>\w+)\s*\{\s*get;\s*set;\s*\}')
    $columns = $row.Groups['body'].Value.Split('|')
    if ($columns.Length -lt 2)
    {
        Add-Failure PropertyRoster "Missing property column for $familyName."
        continue
    }
    $documentedProperties = [regex]::Matches($columns[1], '`(?<name>\w+)`')
    if ($documentedProperties.Count -ne $properties.Count)
    {
        Add-Failure PropertyRoster "$familyName must list exactly $($properties.Count) recorded properties, without duplicates or invented additions."
    }
    foreach ($property in $properties)
    {
        $propertyName = $property.Groups['name'].Value
        if (-not $row.Value.Contains('`' + $propertyName + '`'))
        {
            Add-Failure PropertyRoster "$familyName omits recorded property $propertyName."
        }
    }
}

$ratchetPath = 'tests/AiDotNet.Tests/IntegrationTests/Configuration/OptionsSurfaceRatchetTests.cs'
$initialSource = Get-GitSource '9f7fe07b6' $ratchetPath
$sequenceSource = Get-GitSource '671836a34' $ratchetPath
$initialMatch = [regex]::Match($initialSource, 'private const int Baseline = (?<count>\d+);')
$sequenceMatch = [regex]::Match($sequenceSource, 'private const int Baseline = (?<count>\d+);')
if (-not $initialMatch.Success -or -not $sequenceMatch.Success)
{
    throw 'Cannot locate the recorded baseline constants.'
}
$initial = [int]$initialMatch.Groups['count'].Value
$afterSequence = [int]$sequenceMatch.Groups['count'].Value
$phaseSection = [regex]::Match($spec, '(?s)## 8\. Phasing(?<body>.*?)## 9\. Risks and open questions').Groups['body'].Value
$rows = [regex]::Matches($phaseSection, '(?m)^\|\s*(?:~~)?(?<phase>\d+[ab]?)(?:~~)?\s*\|[^\r\n]+')
$previous = $initial
$transitions = 0
$sequenceFound = $false
foreach ($row in $rows)
{
    $phase = $row.Groups['phase'].Value
    $phaseNumber = [int]([regex]::Match($phase, '^\d+').Value)
    $transition = [regex]::Match($row.Value, '(?<before>\d+)\s*→\s*(?<after>\d+)')
    if (-not $transition.Success) { continue }
    $before = [int]$transition.Groups['before'].Value
    $after = [int]$transition.Groups['after'].Value
    if ($before -ne $previous -or $after -gt $before)
    {
        Add-Failure PhaseContinuity "Phase $phase begins at $before after $previous, or increases the count to $after."
    }
    if ($phaseNumber -eq 2)
    {
        $sequenceFound = $true
        if ($before -ne $initial -or $after -ne $afterSequence)
        {
            Add-Failure RecordedBaseline "Phase 2 must retain its recorded $initial to $afterSequence measurement."
        }
    }
    $previous = $after
    $transitions++
}
if (-not $sequenceFound -or $transitions -eq 0)
{
    Add-Failure RecordedBaseline 'Missing recorded phase-2 transition.'
}
if ($previous -ne 0)
{
    Add-Failure CompletionTarget "Schedule leaves $previous in-scope gaps; excluded types do not justify a nonzero floor."
}

if ($failures.Count -gt 0)
{
    $failures | ForEach-Object { Write-Output "FAIL $_" }
    exit 1
}
Write-Output "PASS: seven source-backed family rows and properties; recorded $initial/$afterSequence baselines; $transitions continuous phase transitions ending at zero."
Write-Output 'This verifies specification consistency against recorded source, not a new reflection measurement or runtime model behavior.'
