<#
.SYNOPSIS
    Turns paper-fidelity results into RuntimeFidelityBaseline entries, and reports what they say.

.DESCRIPTION
    Reads every paper-fidelity-*.jsonl, *.trx and slice-summary.json under -ResultsRoot (the downloaded artifacts of
    a paper-fidelity-nightly.yml run, or of a pull request's paper-fidelity jobs) and:

      - counts the cases by check and outcome;
      - lists separately the outcomes that are NOT baseline material: NonDeterministic and UnknownOption (harness or
        model bugs, see OptionSensitivityHarness.IsBaselineMaterial), Error (the measurement threw), and failed
        TRX results no record accounts for (a test that timed out, or a toy task that threw before its verdict);
      - lists keys measured with conflicting outcomes (two fixtures of one model that disagree), which cannot be
        baselined either;
      - reports each slice's wall time and failed chunks;
      - with -Write, rewrites RuntimeFidelityBaseline.OptionSensitivity and .ToyTaskFailures.

    The rewrite is a merge, so a partial run cannot erase what it did not measure: every key this run measured
    takes its new state (listed with its outcome when it is baseline material, removed when it passed), and every
    existing entry this run did not measure is kept and reported.

    Entries are keyed exactly as the tests key them: ModelGenericTypeDefinitionFullName::OptionName and
    ModelGenericTypeDefinitionFullName::TestName.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $ResultsRoot,
    [string] $BaselineFile = 'tests/AiDotNet.Tests/ModelFamilyTests/Base/RuntimeFidelityBaseline.cs',
    [switch] $Write,
    # Markdown report; also appended to the job summary when GITHUB_STEP_SUMMARY is set.
    [string] $SummaryFile = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$notBaselineMaterial = @('NonDeterministic', 'UnknownOption', 'Error')
$passing = @{ OptionSensitivity = @('Sensitive'); ToyTask = @('Learned'); PaperDefaults = @('Matches') }
$validOptionOutcomes = @('Sensitive', 'Insensitive', 'OptionsNotHeld', 'OptionsNotRouted', 'NoValidPerturbation', 'NonDeterministic', 'UnknownOption')

# ---- records ----
$records = [Collections.Generic.List[object]]::new()
$files = @(Get-ChildItem -LiteralPath $ResultsRoot -Recurse -File -Filter 'paper-fidelity-*.jsonl')
foreach ($file in $files) {
    $lineNumber = 0
    foreach ($line in [IO.File]::ReadLines($file.FullName)) {
        $lineNumber++
        if (-not $line.Trim()) { continue }
        try { $records.Add(($line | ConvertFrom-Json)) }
        catch { Write-Warning "$($file.FullName):$lineNumber is not a JSON record: $($_.Exception.Message)" }
    }
}
Write-Host "$($records.Count) record(s) in $($files.Count) file(s)"

# ---- outcome counts ----
$byOutcome = $records | Group-Object { "$($_.check)|$($_.outcome)" } | Sort-Object Name

# ---- keys and their measured outcomes ----
$measuredOptions = @{}
$measuredToy = @{}
$conflicts = [Collections.Generic.List[string]]::new()
$problems = [Collections.Generic.List[object]]::new()
foreach ($r in $records) {
    if ($r.outcome -cin $notBaselineMaterial) { $problems.Add($r) }
    if ($null -eq $r.model) { continue }
    $key = "$($r.model)::$($r.property)"
    switch ([string] $r.check) {
        'OptionSensitivity' {
            if ($r.outcome -cnotin $validOptionOutcomes) { continue }
            if ($measuredOptions.ContainsKey($key) -and $measuredOptions[$key] -cne $r.outcome) {
                $conflicts.Add("$key measured as both $($measuredOptions[$key]) and $($r.outcome)")
                $measuredOptions[$key] = 'Conflict'
            }
            elseif (-not $measuredOptions.ContainsKey($key)) { $measuredOptions[$key] = [string] $r.outcome }
        }
        'ToyTask' {
            if ($measuredToy.ContainsKey($key) -and $measuredToy[$key] -cne $r.outcome) {
                $conflicts.Add("$key measured as both $($measuredToy[$key]) and $($r.outcome)")
                $measuredToy[$key] = 'Conflict'
            }
            elseif (-not $measuredToy.ContainsKey($key)) { $measuredToy[$key] = [string] $r.outcome }
        }
    }
}

# ---- TRX failures no record accounts for ----
$recorded = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($r in $records) { [void] $recorded.Add("$($r.fixture)|$($r.property)") }
$unrecorded = [Collections.Generic.List[object]]::new()
$trxCounts = @{}
foreach ($trx in @(Get-ChildItem -LiteralPath $ResultsRoot -Recurse -File -Filter '*.trx')) {
    try { [xml] $doc = Get-Content -LiteralPath $trx.FullName -Raw }
    catch { Write-Warning "$($trx.FullName) is not readable XML"; continue }
    foreach ($result in @($doc.GetElementsByTagName('UnitTestResult'))) {
        $outcome = [string] $result.GetAttribute('outcome')
        $trxCounts[$outcome] = 1 + [int] $trxCounts[$outcome]
        if ($outcome -cin @('Passed', 'NotExecuted')) { continue }
        $name = [string] $result.GetAttribute('testName')
        $head = ($name -split '\(')[0]
        $dot = $head.LastIndexOf('.')
        if ($dot -lt 1) { continue }
        $fixture = $head.Substring(0, $dot)
        $method = $head.Substring($dot + 1)
        $option = [regex]::Match($name, 'option: "(?<o>[^"]*)"')
        $property = if ($option.Success) { $option.Groups['o'].Value } else { $method }
        if ($method -like '*_ShipsPaperDefaults') { $property = $null }
        if ($null -ne $property -and $recorded.Contains("$fixture|$property")) { continue }
        $message = $result.SelectSingleNode('.//*[local-name()="Message"]')
        $unrecorded.Add([pscustomobject]@{
            test = $name; outcome = $outcome
            message = $(if ($null -ne $message) { ([string] $message.InnerText).Trim() -replace '\s+', ' ' } else { '' })
        })
    }
}

# ---- slices ----
$slices = @(Get-ChildItem -LiteralPath $ResultsRoot -Recurse -File -Filter 'slice-summary.json' |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json } | Sort-Object slice)

# ---- baseline merge ----
$text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $BaselineFile).Path)
$nl = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
$optionHead = [regex]::new('OptionSensitivity\s*=\s*new Dictionary<string, OptionSensitivityOutcome>\(StringComparer\.Ordinal\)\s*\{')
$toyHead = [regex]::new('ToyTaskFailures\s*=\s*new HashSet<string>\(StringComparer\.Ordinal\)\s*\{')
function Get-Body([string] $Source, [regex] $Head) {
    $m = $Head.Match($Source)
    if (-not $m.Success) { throw "baseline initializer $($Head.ToString()) not found in $BaselineFile" }
    $end = $Source.IndexOf('};', $m.Index + $m.Length, [StringComparison]::Ordinal)
    if ($end -lt 0) { throw "baseline initializer $($Head.ToString()) is not terminated" }
    return [pscustomobject]@{ Start = $m.Index + $m.Length; End = $end; Text = $Source.Substring($m.Index + $m.Length, $end - $m.Index - $m.Length) }
}
$existingOptions = [ordered]@{}
foreach ($m in [regex]::Matches((Get-Body $text $optionHead).Text, '\{\s*"(?<k>(?:[^"\\]|\\.)*)"\s*,\s*OptionSensitivityOutcome\.(?<o>\w+)\s*\}')) {
    $existingOptions[$m.Groups['k'].Value] = $m.Groups['o'].Value
}
$existingToy = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($m in [regex]::Matches((Get-Body $text $toyHead).Text, '"(?<k>(?:[^"\\]|\\.)*)"')) { [void] $existingToy.Add($m.Groups['k'].Value) }

$newOptions = [Collections.Generic.SortedDictionary[string, string]]::new([StringComparer]::Ordinal)
$keptUnmeasured = [Collections.Generic.List[string]]::new()
foreach ($k in $existingOptions.Keys) {
    if (-not $measuredOptions.ContainsKey($k)) { $newOptions[$k] = $existingOptions[$k]; $keptUnmeasured.Add($k) }
}
foreach ($k in $measuredOptions.Keys) {
    $o = $measuredOptions[$k]
    if ($o -ceq 'Conflict' -or $o -cin $notBaselineMaterial -or $o -cin $passing.OptionSensitivity) { continue }
    $newOptions[$k] = $o
}
$newToy = [Collections.Generic.SortedSet[string]]::new([StringComparer]::Ordinal)
foreach ($k in $existingToy) {
    if (-not $measuredToy.ContainsKey($k)) { [void] $newToy.Add($k); $keptUnmeasured.Add($k) }
}
foreach ($k in $measuredToy.Keys) { if ($measuredToy[$k] -ceq 'NotLearned') { [void] $newToy.Add($k) } }

if ($Write) {
    $optionLines = ($newOptions.GetEnumerator() | ForEach-Object { "            { `"$($_.Key)`", OptionSensitivityOutcome.$($_.Value) },$nl" }) -join ''
    $toyLines = ($newToy | ForEach-Object { "            `"$_`",$nl" }) -join ''
    # Rewrite the later initializer first so the earlier one's offsets stay valid.
    $bodies = @(
        [pscustomobject]@{ Body = (Get-Body $text $optionHead); Lines = $optionLines },
        [pscustomobject]@{ Body = (Get-Body $text $toyHead); Lines = $toyLines }
    ) | Sort-Object { $_.Body.Start } -Descending
    foreach ($b in $bodies) {
        $text = $text.Substring(0, $b.Body.Start) + $nl + $b.Lines + '        ' + $text.Substring($b.Body.End)
    }
    $hasBom = ([IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $BaselineFile).Path) | Select-Object -First 3) -join ',' -eq '239,187,191'
    [IO.File]::WriteAllText((Resolve-Path -LiteralPath $BaselineFile).Path, $text, [Text.UTF8Encoding]::new($hasBom))
    Write-Host "wrote $($newOptions.Count) option-sensitivity and $($newToy.Count) toy-task entries to $BaselineFile"
}

# ---- report ----
$md = [Collections.Generic.List[string]]::new()
$md.Add('## Paper fidelity results'); $md.Add('')
$md.Add("$($records.Count) recorded case(s); TRX: " + ((@($trxCounts.Keys | Sort-Object) | ForEach-Object { "$_ $($trxCounts[$_])" }) -join ', ') + '.')
$md.Add('')
$md.Add('| Check | Outcome | Cases |'); $md.Add('|---|---|---:|')
foreach ($g in $byOutcome) { $parts = $g.Name.Split('|'); $md.Add("| $($parts[0]) | $($parts[1]) | $($g.Count) |") }
$md.Add('')
$md.Add("Baseline after merge: $($newOptions.Count) option-sensitivity entr(ies), $($newToy.Count) toy-task entr(ies); $($keptUnmeasured.Count) existing entr(ies) kept because this run did not measure them.")
if ($slices.Count -gt 0) {
    $md.Add(''); $md.Add('| Slice | Classes | Wall (s) | Failed chunks |'); $md.Add('|---:|---:|---:|---:|')
    foreach ($s in $slices) { $md.Add("| $($s.slice) | $($s.classes) | $($s.wallSeconds) | $($s.failedChunks) |") }
}
foreach ($section in @(
        @{ Title = 'NonDeterministic (harness or model bugs, never baselined)'; Rows = @($problems | Where-Object { $_.outcome -ceq 'NonDeterministic' }) },
        @{ Title = 'UnknownOption (generator/runtime disagreement, never baselined)'; Rows = @($problems | Where-Object { $_.outcome -ceq 'UnknownOption' }) },
        @{ Title = 'Error (the measurement threw)'; Rows = @($problems | Where-Object { $_.outcome -ceq 'Error' }) })) {
    $md.Add(''); $md.Add("### $($section.Title): $($section.Rows.Count)")
    foreach ($r in $section.Rows) { $md.Add("- ``$($r.fixture)`` / ``$($r.property)``: $([string] $r.failure)") }
}
$md.Add(''); $md.Add("### Conflicting measurements: $($conflicts.Count)")
foreach ($c in $conflicts) { $md.Add("- $c") }
$md.Add(''); $md.Add("### Failed TRX results without a record: $($unrecorded.Count)")
foreach ($u in $unrecorded) { $md.Add("- ``$($u.test)`` ($($u.outcome)): $($u.message)") }

$report = $md -join "`n"
if ($SummaryFile) { Set-Content -LiteralPath $SummaryFile -Value $report -Encoding utf8 }
if ($env:GITHUB_STEP_SUMMARY) { $report | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Append -Encoding utf8 }
Write-Host $report
