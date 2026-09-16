# Sequence scaffold review proof (#2128)

The same 13 contracts run the real source generator with synthetic compiler inputs.
At PR head `671836a347436f4b023ebe3f02a86ea333ad0686`, four structural checks fail
and nine controls pass. After removing the unreachable XLSTM/Hawk rules and the
shadowed Griffin/Hawk fallback arms, all 13 pass, with no skips.
The final runner targets `net10.0`, `net8.0` and `net471`; all three passed the
same 13 contracts locally (39 passing executions, zero skipped). The original
failure-first comparison above was run on `net10.0`.

All six effective-fixture checks pass **before and after**: XLSTM, Griffin, Hawk,
GLA, GatedDeltaNet and RecurrentGemma retain their bounded dimensions. This change
does not select the previously unreachable XLSTM dimensions or change production
model defaults, seeding, test tolerances, timeouts or iteration budgets.

PR #2130 follow-up adds six semantic compilation checks and twelve negative
controls. Every effective `CreateNetwork` body is compiled against the real
AiDotNet and Tensors assemblies, using the normal test friend identity for the
internal seed scope. Each negative control first proves the original compiles,
then changes either an options property or a named constructor argument: its
syntax still parses, but the real compiler rejects it. The original 13 structural
checks are unchanged.

The expanded net10.0 run is **27 passed / 4 failed before**, **31 passed / 0 failed
after**, with no skips. Both arms use the same real `AiDotNet.dll` (SHA256
`A98360AF0A33125FD793BAC3E3D5ACA749E4BDE64F9C97888F8A66659B981203`)
and only change the generator project/source. The four failures remain the exact
original duplicate-rule/fallback findings; all six real factory compilation
checks pass in both arms. This is not a claim that those factories were previously
broken at runtime.

Reproduce from the repository root. Set `$baseline` to the extracted exact
`671836a347436f4b023ebe3f02a86ea333ad0686` tree, and `$models` to the directory
containing the frozen real `AiDotNet.dll` and `AiDotNet.Tensors.dll`. These are
absolute paths; neither property silently selects a different checkout. The
following commands retain separate build outputs and fail closed:

```powershell
$baseline = (Resolve-Path artifacts/pr2130-sequence-semantic/baseline-src).Path
$models = (Resolve-Path artifacts/pr2130-onnx-contracts/first-pass/bin/AiDotNet.AudioVisualCorrespondenceReview/release_net10.0).Path
if ((Get-FileHash "$models/AiDotNet.dll" -Algorithm SHA256).Hash -ne 'A98360AF0A33125FD793BAC3E3D5ACA749E4BDE64F9C97888F8A66659B981203') { throw 'Wrong model baseline.' }
$proofResults = (New-Item -ItemType Directory -Path ("artifacts/pr2130-sequence-semantic/replay-" + [Guid]::NewGuid().ToString('N')) -ErrorAction Stop).FullName
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'

dotnet test tools/SequenceFixtureReview/SequenceFixtureReview.csproj -c Release -f net10.0 --artifacts-path artifacts/pr2130-sequence-semantic/before -m:1 -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false "-p:ReviewModelDirectory=$models" "-p:GeneratorProjectPath=$baseline/src/AiDotNet.Generators/AiDotNet.Generators.csproj" "-p:GeneratorSourcePath=$baseline/src/AiDotNet.Generators/TestScaffoldGenerator.cs" --logger 'trx;LogFileName=sequence-generator-before.trx' --results-directory $proofResults
if ($LASTEXITCODE -eq 0) { throw 'Expected baseline failures disappeared.' }
[xml]$before = Get-Content "$proofResults/sequence-generator-before.trx" -ErrorAction Stop
if ($before.TestRun.ResultSummary.Counters.failed -ne 4 -or $before.TestRun.ResultSummary.Counters.passed -ne 27) { throw 'Unexpected baseline result.' }

dotnet test tools/SequenceFixtureReview/SequenceFixtureReview.csproj -c Release -f net10.0 --artifacts-path artifacts/pr2130-sequence-semantic/after -m:1 -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false "-p:ReviewModelDirectory=$models" "-p:GeneratorProjectPath=$((Get-Location).Path)/src/AiDotNet.Generators/AiDotNet.Generators.csproj" "-p:GeneratorSourcePath=$((Get-Location).Path)/src/AiDotNet.Generators/TestScaffoldGenerator.cs" --logger 'trx;LogFileName=sequence-generator-after-confirmed.trx' --results-directory $proofResults
if ($LASTEXITCODE -ne 0) { throw 'Current generator contracts failed.' }
[xml]$after = Get-Content "$proofResults/sequence-generator-after-confirmed.trx" -ErrorAction Stop
if ($after.TestRun.ResultSummary.Counters.failed -ne 0 -or $after.TestRun.ResultSummary.Counters.passed -ne 31) { throw 'Unexpected current result.' }
```

For a controlled original-head comparison, materialize that exact Git revision in
a separate checkout, then point `GeneratorProjectPath` and `GeneratorSourcePath`
at its generator project and `TestScaffoldGenerator.cs`, as shown above. Keep the
current contract test file unchanged. The recorded source extraction used
`git archive` on that exact revision for `src/AiDotNet.Generators`,
`Directory.Packages.props`, `NuGet.config` and `global.json`; no source was rewritten.

Original reports: `artifacts/sequence-generator/sequence-generator-before.trx`
(4 failed, 9 passed) and `sequence-generator-after-confirmed.trx` (13 passed).
Expanded reports are under `artifacts/pr2130-sequence-semantic/results`, using the
same before/after filenames and the 27/4 and 31/0 counts above.
The baseline source was extracted directly with `git archive` from the head above;
it was not reconstructed by selectively undoing fixes.

Discovery symbols remain synthetic, but emitted factories are now semantically
compiled against the actual six model APIs. These tests do **not** execute those
factories or prove training, GPU execution, every other real constructor, or a passing
CI shard. The separate real-model/options tests supply their own evidence. The
contracts are also included in the normal test project; no generated leaf test
was edited by hand.
