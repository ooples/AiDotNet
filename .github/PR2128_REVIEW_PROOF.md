# PR #2128 review evidence — 2026-09-11

Reviewed starting head: `671836a347436f4b023ebe3f02a86ea333ad0686`.
This record distinguishes focused local checks from hosted CI and actual model tests.

## Options copy and validation

The frozen options correction is `80e7540d69806c5e26de7a38b12c98932604d979`.
The identical 347 tests against the original source on Linux reported **173 failed,
174 passed**; after correction, **347 passed, zero failed, zero skipped**. Windows
`net10.0`, `net8.0` and `net471` also passed all 347, independently repeated by
the primary reviewer. See [the focused runner](../tests/AiDotNet.OptionsContractTests/README.md)
for the exact source-linked build, commands and test scope.

Two additional contracts verify all ten audio-base properties have value and
beginner documentation and that the newly introduced vision-language base uses
the same tower property names as the document base. With these included, the
primary reviewer's three-framework run passed **349 tests per framework**, no skips:

```powershell
dotnet test tests/AiDotNet.OptionsContractTests/AiDotNet.OptionsContractTests.csproj -c Release --logger 'trx;LogFilePrefix=pr2128-options-root-shared-docs' --results-directory artifacts/options-contracts/independent
```

These are executable scalar-options/metadata contracts, not model training or
GPU performance evidence. Existing default dimensions are pinned; Finch's
pre-PR public options rate of `3e-4` is preserved rather than overwritten by a
new constructor assignment. Its unread model `_learningRate` field was removed.

## Generator and diagnostic controls

[The generator runner](../tools/SequenceFixtureReview/README.md) reports **4 failed,
9 passed before; 13 passed afterward**. All six effective constructor-fixture
checks pass on both sources. The deleted XLSTM/Hawk/Griffin rules were unreachable;
their removal does not substitute a different live fixture. This is source-generator
syntax/selection evidence, not semantic/runtime validation of synthetic models.
The final generator runner also passes all 13 on `net8.0` and `net471`, with no
skips (39 final passing executions across the three target frameworks).

`CI_SHARD_INVENTORY.md` now records the actual cancelled run: one resolver job and
**zero executed shards**, not the obsolete June pass list. The PowerShell example
was parsed and exercised with ten controlled boundary cases: three invalid filters;
executed, empty, skipped-only and missing reports; and changed test, AiDotNet or
Tensors assemblies. It rejects invalid comparisons and restores the arena variable
in all cases. These helper controls did not execute model tests. Arena-off success
is explicitly not a root-cause diagnosis.

Reproduce these documentation-only boundary controls with
`pwsh -NoProfile -File .github/scripts/Test-ArenaComparisonExample.ps1`.

## Adversarial and compatibility checks

An independent review checked the generator cleanup, Finch field removal, new
vision-base naming, documentation, and diagnostic example. It identified that
hashing only the test DLL could compare changed production code; the example now
checks the test, AiDotNet and Tensors assemblies, with a negative control for each.

The request to rename `EagleOptions` was rejected as a breaking compatibility change:
both public `EagleOptions` types already exist at merge-base
`2a53ff3d4e27845773b4c9ad0a52b81f089613f7`. The language-model type was introduced
in `e966541dd941ea7095157e434683dcff95c3711d` (#1231), not this PR. Existing callers
can continue using their namespace qualification/alias. No existing public type
is removed to address that pre-existing naming ambiguity.

The core `src/AiDotNet.csproj` Release/`net10.0` build completed with **zero errors**
and 2,775 existing warnings in 4m48s. This is not a full test-project/compatibility
build or evidence that all CI shards pass. Actual sequence-model behavior and
the final hosted/review readiness disposition need their own completed results.

## Actual model and report behavior

The subsequent [actual-model runner](../tests/AiDotNet.OptionsRuntimeContractTests/README.md)
passed **63/63**, zero skipped, including a separate primary-reviewer replay.
All 17 migrated models were constructed twice for each independent width/depth
comparison. Assertions inspect real embedding tensors, physical blocks (including
contained hybrid/RWKV7 stacks), and materialized parameter lengths, not only options
returned by the model. For example, Mamba width 16 to 32 changes parameters from
**3,728 to 10,624**; RWKV7 depth one to two changes **8,544 to 16,032**. Controls
with unchanged real topology but edited options correctly fail the behavioral guard.

The report measures **1,005 models and 977 gaps across 297 model types**. Restoring
old discovery/report behavior fails four new guards; restoring the direct reflection
read fails the open-generic enum control. A blanket metadata-flag replacement fails
both constant-kind controls because it loses decimal/DateTime defaults. The accepted
fix uses that flag only for the specific open-generic enum boxing exception.

The runtime correction is commit `84b6752f7ad02187de5f8e917721dd95f94b5460`. This is
construction/configuration and report evidence; it does not claim every training
feature, every option, full GPU parity, or completion of the other migration PRs.

## Full test-project integration

The actual `tests/AiDotNet.Tests/AiDotNetTests.csproj` Release/`net10.0` build
completed in 6m59s with **zero errors and 6,819 repository warnings**. After applying
the repository's output-only serial/Workstation-GC runner hardening, the combined
review filter passed **419 tests, zero failed, zero skipped**, in eleven seconds:

```powershell
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false
& ./.github/scripts/harden-xunit-runner.ps1 -RunnerJson tests/AiDotNet.Tests/bin/Release/net10.0/xunit.runner.json
$env:AIDOTNET_FORCE_CPU = '1'
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build --no-restore --filter 'FullyQualifiedName~SequenceModelOptionsContractTests|FullyQualifiedName~SharedOptionsDocumentationContractTests|FullyQualifiedName~GeneratedSequenceFixtureContractTests|FullyQualifiedName~OptionsSurfaceRatchetTests|FullyQualifiedName~RWKV7LanguageModelTests.Model_Constructor_' --logger 'trx;LogFileName=pr2128-full-test-assembly-review.trx' --results-directory artifacts/options-runtime
```

The 419 executions comprise 341 source-linked scalar-option cases, two shared
documentation/naming cases, 13 generator cases and 63 runtime/report cases. Six
additional emitted-XML documentation cases live in the standalone options runner
and passed there on all three frameworks; they are not counted again as main
test-assembly executions.

Loaded binary SHA-256:

- `AiDotNetTests.dll`: `EEC45FF7468B418BF26B952AB67D43C00AA6C07D7E53106C40A5646B134974CB`
- `AiDotNet.dll`: `5FFC1C04DE1D4BD5B58E4065F873667FEBA969C192C7084022DBAB52DC6285E4`

This is the combined review filter on the full assembly, not an all-shards run.

## Compatibility integration

The full main test project also built both compatibility targets (`net8.0` and
`net471`) with **zero errors and 13,445 repository warnings**, in 11m58s:

```powershell
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -p:CompatBuildOnly=true -m:1 -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false
```

The same 419-case filter then passed on each actual compatibility assembly:
**419 passed, zero failed, zero skipped on net8.0 and on net471**. The three
main-assembly targets therefore account for **1,257 passing executions**. These
compatibility runs used the existing serial collection configuration and
`AIDOTNET_FORCE_CPU=1`, `DOTNET_gcServer=0`, `COMPlus_gcServer=0`, with the command
below for each target. Unlike the earlier net10 run, theory pre-enumeration and
discovery diagnostics were left enabled; the reported 8s/10s test durations
exclude full-assembly discovery and are not end-to-end timings.

```powershell
$filter = 'FullyQualifiedName~SequenceModelOptionsContractTests|FullyQualifiedName~SharedOptionsDocumentationContractTests|FullyQualifiedName~GeneratedSequenceFixtureContractTests|FullyQualifiedName~OptionsSurfaceRatchetTests|FullyQualifiedName~RWKV7LanguageModelTests.Model_Constructor_'
dotnet vstest tests/AiDotNet.Tests/bin/Release/net8.0/AiDotNetTests.dll "/TestCaseFilter:$filter" '/Logger:trx;LogFileName=pr2128-full-net8-review.trx' '/ResultsDirectory:artifacts/options-runtime' -- RunConfiguration.MaxCpuCount=1
dotnet vstest tests/AiDotNet.Tests/bin/Release/net471/AiDotNetTests.dll "/TestCaseFilter:$filter" '/Logger:trx;LogFileName=pr2128-full-net471-review.trx' '/ResultsDirectory:artifacts/options-runtime' -- RunConfiguration.MaxCpuCount=1
```

Loaded binary SHA-256:

| Target | Assembly | SHA-256 |
| --- | --- | --- |
| net8.0 | AiDotNetTests.dll | `6B646534A41FE389158733B5D7B8CAAEA61484C1E8810845826D4E5F7A94E56E` |
| net8.0 | AiDotNet.dll | `C3B431114E700958229EF0BAD10EDECBE161038C355119C2FD43796DA55B806D` |
| net471 | AiDotNetTests.dll | `51EB9BE42199A0664CD2CE1ECBB9E9CE14BC55588C7E9F657C45A931BE280175` |
| net471 | AiDotNet.dll | `342F97070EE8617B4D8ECE2BDF5AA093A8430D25A2930BCABF74BBF14C9ACD5D` |

This closes local build/target compatibility for the review changes. It does not
claim all model-family training tests, GPU execution, hosted CI, or the separate
phase-3 migration are complete. In particular, phase 3 must migrate its new
vision-base consumers to `VisionDim`/`VisionLayers` when integrating this branch;
a text-clean merge alone is not semantic compatibility proof.

## Final explicit fixture initialization and replay

The last adversarial check found that .NET Framework excludes the automatic
module initializer. Both actual-model fixture constructors now explicitly call
the existing `TestModuleInitializer.EnsureInitialized()`; this makes focused CPU,
threading and offline-licensing setup independent of other test classes. The
focused runtime runner now targets all three frameworks and passed **63/63 per
target**, zero skips. The entire existing RWKV7 unit fixture also passed **49/49
per target**, zero skips, including forward/state/parameter tests outside the
17 constructor cases. These overlapping counts are not added to the combined
filter as if they were distinct tests.

After those final two fixture edits, the main test project was rebuilt against
the unchanged, already-built core and generator outputs on all three targets:
**zero errors, 11,800 repository warnings, 5m55s**. The first no-restore attempt
had only compatibility targets in its assets file and correctly failed NETSDK1005
for net10.0; that invocation is not counted as build proof. The successful
invocation restored the complete target set:

```powershell
dotnet build tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -m:1 -p:BuildProjectReferences=false -p:UseSharedCompilation=false -p:BuildInParallel=false -p:GeneratePackageOnBuild=false
```

The primary reviewer then reran the identical combined filter, applying the
repository's output-only runner hardening on each target: **419 passed, zero
failed, zero skipped on each of net10.0, net8.0 and net471**. Final TRXs are
`artifacts/options-runtime/pr2128-final-full-net10.0.trx`,
`pr2128-final-full-net8.0.trx` and `pr2128-final-full-net471.trx`. The successful
build log is `artifacts/pr2128-final-restored-test-build.log`.

Final main test-assembly SHA-256 values (superseding the earlier test hashes;
the core hashes above are unchanged):

| Target | AiDotNetTests.dll SHA-256 |
| --- | --- |
| net10.0 | `22311D91F676B2733A09365D5AE0CDD0764F5CABB8AE7909701D6BC2ED17DBDC` |
| net8.0 | `5740BCE49D35FD7CED80800F4C64DB7E4AFA6A93692ECFDD03C24ECD882D2C60` |
| net471 | `630CE96149E2CEE5CFF17A43FCDFF337C4FC8AC99C7B1FCE6B9C0568D3318546` |
