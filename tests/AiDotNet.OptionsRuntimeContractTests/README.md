# Actual-model options regression checks

This runner references the real AiDotNet project and source-links the production test
fixtures. It does not compile substitute models, replacement base classes, or a published
older package in place of the PR's core library.

The options cohort creates all 17 migrated sequence models at vocabulary 32, width 16/32,
one/two blocks, and four tokens. Each comparison changes only width or depth. Assertions
inspect the real embedding parameters, physical layers/contained hybrid blocks, and flattened
parameter lengths after lazy initialization. The hybrid scheduler and RWKV7 stack are counted
through their actual contained blocks, not their constant outer layer count. Models are disposed after every case. Separate
negative controls prove an options object that advertises a changed depth cannot pass while
the actual model remains unchanged. These are construction/topology checks, not a claim that
all architecture features, training recipes, or GPU kernels have been validated.

After building the current core for .NET 10, the narrow command is:

```powershell
$env:AIDOTNET_FORCE_CPU = '1'
dotnet test tests/AiDotNet.OptionsRuntimeContractTests/AiDotNet.OptionsRuntimeContractTests.csproj -c Release -f net10.0 -p:BuildProjectReferences=false -p:GeneratePackageOnBuild=false --filter "FullyQualifiedName~OptionsSurfaceRatchetTests|FullyQualifiedName~RWKV7LanguageModelTests.Model_Constructor_" --logger "trx;LogFileName=options-runtime-net10.trx" --results-directory artifacts/options-runtime
```

The full RWKV fixture is compiled, but the filter executes only its constructor contracts.
The runner source-links the repository's `ModuleInitializer.cs`, `LicenseTestSupport.cs`, and
`xunit.runner.json`, retaining its CPU initialization, offline test licensing, and test scheduling.
Its assembly name matches the main test assembly's existing internal-API access; no production
visibility is widened. The source-linked fixtures also remain in the main test project. The discovery guard refuses
partial assembly loads, empty censuses, and missing migrated models. `ReportRemainingGaps`
writes the complete grouped report through xUnit output rather than discarding it.

Both fixture constructors explicitly call `TestModuleInitializer.EnsureInitialized()`.
The automatic module initializer is excluded on .NET Framework, so merely linking its source
does not initialize a focused `net471` run. The explicit call preserves the same CPU/threading
and offline-licensing setup without relying on another test class running first.

The runner targets `net10.0`, `net8.0` and `net471`. After the final explicit-initialization
correction, the primary reviewer ran the same 63-case filter on all three: **63 passed,
zero failed, zero skipped per target** (189 passing executions). With all three core outputs
already built, reproduce with:

```powershell
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
dotnet test tests/AiDotNet.OptionsRuntimeContractTests/AiDotNet.OptionsRuntimeContractTests.csproj -c Release -m:1 -p:BuildProjectReferences=false -p:BuildInParallel=false -p:UseSharedCompilation=false --filter 'FullyQualifiedName~OptionsSurfaceRatchetTests|FullyQualifiedName~RWKV7LanguageModelTests.Model_Constructor_' --logger 'trx;LogFilePrefix=pr2128-runtime-explicit-init' --results-directory artifacts/options-runtime
```

## Recorded local proof (2026-09-11)

Against the current PR's Release/net10.0 core, the complete filter passed **63/63, zero skips**
in four seconds of test execution. This is 34 actual-model comparisons, two unchanged-topology
negative controls, ten census/report/metadata contracts, and 17 RWKV7 constructor cases.
The census measured 1,005 concrete models and 977 gaps across 297 model types. Reapplying the
old architecture/artifact name exclusions changed the census by exactly zero; the baseline
constant remains 977.

Examples from the real materialized-parameter comparisons:

| Change | Before | After |
| --- | ---: | ---: |
| Mamba width 16 to 32 | 3,728 | 10,624 |
| Jamba depth one to two | 3,760 | 5,136 |
| RWKV7 depth one to two | 8,544 | 16,032 |
| Eagle width 16 to 32 | 4,560 | 15,744 |

The first runtime attempt was not green: 56 passed and four failed. Three failures exposed
`ParameterInfo.HasDefaultValue` attempting to box an enum nested in an open generic model;
one exposed the assertion's incorrect assumption that RWKV7 added an outer layer per block.
The metadata fix preserves the normal reflection API and uses the `HasDefault` flag only
after its specific open-generic-enum `ArgumentException`. It does not drop decimal or DateTime
defaults encoded by attributes. See the runtime's [default-value implementation](https://raw.githubusercontent.com/dotnet/runtime/main/src/coreclr/System.Private.CoreLib/src/System/Reflection/RuntimeParameterInfo.cs)
and the [metadata flag contract](https://learn.microsoft.com/en-us/dotnet/api/system.reflection.parameterattributes).

Isolated test-source mutants referenced the same current core; the production/test worktree
was not reverted for these controls:

| Isolated mutation | Expected failures observed |
| --- | --- |
| Restore old silent partial/empty discovery and discarded report | 4 failed, 1 passed; each of the four new guards rejected the old behavior |
| Restore direct `HasDefaultValue` without the narrow enum fallback | Open-generic control failed; closed-generic control passed |
| Replace all default checks with only `HasDefault` | Both metadata controls failed, detecting lost attribute constants |

The mutant runner was `artifacts/options-runtime/mutants/OptionsMutants.csproj`, with
`BuildProjectReferences=false`, the same module initializer/licensing/xUnit configuration,
and isolated test binaries. The guard mutation restored the former report body and omitted
the new discovery failures. Metadata mutations changed only the `HasDefaultValue` helper;
all assertions were unchanged. TRXs are under `artifacts/options-runtime`:
`options-runtime-net10-corrected.trx`, `options-old-guards-mutant.trx`,
`options-old-metadata-mutant.trx`, and `options-blanket-flags-mutant.trx`.

This bounded proof does not assert full training convergence, GPU parity, every option's
behavior, or completion of the remaining model-family migrations.
