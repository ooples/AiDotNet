# AiDotNet Evolve public baseline — 2026-09-01

This baseline was captured before adding the public evolution engine. It separates pre-existing build/test behavior from regressions introduced by this branch.

## Source identity

- Worktree: `C:/Users/cheat/Temp/aidotnet-evolve`
- Branch: `feature/evolve-quality-diversity`
- Base: `origin/master` at `e0553749e97b3e070f9200bb26cf3f7a7a4e90af`
- .NET SDK: `10.0.303`
- Tensor dependency mode: default package reference; `UseLocalTensors` was not enabled
- The existing dirty `C:/Users/cheat/AiDotNet` Autoformer worktree was not modified

## Build baseline

Command:

```powershell
dotnet build src\AiDotNet.csproj --configuration Release --nologo
```

Result: passed for `net10.0`, `net8.0`, and `net471`; 8,400 pre-existing warnings, 0 errors, elapsed 00:07:07.28. The warning surface is primarily existing generator/analyzer diagnostics. This branch must introduce no new warnings in files it adds or changes.

## Targeted test baseline

All tests ran on `net10.0` in Release configuration.

| Scope | Result |
|---|---:|
| Fully-qualified name contains `AutoML` | 435 passed, 0 failed, 0 skipped |
| Fully-qualified name contains `Genetics` | 216 passed, 0 failed, 0 skipped |
| `AutoMLCompatibilityBaselineTests` | 2 passed, 0 failed, 0 skipped |

The first test attempt intentionally used `--no-restore` in the clean worktree and stopped with `NETSDK1004` because the test asset file did not yet exist. Restoring once resolved the setup condition; it is not counted as a product failure. Existing `NU1608`, analyzer, and xUnit warnings remain baseline noise.

## Compatibility characterization

The existing serialized numeric values are:

| Strategy | Value |
|---|---:|
| `RandomSearch` | 0 |
| `BayesianOptimization` | 1 |
| `Evolutionary` | 2 |
| `MultiFidelity` | 3 |
| `NeuralArchitectureSearch` | 4 |
| `DARTS` | 5 |
| `GDAS` | 6 |
| `OnceForAll` | 7 |

`AutoMLOptions<T, TInput, TOutput>.SearchStrategy` defaults to `RandomSearch`. The characterization tests pin these values and the default while permitting new strategies to be appended.

## Private recovery evidence

The private recovered implementation is preserved in HarmonicEngine commit `7596cd7` and annotated tag `hre-evolve-recovery-2026-09-01`. Its existing full-versus-local-ablation result is negative and is not treated as an OpenEvolve comparison.
