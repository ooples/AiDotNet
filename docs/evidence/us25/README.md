# US-25 consumer deployment verification

Companion [PR #2202](https://github.com/ooples/AiDotNet/pull/2202), [Evolution issue #43](https://github.com/ooples/AiDotNet.Evolution/issues/43), and ready [Tensors PR #1030](https://github.com/ooples/AiDotNet.Tensors/pull/1030).

## Verified local gates

Production source: `5d9a5c19e11b4013ccbb61c6ad4b4f67f3aa3b6a`. Final test-only fixture isolation/denial regression: `b17f649b8bc7d057634ba6e91c2b5cb211e96de4`. Pinned Evolution source: `255feb24369702a32ea9db7a3f8a0b7a847d2762`.

| Gate | Passed | Failed | Skipped | Raw receipt |
| --- | ---: | ---: | ---: | --- |
| Windows .NET 10 consumer integration | 1,044 | 0 | 0 | `net10.0-isolated/consumer.trx` |
| Windows .NET Framework 4.7.1 compatibility | 49 | 0 | 0 | `net471-isolated/consumer.trx` |
| Linux .NET 10 consumer integration, existing local container | 1,044 | 0 | 0 | `consumer-linux.trx` in `linux-local.zip` |

All three suites include all **32 deployment cases**. The actual consumer library was built with its normal analyzers before these tests. After test-only corrections, `BuildProjectReferences=false` reused the unchanged, already-built production binaries. Earlier complete production gates (`net10.0-final`, `net471-final`) passed 1,043 and 48 tests before adding license-denial coverage and fixture isolation; those receipts are retained too.

The trained-model case performs actual CPU `MultipleRegression<double>` training through private AutoML, held-out paired prediction, guarded promotion, registry reload and trained-state deserialization. Its explicit `1e-7` prediction tolerance checks functional integration, not competitor superiority. The synthetic fixture uses the existing async-local internal persistence scope; production adapters retain normal entitlement checks. The separate denial regression checks that model licensing errors propagate.

## Raw archive

[`verification.zip`](verification.zip): **1,086,998 bytes**, SHA-256 `8d591f8013061d3a03da9e19b821ff7142fa91b6aee200c600567076a13cb804`.

Contains local build/test console logs and TRX, including failed and corrected attempts, plus the completed failed Linux job log described below. It is not a claim that every entry passed. Current-head hosted Linux .NET 10/.NET 8 results and scoped coverage are retained separately by the PR's `Compiler-guided evolution` workflow; consult the linked PR's exact-head checks and artifacts. Local Windows results alone do not prove Linux behavior.

[`linux-local.zip`](linux-local.zip): **267,375 bytes**, SHA-256 `9146a76171e7b0ae6040ec1b0468fa2caf69dc2ef055dcead258acf955f408dd`. Contains both the initial targeted Linux pass (32/32) and the full consumer pass (1,044/1,044), with console logs and TRX. These executed the final portable net10 binaries in the already-present Linux SDK container, with networking disabled, read-only source mount, two CPUs, 2 GiB memory and disposable temporary storage. No image download or rebuild was required. Image identity: `sha256:4ea6fe75dd36706bb6d8c3c293d4c4315840f5d76ea28ac97def77e3ec487fa5`.

Executed binary SHA-256: `AiDotNetTests.dll` = `f1e470456922314fc72cc6f61cf13c770774ff8c54c1c37c2fb5a484b296e9f1`; `AiDotNet.dll` = `7065cba7d280c3ed0baa6f780fe4c47f0b63e401ac2a4007c750f2aeb3d06ec4`. This is actual Linux runtime execution, not a Linux source-build or net8-runtime claim. Hosted net8 and the normal repository CI remain merge gates while runners are queued; local cross-platform verification makes the implementation reviewable without treating the queue as a passing check.

## Failures and corrections retained

1. `net10.0.log`: initial real-library build found 11 CS0104 JSON ambiguities caused by globally imported Newtonsoft.Json. Explicit System.Text.Json aliases fixed them.
2. `net10.0-corrected.log`: production built, but new fixture calls omitted required descriptor dictionaries and used a nonexistent `MultipleLinearRegression` name. Corrected to the existing `MultipleRegression` API.
3. `net10.0-test-corrected/consumer.trx`: 1,041 passed, two failed. The core rejects zero evaluation grace; the private retuner now leaves grace null so evaluation remains attached to its outer admission. A real trained/reloaded prediction was `7.4999999940840185`, not decimal-rounding-equal to `7.5`; the fixture now uses explicit absolute tolerance.
4. `net471.log`: deliberately stopped the stale owned compile after discovering the retuner correction. This is an interrupted attempt, not a compatibility pass. Only the verified child compiler/build processes were stopped.
5. `linux-net8.log`: [run 34898244650](https://github.com/ooples/AiDotNet/actions/runs/34898244650), completed job `104157738795`, failed the new AutoML fixture because a fresh runner exhausted ten trial save/load operations during repeated restoration. Test-only async-local isolation removes machine-entitlement dependence without changing production enforcement. The replacement current-head Linux run is a separate required check, not inferred from Windows.

## Reproduction

Build/test `tests/AiDotNet.Evolution.Integration.Tests/AiDotNet.Evolution.Integration.Tests.csproj` for `net10.0` (or `net8.0`); on Windows, also test `tests/AiDotNet.Evolution.Compatibility.Tests/AiDotNet.Evolution.Compatibility.Tests.csproj` for `net471`.

```powershell
$env:DOTNET_GCHeapHardLimit = '0x300000000'
$env:DOTNET_PROCESSOR_COUNT = '4'
dotnet test tests/AiDotNet.Evolution.Integration.Tests/AiDotNet.Evolution.Integration.Tests.csproj -c Release -f net10.0 -m:1 -p:UseLocalEvolution=true -p:EvolutionProjectPath=C:/Users/cheat/AiDotNet.Evolution-us10/src/AiDotNet.Evolution/AiDotNet.Evolution.csproj -p:GeneratePackageOnBuild=false -p:UseSharedCompilation=false --logger 'trx;LogFileName=consumer.trx'
```

Point `EvolutionProjectPath` at an independently checked-out pinned core source. Do not place core beneath the consumer where it inherits the consumer's central package configuration. No package publication/cache substitution is needed. These source-path gates do not waive ordinary NuGet dependency-path failures or the story's declared dependencies. No paid model calls, GPU superiority experiments, production activation or merges were performed.
