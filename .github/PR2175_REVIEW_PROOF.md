# PR #2175 review proof

This is local source/runtime evidence, not a claim that remote CI is green. Reviewed GitHub head:
`1034da3c513e0da738205f07aa9829c07708e83d`, branch `fix/financial-agent-exploration`, target `master`.
A fresh query confirmed that head and ten unresolved threads, with `hasNextPage=false` for both
thread and comment pagination. No reviews were resolved or remote state changed by this implementation agent.

## Review mapping

All abbreviated thread IDs below have prefix `PRRT_kwDOKSXUF86`.

| Thread / comment | Correction or explicit review decision |
| --- | --- |
| [hpspn / 3993742717](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742717) | Exact action length and one-hot comparison in `T`, including signed zero and decimal values adjacent to one; owned transition vectors. |
| [hpspq / 3993742722](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742722) | Consume the complete bounded current-policy rollout once before either update; no replay sampling, old-policy leftovers, or partial-failure retry. |
| [hpspv / 3993742727](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742727) | Reject negative legacy hidden counts before architecture/seed mutation; preserve zero and positive counts. |
| [hpsp0 / 3993742736](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742736) | Complete value/default/units and beginner documentation, including A2C rollout semantics. |
| [hpsp5 / 3993742743](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742743) | Document the actual compatibility default and primary-paper differences below; do not invent one paper default or introduce unrequested breaking presets. |
| [hpsp7 / 3993742746](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742746) | Validate both epsilon endpoints as finite probabilities in [0,1] before checking their ordering. |
| [hpsp- / 3993742752](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742752) | Invoke virtual options validation before base-option copying or derived resource/architecture mutation; prove all five constructors honor it. |
| [hpsqE / 3993742759](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742759) | Different-seed controls for all five agents, retaining same-seed checks. |
| [hpsqP / 3993742773](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742773) | Every real policy/critic/target architecture, including all four SAC critics, actual widths/input geometry and independent storage. |
| [hpsqW / 3993742781](https://github.com/ooples/AiDotNet/pull/2175#discussion_r3993742781) | Caller layer identity and order, not just count. |

### Default provenance

Before the option-wiring commit, `513426930aeabf17c7eb9e83cda97f7b86e34d1c` used two 64-wide
hidden layers in `TradingAgentBase.EnsureDefaultLayers`, while its `HiddenLayers=[256,128,64]`
option was not consumed there. Commit `7a3de9cc2700ef9b786287777beaecfa16e9e49e` exposed that actual default.

The papers do not prescribe one shared financial MLP:

- PPO's MuJoCo experiment used 64-by-64 Tanh policy layers. That experiment-specific match does
  not establish a universal trading architecture. [PPO, section 6.1](https://arxiv.org/pdf/1707.06347)
- SAC reported two 256-wide ReLU hidden layers. [SAC supplement, appendix D/table 1](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf)
- Asynchronous actor-critic used different Atari and continuous-control architectures, not this
  one-step financial feature MLP. [Asynchronous Methods, sections 8–9](https://arxiv.org/html/1602.01783)
- Original DQN used a pixel-input convolutional architecture. [DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

Therefore `[64,64]` remains an explicitly documented AiDotNet compatibility default. Reproductions
configure the relevant experiment's architecture explicitly; this does not claim a universal paper recommendation.

## Before/after evidence

The runner references the actual library project and published `AiDotNet.Tensors 0.130.3`, not a
source-linked substitute or unpublished package. Artifact paths below are relative to `artifacts/pr2175-review`.

| Actual snapshot | Result | TRX |
| --- | --- | --- |
| Exact head, first baseline | 47 pass / 40 fail / 0 skip | `results/financial-review-before.trx` |
| Exact head, corrected/expanded baseline | 50 pass / 56 intended failures / 0 skip, 106 total | `results/financial-review-final-before.trx` |
| First corrected core | 106 pass / 0 fail / 0 skip | `results/financial-review-first-after.trx` |
| Adversarial storage controls against that first core | 4 pass / 4 intended failures / 0 skip | `results/financial-live-storage-final-before.trx` |
| Final net10 core, all financial and new boundary controls | 115 pass / 0 fail / 0 skip | `results/financial-review-final-net10.trx` |
| Existing shared regressions, exact-head positive control | 39 pass / 0 fail / 0 skip | `results/financial-shared-before-net10.trx` |
| Existing shared regressions, final net10 core | 39 pass / 0 fail / 0 skip | `results/financial-shared-final-net10.trx` |

The first forty failures included one unsuitable custom-layer serialization fixture; it was replaced
with real serializable Dense layers before the final 106-case baseline. That earlier result is retained,
but is not claimed as forty product defects. The final baseline failures all reach intended assertions.
Stronger all-network/SAC ownership and all-five-agent seed controls passed before the fixes; these are
regression coverage, not invented pre-existing failures.

The negative controls cover malformed actions, vector ownership, whole-rollout consumption, capacity,
initial versus subsequent batch readiness, failures before and after the critic update, successful and
failed restore, explicit gradients, ordinary live tensor/network/layer writes, delayed selected actions,
and opaque legacy compatibility.

The later adversarial controls distinguish real policy state from bookkeeping:

- Real multi-row BatchNorm prediction preserves running means/variances and versions, even after
  a caller sets the layer to training mode. External running-mean mutation changes actual prediction
  and invalidates stale behavior. Scratch-only forward churn is harmless.
- Nested opaque layers were making an unwanted `GetParameters` copy per Store; a counting real
  Dense-backed legacy layer proves it. Unchanged fp16 conversion snapshots falsely invalidated actions.
- Reordering unchanged buffer identities with an alias registration falsely changed a list-based stamp.

The shared internal iterator now reads physical full-/half-precision storage identity/version from the
existing ordered declarations, without checkpoint payloads. Reference-keyed reused dictionaries remove
alias/order dependence. Opaque children retain the explicit owner-update contract. A direct control
asserts the original `Tensor<Half>` object and its version, not just a matching numeric conversion.

### Shared chunk-write numerical defect

`SetParameterChunks` copied through `AsWritableSpan` without publishing a tensor version or
invalidating the identity-keyed inference cache. In a warmed `[8,512] x [512,64]` GEMM with inputs 1
and initial weights 2, changing only weight index 1 to 3 still returned **1024 instead of 1025**.
A whole-array change initially passed because Tensors 0.130.3 samples a content fingerprint; the
interior-element control proves the missing writer invalidation.

Each actual destination write now advances its version and uses targeted existing CPU-array and GPU
persistent/resident invalidation APIs. No aggregate parameter copy or global CPU-cache flush was added.
Tests preserve destination/backing-array identity and prove malformed streams change neither values
nor versions. CPU results do not establish a measured GPU numerical/performance result.

### Frozen baseline hashes

| Artifact | SHA-256 |
| --- | --- |
| Exact-head core: `before/bin/AiDotNet/release_net10.0/AiDotNet.dll` | `2ABB8A40B4A4394867E504ADEC7287AC10C60C12AFBAFA14A3025CB72F2A453D` |
| 106-case final-before test DLL | `C5AE4D862A522E4959E480A1B8C6F0D8F783CD7F18500F7665D406AD9FA60A98` |
| First corrected core: `after/bin/AiDotNet/release_net10.0/AiDotNet.dll` | `CB53AD61DC05AEF49B27A3E5731DF7A75D4399A3A8552DCEBDF92810F0B26231` |
| Eight-case storage-before runner | `AD05EBAFB17CCA2BBEACF8B3E22ACA6B3DBBE152E3C6B5FFA4C1BA3C3564341D` |

Initial and final-before runners are archived in `initial-before-runner` and `final-before-runner`.
Non-destructive NTFS compression preserved the frozen hashes. This task deleted no files.

## Final validation

The final net10 actual-library build succeeded with 0 errors / 2,761 warnings in 4m52s;
`final-net10-retry-build.log` records the result. All 115 focused cases passed in one serial process
(10s). Unlike the intermediate 106-pass run, this includes the final snapshot-free iterator and all
nine live-storage controls. Eight of those controls have the recorded earlier-core negative result;
the ninth directly calls the new internal storage API and is after-only identity/version coverage.
An independent reviewer checked the exact core/test hashes below and replayed the frozen focused
binary: 115 passed / 0 failed / 0 skipped in 8s, recorded separately in
`final/results/financial-review-root-independent115.trx`.

The first final compiler failed with CS8104 / insufficient disk space while writing its output;
see `final-net10-build.log`. Space was subsequently recovered externally, not by this task's deletion.
The failed build is not treated as successful proof. The unchanged production snapshot also built
successfully on net8 (0 errors / 2,761 warnings, 3m34s) and net471 (0 errors / 2,763 warnings, 3m26s).
The focused 115-case suite passed on net8 in 10s and net471 in 16s, with no failures or skips.

| Target | Focused cases | Existing shared cases | Final TRX files under `results/` |
| --- | --- | --- | --- |
| net10.0 | 115 passed, 0 failed/skipped | 39 passed, 0 failed/skipped | `financial-review-final-net10.trx`, `financial-shared-final-net10.trx` |
| net8.0 | 115 passed, 0 failed/skipped | 39 passed, 0 failed/skipped | `financial-review-final-net8.trx`, `financial-shared-final-net8.trx` |
| net471 | 115 passed, 0 failed/skipped | 39 passed, 0 failed/skipped | `financial-review-final-net471.trx`, `financial-shared-final-net471.trx` |

Build logs are `final-net10-retry-build.log`, `final-net8-build.log`, and `final-net471-build.log`;
the corresponding `final-<target>-tests.log` and `shared-<target>-tests.log` retain runner output.
The shared suites took 10s / 10s / 16s respectively. Counts represent 154 selected cases per
framework, not 462 different test cases or coverage of the excluded foundation-scale lane.

### Existing shared-boundary regression cohort

The separate `AiDotNet.FinancialSharedReview` project links existing tests unchanged and references
the actual library. Its 39 ordinary cases pass both against the exact-head core (13s) and the final
core (10s). This preserves the frozen 115-case runner while exercising the shared changes:

| Existing class | Cases | Relevant contract |
| --- | ---: | --- |
| `ParameterStateChunkTests` | 3 | Persistent buffers, sparse payload and generated component parity. |
| `ParameterCapabilityGateTests` | 4 | Registered sources, roles, visible restore and owner write-through. |
| `CompositeLayerOwnershipTests` | 5 | Nested layers do not double-own child tensors. |
| `LayerParameterRoundTripTests` | 2 | Deferred parameter and serialized shape restoration. |
| `PredictorParameterStreamingTests` | 17 | Per-index chunk order, streamed restore and real predictor clone output. |
| `DeepAgentsIntegrationTests` | 8 | Real DQN/actor-critic/continuous/offline/model-based/multi-agent workflows, parameter updates and supported checkpoint roundtrips. |

The two existing EMMDiT `HeavyTimeout` tests require the separate foundation-scale lane and are
explicitly not selected here; no test was newly skipped, reduced or given a larger timeout. The
test-only xUnit framework calls the existing CPU initializer before discovery/execution, including
on net471 where the production test module initializer is not automatic.

| Final artifact under `final/bin` | SHA-256 |
| --- | --- |
| `AiDotNet/release_net10.0/AiDotNet.dll` | `4C5C6F988C22D27039D58EB15477D91E57E64D802A04A3331BC41F847D0F8C93` |
| `AiDotNet.FinancialAgentReview/release_net10.0/AiDotNetTests.dll` | `77F970E9E9BC18D820007A931901AFC3DA5EE4AD4EC0D2099A74BC9D51F191C5` |
| `AiDotNet.FinancialSharedReview/release_net10.0/AiDotNetTests.dll` | `EF9E55E84CF78202CAF7FF3A59026F3A32F42E60F175BBBAEDDFAC52D34E6E2F` |
| `AiDotNet/release_net8.0/AiDotNet.dll` | `47A4404F459D3BCDB2E39D076C402E6314CDE628294ECDAE677E638BBBA1BE91` |
| `AiDotNet.FinancialAgentReview/release_net8.0/AiDotNetTests.dll` | `D75114D22511003A920114254E7A5BDD047CB0DC7A1B7BA298C9D65DEA687F21` |
| `AiDotNet.FinancialSharedReview/release_net8.0/AiDotNetTests.dll` | `9122EDF671C8DE6541B1EC0C456154BA74E111188E87EA0C9E739C7E86DAAC21` |
| `AiDotNet/release_net471/AiDotNet.dll` | `57ED244696C9E2ACD315A8973DC7795446CCDB96C22946D29174EFCF7D7E89F6` |
| `AiDotNet.FinancialAgentReview/release_net471/AiDotNetTests.dll` | `482951B7ADCB6844D7D044A17BF13988DC938225DEFC6AF1A0E32E7B7E3B1DD8` |
| `AiDotNet.FinancialSharedReview/release_net471/AiDotNetTests.dll` | `505964D58BC074450F9C96BF1020AD4D1FD0E4085994A6ECE88BE5492529B738` |

## Reproduce the final cohort

From the repository root in PowerShell, build into a fresh artifact directory. Set the target
framework explicitly; run frameworks serially.

```powershell
$ErrorActionPreference = 'Stop'
$targetFramework = 'net10.0'
$artifactRoot = Join-Path $PWD ('artifacts/pr2175-reproduce-' + [Guid]::NewGuid().ToString('N'))
$project = 'tests/AiDotNet.FinancialAgentReview/AiDotNet.FinancialAgentReview.csproj'
$env:AIDOTNET_FORCE_CPU = '1'
$env:DOTNET_gcServer = '0'
$env:COMPlus_gcServer = '0'
# Select the cached AVX2 route on AVX-512 hosts for the interior-weight control.
$env:DOTNET_EnableAVX512F = '0'
$env:COMPlus_EnableAVX512F = '0'
dotnet build $project -c Release -f $targetFramework --artifacts-path $artifactRoot -m:1 `
  -p:UseSharedCompilation=false -p:CopyLocalRuntimeTargetAssets=false `
  -p:_GetChildProjectCopyToOutputDirectoryItems=false -p:GeneratePackageOnBuild=false -v:quiet --nologo
if ($LASTEXITCODE -ne 0) { throw 'Actual build failed; do not run old binaries.' }
$runnerDirectory = Join-Path $artifactRoot ('bin/AiDotNet.FinancialAgentReview/release_' + $targetFramework)
$runner = Join-Path $runnerDirectory 'AiDotNetTests.dll'
if (-not (Test-Path -LiteralPath $runner)) { throw 'Fresh test assembly missing.' }
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 `
  -RunnerJson (Join-Path $runnerDirectory 'xunit.runner.json')
if ($LASTEXITCODE -ne 0) { throw 'Runner hardening failed.' }
dotnet vstest $runner ('--Logger:trx;LogFileName=financial-' + $targetFramework + '.trx') `
  ('--ResultsDirectory:' + (Join-Path $artifactRoot 'results'))
if ($LASTEXITCODE -ne 0) { throw 'Regression cohort failed.' }

# Reuse only the successfully built actual core from this same artifact root.
$sharedProject = 'tests/AiDotNet.FinancialSharedReview/AiDotNet.FinancialSharedReview.csproj'
dotnet build $sharedProject -c Release -f $targetFramework --artifacts-path $artifactRoot -m:1 `
  -p:BuildProjectReferences=false -p:UseSharedCompilation=false `
  -p:CopyLocalRuntimeTargetAssets=false -p:_GetChildProjectCopyToOutputDirectoryItems=false `
  -p:GeneratePackageOnBuild=false -v:quiet --nologo
if ($LASTEXITCODE -ne 0) { throw 'Shared-boundary runner build failed.' }
$sharedDirectory = Join-Path $artifactRoot ('bin/AiDotNet.FinancialSharedReview/release_' + $targetFramework)
pwsh -NoProfile -File .github/scripts/harden-xunit-runner.ps1 `
  -RunnerJson (Join-Path $sharedDirectory 'xunit.runner.json')
if ($LASTEXITCODE -ne 0) { throw 'Shared runner hardening failed.' }
dotnet vstest (Join-Path $sharedDirectory 'AiDotNetTests.dll') `
  '--TestCaseFilter:Category!=HeavyTimeout' `
  ('--Logger:trx;LogFileName=financial-shared-' + $targetFramework + '.trx') `
  ('--ResultsDirectory:' + (Join-Path $artifactRoot 'results'))
if ($LASTEXITCODE -ne 0) { throw 'Shared-boundary regression cohort failed.' }
```

## Limits and compatibility

- Pending rollout/action provenance is runtime-only. The shared pre-restore hook invalidates it
  before topology/state mutation, including failed restores. Other RL agents retain a no-op pre-hook
  and their existing successful-restore hook.
- Ordinary published live-storage writes are detected. Retained raw spans, inference-mode mutation
  and opaque custom storage require the agent's explicit update boundary. Concurrent policy mutation
  during collection is unsupported. Copied/manual actions have no verifiable original selection
  provenance; the caller must supply current-policy sampled actions.
- `BatchSize > ReplayBufferSize` remains unable to reach autonomous readiness; this is documented,
  not silently reinterpreted. Initial warmup is not repeated after consuming a rollout.
- No correctness tolerances/assertions were relaxed, generated model-test leaves manually edited,
  model-specific serializer introduced, null-forgiving operator added, or closed string policy dispatch added.
- This is bounded CPU review evidence, not all-model shards, a GPU benchmark, or a claim of profitable
  trading performance. Remote review/CI still needs the eventual pushed head.
