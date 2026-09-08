# AiDotNet CI Shard Failure Inventory & Work-Division Plan

**Source of truth:** clean `Build & SonarCloud` run on `master` @ `01a4ddb4` (Tensors **0.102.12**), run id `28023414937`.
**Result:** **44 pass / 19 fail / 1 cancelled** (`NeuralNetworks T-Z`, runner lost) of 64 test shards.
**Generated:** 2026-06-23. Re-run the run and re-classify before locking assignments if master has moved.

> **Already fixed (do not re-investigate):** the fp32 small-M GEMM `ArrayPool` host-crash that was aborting every CNN/diffusion shard wholesale — fixed in **Tensors #672 → 0.102.12**, on master via #1667. Verified locally: `ResNetNetworkTests` 56/56, `ConvolutionalNeuralNetworkTests` 21/21.

---

## TL;DR — the 19 failures are 4 distinct causes, not 19 model bugs

| # | Root cause | Shards | One fix greens many? |
|---|-----------|:------:|----------------------|
| ① | **Stack overflow** (host crash, recursion) | 4 (+1 cancelled) | Likely yes — probably one shared recursion |
| ② | **Inference arena corruption** (`#1661`, default-ON) | 3 (+~7 in ③) | **Yes — biggest lever** |
| ③ | **Host killed mid-run** (arena-crash *or* OOM; log truncated) | 8 | Partly (triage → ② or ④/OOM) |
| ④ | **Genuine per-model bugs** (training / clone / construction) | 4 | No — individual |

The arena bug (②) and most of the silent diffusion deaths (③) are the **same root cause** in two manifestations: the arena recycles a tensor across denoise steps → wrong shape (`Input has N channels but layer expects M`). When that corruption is *caught* it's a `[FAIL]` (②); when it corrupts `ArrayPool` it's a **native host crash with no `[FAIL]` logged** (③).

**Confirmed locally (this box, 32 GB):** `AIDOTNET_INFERENCE_ARENA=0` turns **DDPM 38/38** and **SyncDreamer** green (both fail with the arena on). That env var is the **key enabler for parallel work** — see below.

---

## Per-shard classification

### ① Stack overflow — host crash (recursion bug)
`Test host process crashed : Stack overflow.`

- `ModelFamily - NeuralNetworks O-R`
- `ModelFamily - NeuralNetworks S`
- `ModelFamily - NeuralNetworks A-L`
- `ModelFamily - Generated Layers A-M`
- `ModelFamily - NeuralNetworks T-Z` — **cancelled** (runner lost mid-run; no reason logged). Given the other 3 NN ranges all stack-overflow, this is the likely cause; could also be OOM. Triage locally.

*Almost certainly one shared recursion (a `Clone`/forward/property that calls itself). Find it once, likely fixes all five.*

### ② Inference arena corruption (#1661) — confirmed
Caught `Input has N channels but layer expects M` failures + arena signature in the log.

| Shard | Failing model(s) |
|---|---|
| `ModelFamily - Diffusion A-C` | `ControlNetXSModelTests` (all 7) |
| `ModelFamily - Diffusion Stable` | `StableSRModelTests` (all), `StableDiffusion3ModelTests.Predict_ShouldBeDeterministic` |
| `ModelFamily - Diffusion Step-Sync` | `SUPIRModelTests` (all), `SyncDiffusionModelTests`, `Step1XEditModelTests` (training subset) |

### ③ Host killed mid-run — arena-crash OR OOM (needs local triage)
Shard died before xUnit logged any `[FAIL]` (`0 FAILs`, no crash reason captured). Diffusion-heavy → **leading hypothesis: arena escalating to a native pool-corruption crash** (DDPM lives in `03b`, SyncDreamer in `SA-SD` — both confirmed arena locally). Some may be genuine 16 GB-runner OOM.

- `Unit - 03b Diffusion Models Control/DDPM/Preprocessor`  *(DDPM = arena, confirmed)*
- `Unit - 03d Diffusion Models FastGen/Conditioner`
- `ModelFamily - Diffusion D-I`
- `ModelFamily - Diffusion J-M`
- `ModelFamily - Diffusion N-R`
- `ModelFamily - Diffusion SA-SD`  *(SyncDreamer = arena, confirmed)*
- `ModelFamily - Diffusion SE-SP`
- `Integration C`

### ④ Genuine per-model bugs (NOT arena — these are training/clone/construction)
Arena is inference-only; `GradientFlow` / `LossStrictlyDecreases` / `Training_*` failures are real model bugs.

| Shard | Failing test(s) | Smell |
|---|---|---|
| `ModelFamily - NeuralNetworks M-N` | `NeuralTuringMachineTests` — `GradientFlow_ShouldBeNonZeroAndFinite`, `LossStrictlyDecreasesOnMemorizationTask`, `Training_ShouldChangeParameters` | tape/gradient no-op |
| `ModelFamily - Generated Layers N-Z` | `TimeMachineTests` (training ×3), `WhisperTimestampedTests` (`Training_ShouldChangeParameters`, `DifferentInputs_AfterTraining`) | tape/gradient no-op |
| `ModelFamily - Diffusion T-Z` | `TCDModelTests.Clone_ShouldProduceIdenticalOutput` | clone / stale weight-cache |
| `Integration D` | `DefaultConstructionTests.AllDefaultConstructableModels_ShouldConstructWithoutException` | one model ctor throws |

---

## Work-division — 4 parallel streams (non-overlapping files)

> **Enabler for everyone:** run your shard locally with **`AIDOTNET_INFERENCE_ARENA=0`** to strip out the arena corruption (②/③) so your area's *true* residual failures surface. No need to wait for Stream A to merge.
>
> ```
> AIDOTNET_INFERENCE_ARENA=0 dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Debug -f net10.0 \
>   --filter "FullyQualifiedName~<YourModelTests>"
> ```

Tracked as **Epic #1673** with one GitHub issue per chunk:

| Issue | Stream | Scope | Files (no conflict) | Owner |
|---|---|---|---|---|
| **#1668** | **A — Inference arena (#1661)** | Fix cross-step tensor recycling in the denoise loop (a layer's cross-forward cached buffer takes arena memory that `Reset()` recycles → shape aliasing). Or gate the arena off for diffusion denoise until the deep fix lands. **Greens ② + the diffusion half of ③ (~10 shards).** | `src/Diffusion/DiffusionModelBase.cs`, `src/NeuralNetworks/InferenceArenaSettings.cs`, Tensors arena | **Claude** (root-caused) |
| **#1669** | **B — Stack overflow** | Find the recursion crashing the ① shards (`NeuralNetworks A-L/O-R/S/T-Z` + `Generated A-M`). Bisect which model self-recurses (`Clone`, `GetParameters`, property getters, forward). | offending NN/Generated model `.cs` | 1 dev |
| **#1670** | **D — Training bugs** | NeuralTuringMachine + TimeMachine + WhisperTimestamped training/gradient no-op (likely shared sub-layer registration gap). | those 3 model `.cs` | 1 dev |
| **#1671** | **D — Misc bugs** | TCD clone-equivalence + Integration-D default-construction. | TCD model `.cs`, the throwing ctor | 1 dev |
| **#1672** | **C — OOM footprint** | Genuine 16 GB-runner OOM residual; test-scale variants. **Gated on #1668** (triage which ③ shards are truly OOM after the arena fix). | `tests/.../ModelFamilyTests/Base/*TestBase.cs`, model construction | 1 dev |

### Sequencing
1. **A and B first / concurrently** — they are *crash* causes that abort shards and mask everything else.
2. **C and D in parallel now** using `AIDOTNET_INFERENCE_ARENA=0` locally (they don't need A merged to see their failures).
3. After A + B land, **one CI re-run** gives the clean residual; reassign any surprises to D.

### Triage step that splits ③ between A and C (parallelizable, ~1 shard each)
For each ③ shard, run locally with `AIDOTNET_INFERENCE_ARENA=0`:
- **Passes** → it was the arena → covered by **Stream A**.
- **Still fails / OOMs** → **Stream C** (OOM) or **Stream D** (genuine), by signature.

---

## Reproduce / verify

```bash
# the clean inventory run
gh run view 28023414937 --repo ooples/AiDotNet

# a confirmed arena case (passes with arena off, fails with it on)
AIDOTNET_INFERENCE_ARENA=0 dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Debug -f net10.0 \
  --filter "FullyQualifiedName~UnitTests.Diffusion.Models.DDPMModelTests"   # 38/38 with arena off

# a stack-overflow shard (host crash)
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Debug -f net10.0 \
  --filter "FullyQualifiedName~ModelFamilyTests.NeuralNetworks"             # crashes: Stack overflow
```

## Pass list (44) — for reference, don't touch
All non-listed shards passed, including: NN-Classic (ResNet/VGG/DenseNet), NN-Efficient, NN-VLM, all 13 `Unit` non-diffusion shards, `Integration A-B/E-G/H-L/M/N-O/P-Q/R/S/T-Z` (only C and D fail), Classification, Clustering/GP, Regression, TimeSeries, Code/Forecast/Segment/Survival, several `Diffusion Contracts` unit shards (`03a/03c1/03c2/03c4/03c5`), TopLevel, Serving.
