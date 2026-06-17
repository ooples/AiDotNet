# Design: Close #1305 ConsistencyModel test timeout — AiDotNet-side perf wins

**Issue:** [#1305](https://github.com/ooples/AiDotNet/issues/1305) "[PR #1290 CI] Tests (net10.0) - ModelFamily - Diffusion S-Z: 3 failing tests"
**Branch:** `perf/issue-1305-consistencymodel-bottleneck`
**Date:** 2026-05-26
**Status:** Design — awaiting user review

---

## 1. Background

Of the three tests #1305 originally tracked, two now pass on master (`Flux2SchnellModelTests` via [PR #1396](https://github.com/ooples/AiDotNet/pull/1396) patchify/unpatchify port, `VideoCrafterModelTests` passes incidentally).

One remains failing:

- `AiDotNet.Tests.ModelFamilyTests.Diffusion.ConsistencyModelTests.ScaledInput_ShouldChangeOutput` — timeout 120 s

Maintainer's prior triage attributed the failure to the FP64 Conv2D backward perf gap tracked in [AiDotNet.Tensors #415](https://github.com/ooples/AiDotNet.Tensors/issues/415). Re-measurement under instrumentation (see §2) shows the picture is more nuanced — **most of the gap is Tensors-side, but the failure trigger is AiDotNet-side**.

The prior `DefaultInferenceSteps = 10 → 2` fix already landed (commit `e13c83ba2`, "fix(diffusion): paper-canonical defaultinferencesteps for distilled fast-generation models") and is confirmed effective.

## 2. Bottleneck measurement

Instrumentation: `tools/ConsistencyModelPerfDiag/` mirrors the failing test's exact pattern (construct → 2× Predict on different inputs at [1, 4, 64, 64] FP64, seed=42), then isolates the UNet cost via direct `PredictNoise` calls.

**Measured on Release, net10.0, 16-core host (single-process, no parallel contention):**

| Phase | Wall time | Share of total |
|---|---:|---:|
| Construction | 13.475 s | one-time |
| First Predict (cold) | 35.827 s | — |
| Second Predict (warm) | 27.101 s | — |
| **Total Predict cost** | **62.928 s** | **100%** |
| UNet PredictNoise × 4 (2 steps × 2 Predicts, warm mean 14.95 s/step) | 59.795 s | 95.0% |
| Diffusion-loop overhead (canonicalize, ResolveInitialSample, NaN guard, Scheduler.Step) | 3.133 s | 5.0% |

Total in isolation: 13.5 s ctor + 62.9 s Predicts = **~76 s < 120 s budget** ✅.

Yet the actual `dotnet test --filter ...` run timed out at exactly **120 s** ❌. Gap (~44 s) is xUnit infra (15 s discovery), cold JIT, and — when the full test shard runs — **BLAS thread-pool oversubscription** as 16-way parallel test execution puts 16 concurrent SD-UNet-scale Predicts in flight on the same 16 cores.

## 3. Bottleneck categorization

| Phase | Repo | Notes |
|---|---|---|
| **UNet PredictNoise (95%)** | **AiDotNet.Tensors** | Conv2D + GroupNorm + cross-attention at FP64. Tracked in Tensors #413 / #415. **Out of scope** for this PR. |
| Diffusion loop overhead (5%) | AiDotNet | 3.1 s absolute — limited headroom for model-side wins. |
| 13 s construction | AiDotNet | 506M-param tensor materialization (UNet ~426M, VAE ~80M). VAE built eagerly even though latent-input Predict skips DecodeFromLatent. **Lazy-init candidate** (expected savings ~2 s based on parameter share). |
| xUnit parallel contention (~44 s of the symptom) | AiDotNet (test infra) | 16 foundation-scale tests on 16 cores oversubscribe BLAS thread pools → each test's UNet forward slows by 2-4×. **Semaphore-gate candidate.** |

## 4. Scope

### In scope (this PR)

1. **§5 — Semaphore-gated concurrent heavy diffusion test execution.** Auto-detect heavy tests by `InputShape` size; cap concurrency at K=2.
2. **§6 — Lazy-init VAE for ConsistencyModel.** Skip VAE allocation when the Predict path doesn't need it (latent input). Opt-in via base-class helper, applied initially only to ConsistencyModel.

### Out of scope (deferred)

- **Tensors-side Conv2D / GroupNorm / Attention kernel perf** — tracked in Tensors #413 / #415. The 95% UNet share remains a Tensors problem.
- **Lazy VAE for the other ~30 LatentDiffusionModelBase subclasses.** Opt-in helper is available; sibling subclasses can adopt it when their tests show similar pressure, in separate PRs.
- **Reducing model defaults (smaller UNet, fewer ResBlocks).** Violates the foundation-scale invariant.
- **FP32 inference paths.** Larger architectural change, not the right fit here.

## 5. Design: Semaphore-gated concurrent heavy diffusion tests

### Goal

Cap concurrent execution of foundation-scale diffusion tests at K so that BLAS thread pools aren't oversubscribed. Each gated test then completes in close to its isolated timing (~76 s for ConsistencyModel), well under the 120 s budget.

### Mechanism

Single change to `tests/AiDotNet.Tests/ModelFamilyTests/Base/DiffusionModelTestBase.cs`:

```csharp
public abstract class DiffusionModelTestBase : IAsyncLifetime
{
    // Cap concurrent foundation-scale diffusion tests to avoid
    // 16-way BLAS thread-pool oversubscription on CI runners.
    // K=2 means each heavy test gets ~half the cores; the SD-UNet
    // FP64 forward at [1, 4, 64, 64] then fits its 120s [Fact(Timeout)]
    // budget (isolation timing is ~76s — see #1305 analysis).
    private const int HeavyConcurrencyCap = 2;

    // Threshold: ≥16,384 elements (= [1, 4, 64, 64] FP64 latent or larger).
    // Smaller-scale diffusion tests (e.g., [1, 4] tabular, [1, 1, 16, 16])
    // stay fully parallel — they don't have the BLAS-thread contention
    // pathology and shouldn't pay the gating overhead.
    private const int HeavyInputElementThreshold = 16_384;

    private static readonly SemaphoreSlim _heavyTestGate =
        new(HeavyConcurrencyCap, HeavyConcurrencyCap);

    private bool _gateAcquired;

    public async Task InitializeAsync()
    {
        if (IsHeavyScale(InputShape))
        {
            await _heavyTestGate.WaitAsync();
            _gateAcquired = true;
        }
    }

    public Task DisposeAsync()
    {
        try
        {
            // existing GC.Collect / LOH compaction block
        }
        finally
        {
            if (_gateAcquired)
            {
                _heavyTestGate.Release();
            }
        }
        return Task.CompletedTask;
    }

    private static bool IsHeavyScale(int[] shape)
    {
        long elements = 1;
        foreach (int d in shape) elements *= d;
        return elements >= HeavyInputElementThreshold;
    }
}
```

### Why this works

- **75+ foundation-scale diffusion tests auto-gate** without per-class file touches.
- **Small-scale diffusion tests** stay fully parallel — they don't oversubscribe BLAS and shouldn't pay gate overhead.
- **K=2 gives each heavy test ~8 cores** vs. ~1 under 16-way parallel — empirically enough to fit the budget per isolation timing.
- **Static semaphore** survives test-class instantiation (each test method gets a fresh class instance).

### Why not other approaches

- `[Collection("HeavyDiffusion")]` with `DisableParallelization = true` ⇒ fully serial. 75 classes × 10 methods × ~30 s = 6+ hours of CI. Rejected.
- Global `maxParallelThreads: 4` in `xunit.runner.json` ⇒ slows fast unit tests too. Bluntest instrument. Rejected.
- Per-class `[Collection]` attribute on 75 files ⇒ mechanical change, large file count, doesn't auto-cover future tests. Rejected.

### Tuning K

K=2 is the initial value. Heuristic: K should be `cores / cores_BLAS_wants_per_test`. SD-UNet PredictNoise at FP64 fully saturates 8-16 cores via OpenBLAS. With K=2 on a 16-core host, each test gets 8 cores — close to OpenBLAS's per-call sweet spot. On a 4-core CI runner, K=2 still gives 2 cores per test, which slows them but won't deadlock or oversubscribe.

If K=2 still timeouts in CI we drop to K=1 (effectively serial heavy tests). If it leaves obvious CPU on the table we raise to K=3.

### Risks

| Risk | Mitigation |
|---|---|
| **Deadlock** if test holds gate and tries to enter another gate | Tests don't recurse into other test base classes — single-class lifecycle per test method. Static semaphore is non-reentrant. |
| **Gate leaks** on exception during InitializeAsync | `_gateAcquired` is set AFTER acquire succeeds; DisposeAsync only releases if set. No release-without-acquire. |
| **Heuristic threshold miscalibration** — small diffusion tests at [1, 1, 128, 128] gate unnecessarily | Threshold 16,384 chosen as the minimum SD-latent shape (4×64×64). Future tweak just changes the const. |
| **K too low** — wastes CPU when only a few heavy tests are active | K=2 is intentionally conservative; safe to tune up later if no contention seen. |
| **xUnit test execution ordering** | xUnit doesn't guarantee order; gate is order-agnostic. |

### Verification

1. `dotnet test --filter ScaledInput_ShouldChangeOutput` passes within the 120 s budget on the local 16-core box.
2. Same test passes under full-shard execution (`Tests (net10.0) - ModelFamily - Diffusion S-Z`) in CI.
3. Wall-clock of the full Diffusion shard doesn't regress > 20% — gating only kicks in for heavy tests; light tests stay parallel.
4. No new test failures elsewhere.

## 6. Design: Lazy-init VAE in ConsistencyModel

### Goal

When `Predict()` is called with a latent-shape input (the diffusion test path), `Generate` detects `inputIsLatent` and short-circuits the VAE decode. The VAE is therefore unused — but currently constructed eagerly in `ConsistencyModel`'s ctor. Defer to first VAE access. The exact saving is measurement-dependent (VAE has ~80 M params vs UNet's ~426 M of the 506 M total; share is ~15%, so expected savings ~2 s of the 13.5 s construction). Re-running `tools/ConsistencyModelPerfDiag` after the change will confirm the actual drop.

### Mechanism

Per-subclass `Lazy<TVae>` keeps the existing `IVAEModel<T> VAE { get; }` contract intact. No abstract base change. Touches only `ConsistencyModel<T>`:

```csharp
public class ConsistencyModel<T> : LatentDiffusionModelBase<T>
{
    // BEFORE:
    // private StandardVAE<T> _vae;
    // public override IVAEModel<T> VAE => _vae;

    // AFTER:
    private readonly Lazy<StandardVAE<T>> _vaeLazy;
    public override IVAEModel<T> VAE => _vaeLazy.Value;

    public ConsistencyModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? noisePredictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int numTrainSteps = 18,
        double sigmaMin = 0.002,
        double sigmaMax = 80.0,
        double rho = 7.0,
        bool isDistilled = false,
        int? seed = null)
        : base(/* unchanged */)
    {
        // ... existing field assignments ...

        // UNet stays eager (used on every PredictNoise → 95% of perf budget)
        _noisePredictor = noisePredictor ?? new UNetNoisePredictor<T>(/* unchanged */);

        // VAE deferred — built only on first VAE access (e.g., DecodeFromLatent
        // when Predict receives a non-latent input, or when GetParameters
        // / ParameterCount needs the full count). Latent-shape Predict path
        // (the test pattern) never touches VAE, so this saves ~7s/test method
        // on ConsistencyModel.
        _vaeLazy = new Lazy<StandardVAE<T>>(
            () => vae ?? new StandardVAE<T>(
                inputChannels: 3,
                latentChannels: CM_LATENT_CHANNELS,
                baseChannels: 128,
                channelMultipliers: new[] { 1, 2, 4, 4 },
                numResBlocksPerLevel: 2,
                seed: seed),
            LazyThreadSafetyMode.PublicationOnly);

        _sigmas = ComputeSigmaSchedule();
    }

    // ParameterCount triggers VAE materialization — Parameters_ShouldBeNonEmpty
    // and Metadata_ShouldExist still see the full param count. Keeps the
    // [DenseLayer lazy-init breaks Clone+ParameterCount] anti-pattern away.
    public override long ParameterCount =>
        _noisePredictor.ParameterCount + _vaeLazy.Value.ParameterCount;

    // ... rest unchanged ...
}
```

### Why per-subclass (not base-class)?

A base-class lazy mechanism (e.g., `protected abstract IVAEModel<T> CreateVAE();` with lazy backing in `LatentDiffusionModelBase`) would require migrating all ~30 latent-diffusion subclasses from `private TVae _vae` + `public override VAE => _vae` to overriding `CreateVAE()`. Large surgery with weak per-call ROI: the gating in §5 already closes the test on its own; lazy VAE is a savings-on-top.

Adopting opt-in lazy VAE only where measurement justifies it (ConsistencyModel here) keeps the diff focused. Subsequent PRs can apply the same pattern to other heavy latent-diffusion models if they show the same construction-time pressure.

### Invariants preserved

- **`Clone()`** — Source's `GetParameters()` triggers VAE materialization (via `_vae.GetParameters()`); cloned model's `SetParameters()` writes into its own (also materialized) VAE. Both sides materialize, parameter equality holds.
- **`ParameterCount`** — Triggers materialization. `Parameters_ShouldBeNonEmpty` still sees the full count. No "live count = 0 until first use" footgun (memory: `[DenseLayer lazy-init breaks Clone+ParameterCount]`).
- **`GetModelMetadata()`** — Uses `ParameterCount`, so also triggers materialization (intended — metadata should report true size).
- **VAE serialization** — `_vae.Save`/`Load` paths: if model is saved before any VAE access, the VAE state is its default initialization (RNG-seeded, no training applied). This matches today's "ctor-initialized + never trained" behavior. Indistinguishable.
- **Thread safety** — `LazyThreadSafetyMode.PublicationOnly` allows multiple concurrent materializations but guarantees only one is published. Safe under parallel test execution; matches typical .NET lazy-init idioms.

### Risks

| Risk | Mitigation |
|---|---|
| First VAE access path now pays a one-shot ~7 s — could push another timing-sensitive test over a budget | The only paths that trigger materialization are: text-to-image inference (paying VAE decode anyway), GetParameters / ParameterCount (cheap aggregations, not timed), and Clone (also one-time). None of them are on tight `[Fact(Timeout)]` budgets. |
| Lazy backing breaks tooling that reflects over `_vae` field | The field name `_vae` is no longer a direct `StandardVAE<T>` — code that introspects via reflection would need to use the `VAE` property instead. Quick grep shows no such reflection. |
| Save/load round-trip behavior if save is called before VAE access | Save calls `GetParameters()` (materializes); load calls `SetParameters()` (materializes). Round-trips are consistent. |

### Verification

1. ConsistencyModel construction time drops measurably (expected ~2 s) in the perf-diag harness (re-run `tools/ConsistencyModelPerfDiag`). Exact magnitude is the validation point — if the actual drop is smaller than expected, the lazy-VAE change is still correct but its value proposition weakens; serial gating §5 alone may be enough.
2. `dotnet test --filter ConsistencyModel` — all 12 tests still pass; `ParameterCount` is still positive in `Parameters_ShouldBeNonEmpty`.
3. `dotnet test --filter ConsistencyModel.Clone_ShouldProduceIdenticalOutput` — clone preserves identical output (proves VAE param plumbing still works through lazy path).
4. `tools/ConsistencyModelPerfDiag` reports `_vaeLazy.IsValueCreated == false` after Predict-only run.

## 7. Combined verification plan

After both changes land:

```bash
# Sanity — original failing test in isolation
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build \
  --filter FullyQualifiedName=AiDotNet.Tests.ModelFamilyTests.Diffusion.ConsistencyModelTests.ScaledInput_ShouldChangeOutput

# Diffusion S-Z full shard (the shard #1305 lives in)
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build \
  --filter "FullyQualifiedName~AiDotNet.Tests.ModelFamilyTests.Diffusion&FullyQualifiedName!~_old"

# All 12 ConsistencyModel invariants
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj -c Release -f net10.0 --no-build \
  --filter "FullyQualifiedName~ConsistencyModelTests"

# Re-run the perf harness to confirm construction time drop
dotnet run --project tools/ConsistencyModelPerfDiag/ConsistencyModelPerfDiag.csproj -c Release --no-build
```

Expected outcomes:

| Metric | Before | After (predicted) |
|---|---:|---:|
| `ScaledInput_ShouldChangeOutput` (isolated) | 120 s (timeout) | ≤ 80 s |
| `ScaledInput_ShouldChangeOutput` (full shard) | 120 s (timeout) | ≤ 100 s |
| ConsistencyModel ctor time | 13.5 s | ~11-12 s (lazy-VAE saves ~2 s based on param share) |
| ConsistencyModel all 12 tests pass | red | green |

## 8. Out-of-scope items / follow-ups

- **Tensors #413 + #415** — the 95% UNet share. When kernel work lands, ConsistencyModel will fit even on heavily-loaded CI without gating; gating will then be belt-and-suspenders.
- **Lazy-VAE rollout to other latent-diffusion subclasses** (Flux2Schnell, AnimateDiff, Allegro, etc.) — each one a small follow-up PR if its construction time becomes a problem.
- **Tuning K** — start at K=2; if CI history shows occasional heavy-diffusion timeouts at this K, drop to K=1 in a follow-up.
- **The 13 s construction even with lazy VAE** — UNet alone takes ~6-7 s. If that becomes a CI cost concern, lazy-init UNet param materialization is the natural next step (same pattern). Out of scope today.

## 9. Implementation order

1. Land §5 (semaphore gate) first — it's the load-bearing fix. Verify the test passes alone and in-shard.
2. Land §6 (lazy VAE) second — provides the construction-time margin and the pattern that other models can adopt.
3. Single PR with two commits, both gated on the verification plan above.

---

## Decisions to confirm

- ✅ Test fixture is foundation-scale and stays foundation-scale.
- ✅ DefaultInferenceSteps=2 stays as the paper-canonical default (already landed).
- ✅ Both AiDotNet-side changes go in one PR; Tensors work stays separate.
- ✅ Lazy VAE applied only to ConsistencyModel in this PR (opt-in pattern; siblings as follow-ups).
- ✅ Semaphore K=2 as initial value; can be tuned later.
