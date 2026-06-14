using System.Linq;
using System.Runtime;
using System.Threading;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Non-generic shim: <c>DiffusionModelTestBase</c> is the existing inheritance
/// target for ~60 diffusion test classes, all of which assume the underlying
/// model is <see cref="IDiffusionModel{T}"/> with <c>T = double</c>. The
/// generic <see cref="DiffusionModelTestBase{TNum}"/> carries all the actual
/// test invariants; this shim binds them to FP64 so existing inheritors keep
/// working with zero call-site changes. Paper-scale diffusion models (e.g.
/// <c>SDXLInpainting</c> at default 320-baseChannels 2048-contextDim UNet,
/// 2.6 B parameters) inherit from <see cref="DiffusionModelTestBase{TNum}"/>
/// with <c>TNum = float</c> directly — FP64 doubles the per-tensor memory
/// footprint and pushes the 2.6 B-param SDXL UNet past the 16 GB RAM ceiling
/// of CI hosts (verified at construction: <c>SDXLInpaintingModel&lt;double&gt;</c>
/// OOMs in the 1.28 B-element kernel allocation; <c>SDXLInpaintingModel&lt;float&gt;</c>
/// fits in ~3.8 GB managed heap). FP32 is also the production-canonical type
/// for diffusion-model weights (SD/SDXL/Flux/SD3 paper checkpoints are FP32
/// master / FP16 working), so paper-scale tests using <c>&lt;float&gt;</c> mirror
/// the actual deployment configuration rather than an FP64 test-only path
/// that would be silently incorrect.
/// </summary>
public abstract class DiffusionModelTestBase : DiffusionModelTestBase<double>
{
}

/// <summary>
/// Base test class for diffusion models implementing IDiffusionModel&lt;double&gt;.
/// Tests mathematical invariants: denoising convergence, output sensitivity,
/// training stability, scheduler consistency, and noise schedule properties.
/// </summary>
/// <remarks>
/// Implements <see cref="IAsyncLifetime"/> to force a full GC cycle between
/// tests. Without this hint, sequential Diffusion tests on 16 GB Windows CI
/// runners accumulate undisposed weight-tensor backing arrays that sit in
/// gen-2 heap — the ~255-test Diffusion shards hit OOM within 45 min
/// wall-clock because GC never has time to run a compacting collection.
/// See issue #1136. The forced <c>GC.Collect → WaitForPendingFinalizers → GC.Collect</c>
/// sequence (standard two-pass pattern) runs AFTER each test disposes its
/// model (via <c>using var model = CreateModel()</c>), reclaiming the rented
/// weight buffers returned to the TensorAllocator pool on Dispose.
/// </remarks>
public abstract class DiffusionModelTestBase<TNum> : IAsyncLifetime
    where TNum : struct, IEquatable<TNum>, IFormattable
{
    /// <summary>
    /// Static lock serializing concurrent teardowns. xunit parallelizes across
    /// test-classes by default, so two derived test classes can hit
    /// <see cref="DisposeAsync"/> concurrently on different threads.
    /// <see cref="GCSettings.LargeObjectHeapCompactionMode"/> is process-
    /// global, so concurrent toggles race. Serializing the whole
    /// mode-set → collect → wait → mode-set → collect sequence keeps LOH
    /// compaction deterministic per teardown.
    /// </summary>
    private static readonly object _lohCompactionGate = new();

    /// <summary>
    /// Caps concurrent foundation-scale diffusion tests to avoid BLAS thread-
    /// pool oversubscription when many SD-UNet-scale Predicts run in parallel
    /// on the same machine. xUnit's parallelizeTestCollections=true puts one
    /// test class per core; if 16 of them are simultaneously inside an FP64
    /// SD-UNet forward (each wanting all 16 cores via OpenBLAS), every test
    /// gets ~1 core and the per-step latency multiplies by 4-8×, blowing the
    /// 120 s <c>[Fact(Timeout)]</c> envelope even though each test fits the
    /// budget in isolation. See <c>tools/ConsistencyModelPerfDiag</c> for the
    /// measurement that motivated this (issue #1305 ConsistencyModel:
    /// 76 s isolated vs 120 s under-contention timeout).
    /// </summary>
    private const int HeavyConcurrencyCap = 2;

    /// <summary>
    /// Element-count threshold above which a test counts as "heavy" and gates
    /// through <see cref="_heavyTestGate"/>. 16,384 = the latent-shape product
    /// for the canonical SD pipeline at [1, 4, 64, 64]; everything at or above
    /// this scale uses an SD-UNet-class noise predictor whose FP64 forward
    /// saturates the BLAS thread pool. Smaller-scale diffusion tests (tabular
    /// [1, 4] or single-channel [1, 1, 16, 16]) bypass the gate so they can
    /// stay fully parallel.
    /// </summary>
    private const int HeavyInputElementThreshold = 16_384;

    private static readonly SemaphoreSlim _heavyTestGate =
        new(HeavyConcurrencyCap, HeavyConcurrencyCap);

    /// <summary>
    /// Per-test-instance flag tracking whether this instance acquired
    /// <see cref="_heavyTestGate"/>. Only released in <see cref="DisposeAsync"/>
    /// if acquired here, so a failure during InitializeAsync (gate not yet
    /// acquired) can't trigger a release-without-acquire.
    /// </summary>
    private bool _heavyGateAcquired;

    /// <summary>
    /// Before-test hook. Acquires the heavy-diffusion gate if this test's
    /// <see cref="InputShape"/> implies foundation-scale (≥ 16,384 elements).
    /// </summary>
    public async Task InitializeAsync()
    {
        if (IsHeavyScale(InputShape))
        {
            await _heavyTestGate.WaitAsync().ConfigureAwait(false);
            _heavyGateAcquired = true;
        }
    }

    /// <summary>
    /// True when the supplied input shape implies foundation-scale work
    /// (SD-UNet or larger). Computed from product-of-dims so a future
    /// [1, 16, 32, 32] (Flux/SD3) or [1, 8, 64, 64] (CogVideo) shape
    /// auto-gates without any per-class plumbing.
    /// </summary>
    private static bool IsHeavyScale(int[] shape)
    {
        if (shape is null || shape.Length == 0) return false;
        long elements = 1;
        foreach (int d in shape)
        {
            if (d <= 0) return false;
            elements *= d;
            if (elements >= HeavyInputElementThreshold) return true;
        }
        return false;
    }

    /// <summary>
    /// After-test hook. Forces a blocking compacting Gen-2 GC (with explicit
    /// Large Object Heap compaction) to reclaim weight tensors that the
    /// <c>using var model</c> Dispose released AND defragment the LOH
    /// between tests.
    /// </summary>
    /// <remarks>
    /// Diffusion model weight tensors are typically several hundred MB each,
    /// well above the 85KB LOH threshold. Plain <see cref="GC.Collect"/> sweeps
    /// the LOH but does NOT compact it — over hundreds of sequential tests,
    /// LOH fragmentation accumulates until the next allocation can't find a
    /// contiguous region even when total free bytes remain large, producing
    /// <see cref="OutOfMemoryException"/>. Setting
    /// <see cref="GCLargeObjectHeapCompactionMode.CompactOnce"/> on the next
    /// Gen-2 pass forces LOH compaction; the mode auto-resets to Default
    /// after each use, so this is scoped per-teardown. The entire sequence
    /// runs under <see cref="_lohCompactionGate"/> so parallel teardowns
    /// don't race on the process-global flag.
    /// </remarks>
    public Task DisposeAsync()
    {
        try
        {
            lock (_lohCompactionGate)
            {
                // First pass: compacting Gen-2 + LOH reclaims everything unreachable
                // including the just-Disposed model's weight tensors.
                GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
                GC.Collect(generation: 2, mode: GCCollectionMode.Forced, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();

                // Second pass: finalizer-released memory (e.g. GPU-pool return paths)
                // and any LOH allocations from finalizers.
                GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
                GC.Collect(generation: 2, mode: GCCollectionMode.Forced, blocking: true, compacting: true);
            }
        }
        finally
        {
            if (_heavyGateAcquired)
            {
                _heavyTestGate.Release();
                _heavyGateAcquired = false;
            }
        }
        return Task.CompletedTask;
    }


    protected abstract IDiffusionModel<TNum> CreateModel();

    /// <summary>
    /// Numeric-operation handle for <typeparamref name="TNum"/> — used by helpers
    /// that need to construct tensor elements from doubles (e.g. random fills,
    /// constant fills) without hard-coding the element type. The
    /// double-precision codebase pattern is direct assignment to a double; the
    /// generic equivalent goes through <see cref="INumericOperations{T}.FromDouble"/>.
    /// </summary>
    protected static readonly INumericOperations<TNum> _numOps =
        MathHelper.GetNumericOperations<TNum>();

    /// <summary>
    /// Converts a <typeparamref name="TNum"/> tensor element to <c>double</c>
    /// for assertion / Math.* arithmetic. Centralized so every test method
    /// uses the same boundary conversion (rather than scattering
    /// <c>Convert.ToDouble</c> calls). The runtime <c>(double)(object)val</c>
    /// boxing fallback is the standard pattern for generic numeric →
    /// well-known-type conversion in this codebase (see
    /// <c>ModelTestHelpers.cs</c> for the broader pattern).
    /// </summary>
    protected static double ToDouble(TNum v) => Convert.ToDouble(v);

    protected virtual int[] InputShape => [1, 4];
    protected virtual int[] OutputShape => [1, 4];

    /// <summary>
    /// Number of training iterations used by post-training invariants. Virtual
    /// so paper-scale Foundation models can override down to fit the xunit
    /// 120s per-test timeout. Default chosen to match the existing baseline
    /// in <see cref="Training_ShouldReducePredictionError"/>.
    /// </summary>
    protected virtual int TrainingIterations => 10;

    protected Tensor<TNum> CreateRandomTensor(int[] shape, Random rng)
    {
        var tensor = new Tensor<TNum>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = _numOps.FromDouble(rng.NextDouble());
        return tensor;
    }

    protected Tensor<TNum> CreateConstantTensor(int[] shape, double value)
    {
        var tensor = new Tensor<TNum>(shape);
        var typed = _numOps.FromDouble(value);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = typed;
        return tensor;
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Training Should Reduce Prediction Error
    //
    // Per DDPM (Ho et al. 2020, Algorithm 1), diffusion training minimizes the
    // MEAN squared error between the true noise ε and the model's predicted noise
    // ε_θ(√ᾱₜ·x₀ + √(1−ᾱₜ)·ε, t). There is NO supervised "target output" in
    // diffusion — the data point is the clean sample x₀, noise is added, and the
    // model predicts that noise. So the valid, paper-faithful "training reduces
    // error" check is: does the noise-prediction MSE at a FIXED probe (x₀, ε, t)
    // go down (or at least not up) after training?
    //
    // The earlier formulation measured MSE(Generate(x₀), random_target). That is
    // not a paper quantity and is causally unrelated to the training objective:
    // `Train` ignores the target argument (it self-samples noise, exactly per the
    // paper), so Generate(x₀) drifts toward the model's own learned denoising
    // fixed point — away from an arbitrary random target — for ANY model whose
    // sampler genuinely depends on its weights. Expressive models (e.g. the
    // attention-UNet Imagen2 config) therefore "failed" that check while training
    // perfectly correctly (output bounded, converges to a fixed point). The probe
    // below measures the actual objective and only fails if training makes
    // noise-prediction WORSE — a real gradient-sign / divergence bug.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task Training_ShouldReducePredictionError()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        // Treat the random tensor as the clean sample x₀ (the diffusion data point).
        var x0 = CreateRandomTensor(InputShape, rng);

        // Fixed noise-prediction probe held constant across the before/after
        // measurement: a single (noise ε, timestep t) so we compare like-for-like.
        // (Per-step Train uses a random t internally, so raw per-step loss values
        // are confounded by timestep magnitude; a fixed probe is not.) Mid-range t.
        int probeT = System.Math.Max(1, model.Scheduler.TrainTimesteps / 2);
        var probeNoiseVec = new Vector<TNum>(x0.Length);
        for (int i = 0; i < probeNoiseVec.Length; i++)
            probeNoiseVec[i] = _numOps.FromDouble(rng.NextDouble() * 2.0 - 1.0);
        var noisyProbe = new Tensor<TNum>(x0._shape, model.Scheduler.AddNoise(x0.ToVector(), probeNoiseVec, probeT));
        var probeNoise = new Tensor<TNum>(x0._shape, probeNoiseVec);

        double errBefore = ComputeMSE(model.PredictNoise(noisyProbe, probeT), probeNoise);

        // Train on x₀ (the diffusion data point). The target argument is unused by
        // the diffusion training path per the paper; pass x₀ for clarity.
        for (int i = 0; i < TrainingIterations; i++)
            model.Train(x0, x0);

        double errAfter = ComputeMSE(model.PredictNoise(noisyProbe, probeT), probeNoise);

        if (!double.IsNaN(errBefore) && !double.IsNaN(errAfter))
        {
            Assert.True(errAfter <= errBefore + 1e-6,
                $"Training increased the noise-prediction error: before={errBefore:F6}, after={errAfter:F6}. " +
                "DDPM training (Ho et al. 2020, Alg. 1) minimizes MSE(ε, ε_θ); after training on x₀ the model " +
                "must predict the noise at a fixed (x₀, ε, t) probe at least as well as before — an increase " +
                "indicates a gradient-sign error, divergence, or first-step explosion in the training path.");
        }
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Sensitivity to Input
    // Different inputs must produce different outputs. A model that
    // ignores its input is fundamentally broken.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input1 = CreateConstantTensor(InputShape, 0.1);
        var input2 = CreateConstantTensor(InputShape, 0.9);

        var output1 = model.Predict(input1);
        var output2 = model.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ToDouble(output1[i]) - ToDouble(output2[i])) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Model produces identical output for inputs [0.1,...] and [0.9,...]. Input is being ignored.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Scaled Input Changes Output
    // f(x) ≠ f(10x) — the model should be sensitive to magnitude.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ScaledInput_ShouldChangeOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var input = CreateRandomTensor(InputShape, rng);
        var scaledInput = new Tensor<TNum>(InputShape);
        for (int i = 0; i < input.Length; i++)
            scaledInput[i] = _numOps.FromDouble(ToDouble(input[i]) * 10.0);

        var output1 = model.Predict(input);
        var output2 = model.Predict(scaledInput);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ToDouble(output1[i]) - ToDouble(output2[i])) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Output unchanged when input scaled 10x. Forward pass may ignore input values.");
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Output Shape Preservation
    // For a diffusion model, output shape should match input shape
    // (denoising maps noisy input → clean output of same dimensions).
    // =====================================================

    [Fact(Timeout = 120000)]
    public virtual async Task OutputShape_ShouldMatchInputShape()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output = model.Predict(input);
        Assert.Equal(input.Length, output.Length);
    }

    // =====================================================
    // MATHEMATICAL INVARIANT: Finite Output Before and After Training
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ForwardPass_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        Assert.True(output.Length > 0, "Output should not be empty.");
        for (int i = 0; i < output.Length; i++)
        {
            var v = ToDouble(output[i]);
            Assert.False(double.IsNaN(v), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(v), $"Output[{i}] is Infinity.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ForwardPass_ShouldBeFinite_AfterTraining()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);

        // Half-iterations baseline: a sanity probe that finite-output holds
        // even before full training has converged.
        int finiteCheckIters = Math.Max(1, TrainingIterations / 2);
        for (int i = 0; i < finiteCheckIters; i++)
            model.Train(input, target);

        var output = model.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            var v = ToDouble(output[i]);
            Assert.False(double.IsNaN(v), $"Output[{i}] is NaN after training.");
            Assert.False(double.IsInfinity(v), $"Output[{i}] is Infinity after training.");
        }
    }

    // =====================================================
    // DIFFUSION INVARIANT: Noise Schedule Monotonicity
    // At higher timesteps, the noise magnitude should increase.
    // This verifies the noise schedule is properly configured.
    // =====================================================

    [Fact(Timeout = 120000)]
    public virtual async Task NoiseSchedule_ShouldBeMonotonic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        // The noise schedule is the SCHEDULER's signal-retention curve: the cumulative product
        // of alphas (the fraction of the original signal retained at timestep t) must not
        // increase as t grows, because every step only adds noise. Equivalently the noise
        // fraction 1 - alpha_cumprod is monotonically non-decreasing. That is the actual
        // "noise schedule is monotonic" invariant this test is named for, and it holds for
        // every correctly-configured scheduler (DDPM/cosine/linear/flow-matching).
        //
        // (The previous proxy scaled the model's INPUT and checked the OUTPUT magnitude. That
        // is architecturally invalid for input-normalizing noise predictors such as DiT — its
        // LayerNorm removes the input scale, so the denoised sample's magnitude is independent
        // of the input scale by design. Every DiT-based diffusion model failed the old proxy
        // for that reason, not because of a real bug.)
        var scheduler = model.Scheduler;
        int n = scheduler.TrainTimesteps;
        if (n < 2) return;

        double prevSignal = ToDouble(scheduler.GetAlphaCumulativeProduct(0));
        int violations = 0;
        double worst = 0;
        for (int t = 1; t < n; t++)
        {
            double signal = ToDouble(scheduler.GetAlphaCumulativeProduct(t));
            double increase = signal - prevSignal;
            if (increase > 1e-9) { violations++; worst = Math.Max(worst, increase); }
            prevSignal = signal;
        }

        Assert.True(violations == 0,
            $"Noise schedule not monotonic: alpha-cumulative-product (signal retention) increased " +
            $"with timestep at {violations} step(s) (worst +{worst:E3}); it must be non-increasing " +
            "as noise accumulates.");
    }

    // =====================================================
    // DIFFUSION INVARIANT: Output Range Validity
    // Generated output should be bounded — no exploding values.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputRange_ShouldBeValid()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var output = model.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            var v = ToDouble(output[i]);
            Assert.True(Math.Abs(v) < 1e6,
                $"Output[{i}] = {v:E4} exceeds bound of 1e6. " +
                "Diffusion model is producing unbounded output.");
        }
    }

    // =====================================================
    // BASIC CONTRACTS: Determinism, Clone, Metadata, Parameters, Scheduler
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var out1 = model.Predict(input);
        var out2 = model.Predict(input);

        for (int i = 0; i < out1.Length; i++)
            Assert.Equal(ToDouble(out1[i]), ToDouble(out2[i]), 12); // Tensors 0.16.0 deterministic BLAS — exact match expected
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldProduceIdenticalOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        var original = model.Predict(input);
        var cloned = model.Clone();
        var clonedOutput = cloned.Predict(input);

        Assert.Equal(original.Length, clonedOutput.Length);
        for (int i = 0; i < original.Length; i++)
            Assert.Equal(ToDouble(original[i]), ToDouble(clonedOutput[i]));
    }

    [Fact(Timeout = 120000)]
    public async Task Metadata_ShouldExist()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTensor(OutputShape, rng);
        model.Train(input, target);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();
        // Check ParameterCount rather than GetParameters().Length — both answer the
        // same question ("does the model have learnable parameters?") but
        // ParameterCount reads the declared count without forcing lazy layers to
        // materialize their weight tensors (which at DiT-XL scale is ~4 GB and
        // OOMs CI runners just for an existence check).
        Assert.True(model.ParameterCount > 0,
            "Diffusion model should have learnable parameters.");
    }

    [Fact(Timeout = 120000)]
    public async Task Scheduler_ShouldBeNonNull()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();
        Assert.NotNull(model.Scheduler);
    }

    private double ComputeMSE(Tensor<TNum> output, Tensor<TNum> target)
    {
        double mse = 0;
        int len = Math.Min(output.Length, target.Length);
        if (len == 0) return double.NaN;
        for (int i = 0; i < len; i++)
        {
            double diff = ToDouble(output[i]) - ToDouble(target[i]);
            mse += diff * diff;
        }
        return mse / len;
    }
}
