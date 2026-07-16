using System;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.DistributedTraining;

/// <summary>
/// Tests for the ZeRO-Offload / FSDP CPUOffload equivalent flags on
/// <see cref="IShardingConfiguration{T}"/> and the runtime contract enforced
/// by <c>ShardedOptimizerBase.RunWrappedOptimizerStep</c>.
/// </summary>
/// <remarks>
/// The engine SWAP is validated against a genuine non-CPU outer engine (a
/// <see cref="DispatchProxy"/> that is not a <see cref="CpuEngine"/> and throws if any
/// tensor op is invoked on it — the swap only does an <c>is CpuEngine</c> check and an
/// identity comparison, never calls ops): the wrapped step must observe a
/// <see cref="CpuEngine"/>, and the EXACT outer instance must be restored afterwards,
/// including when the wrapped step throws.
///
/// All timed facts are <c>async Task</c>: xUnit's <c>Timeout</c> cannot reliably interrupt
/// a synchronous <c>void</c> test (it can only cancel between awaits), so the timeout would
/// otherwise be silently unenforced.
/// </remarks>
public class CpuOffloadShardingConfigTests
{
    // ── config-surface tests ─────────────────────────────────────────────

    [Fact(Timeout = 60000)]
    public async Task ShardingConfiguration_DefaultsAllOffloadFlagsToFalse()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = new ShardingConfiguration<double>(backend);

        Assert.False(config.CpuOffloadOptimizer);
        Assert.False(config.CpuOffloadGradients);
        Assert.False(config.CpuOffloadParams);
    }

    [Fact(Timeout = 60000)]
    public async Task ShardingConfiguration_CreateForZeROOffload_SetsOnlyOptimizerFlag()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = ShardingConfiguration<double>.CreateForZeROOffload(backend);

        Assert.True(config.CpuOffloadOptimizer, "CreateForZeROOffload must enable optimizer-state offload.");
        Assert.False(config.CpuOffloadGradients, "Stage-1 offload must not touch gradients.");
        Assert.False(config.CpuOffloadParams, "Stage-1 offload must not touch parameters.");
    }

    [Fact(Timeout = 60000)]
    public async Task ShardingConfiguration_CreateForZeROOffloadFull_SetsAllThreeFlags()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = ShardingConfiguration<double>.CreateForZeROOffloadFull(backend);

        Assert.True(config.CpuOffloadOptimizer);
        Assert.True(config.CpuOffloadGradients);
        Assert.True(config.CpuOffloadParams);
    }

    [Fact(Timeout = 60000)]
    public async Task ShardingConfiguration_CreateForZeROOffload_ThrowsOnNullBackend()
    {
        await Task.Yield();
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateForZeROOffload(null!));
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateForZeROOffloadFull(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task ShardingConfiguration_OffloadFlags_AreIndependentlySettable()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = new ShardingConfiguration<double>(backend)
        {
            CpuOffloadOptimizer = true,
            CpuOffloadGradients = false,
            CpuOffloadParams = true,
        };

        Assert.True(config.CpuOffloadOptimizer);
        Assert.False(config.CpuOffloadGradients);
        Assert.True(config.CpuOffloadParams);
    }

    // ── runtime contract tests: the ACTUAL engine swap ───────────────────
    //
    // These drive RunWrappedOptimizerStep with a genuine non-CPU outer engine so the
    // swap branch executes (the previous tests set the outer to CpuEngine, which the
    // scope correctly no-ops, so they never exercised a swap at all).

    [Fact(Timeout = 60000)]
    public async Task RunWrappedOptimizerStep_SwapsToCpuAndRestoresExactOuter_WhenFlagOn_AndOuterIsNonCpu()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = ShardingConfiguration<double>.CreateForZeROOffload(backend);

        var priorEngine = AiDotNetEngine.Current;
        try
        {
            var outer = ThrowingEngineProxy.Create();
            AiDotNetEngine.Current = outer;
            Assert.IsNotType<CpuEngine>(outer);

            IEngine? seen = null;
            var stub = new EngineCapturingOptimizer<double, Matrix<double>, Vector<double>>(
                capture: e => seen = e);
            var sharded = new DDPOptimizer<double, Matrix<double>, Vector<double>>(stub, config);

            sharded.Optimize(new OptimizationInputData<double, Matrix<double>, Vector<double>>());

            // The wrapped step ran on a swapped-in CpuEngine (offload contract) ...
            Assert.IsType<CpuEngine>(seen);
            // ... and the EXACT outer instance was restored afterwards.
            Assert.Same(outer, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task RunWrappedOptimizerStep_RestoresExactOuter_WhenWrappedStepThrows()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = ShardingConfiguration<double>.CreateForZeROOffload(backend);

        var priorEngine = AiDotNetEngine.Current;
        try
        {
            var outer = ThrowingEngineProxy.Create();
            AiDotNetEngine.Current = outer;

            var stub = new EngineCapturingOptimizer<double, Matrix<double>, Vector<double>>(
                capture: _ => throw new InvalidOperationException("boom inside wrapped step"));
            var sharded = new DDPOptimizer<double, Matrix<double>, Vector<double>>(stub, config);

            Assert.Throws<InvalidOperationException>(() =>
                sharded.Optimize(new OptimizationInputData<double, Matrix<double>, Vector<double>>()));

            // Even though the wrapped step threw, the offload scope's Dispose must have
            // restored the exact outer engine (and released the engine-swap gate).
            Assert.Same(outer, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task RunWrappedOptimizerStep_IsNoOp_WhenOuterAlreadyCpu()
    {
        await Task.Yield();
        // Documents the fast path: when the outer engine is already a CpuEngine there is
        // nothing to swap, so the wrapped step sees that same instance and it is untouched.
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = ShardingConfiguration<double>.CreateForZeROOffload(backend);

        var priorEngine = AiDotNetEngine.Current;
        try
        {
            var outer = new CpuEngine();
            AiDotNetEngine.Current = outer;

            IEngine? seen = null;
            var stub = new EngineCapturingOptimizer<double, Matrix<double>, Vector<double>>(
                capture: e => seen = e);
            var sharded = new DDPOptimizer<double, Matrix<double>, Vector<double>>(stub, config);

            sharded.Optimize(new OptimizationInputData<double, Matrix<double>, Vector<double>>());

            Assert.Same(outer, seen);
            Assert.Same(outer, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task RunWrappedOptimizerStep_WrappedRunsUnchanged_WhenFlagOff()
    {
        await Task.Yield();
        // Flag off → the sharded wrapper is a pass-through; the outer engine reference the
        // wrapped optimizer sees is the same instance we set before the step (no scope swap).
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = new ShardingConfiguration<double>(backend); // flag off

        var priorEngine = AiDotNetEngine.Current;
        try
        {
            var outer = new CpuEngine();
            AiDotNetEngine.Current = outer;

            IEngine? seen = null;
            var stub = new EngineCapturingOptimizer<double, Matrix<double>, Vector<double>>(
                capture: e => seen = e);
            var sharded = new DDPOptimizer<double, Matrix<double>, Vector<double>>(stub, config);

            sharded.Optimize(new OptimizationInputData<double, Matrix<double>, Vector<double>>());

            Assert.Same(outer, seen);
            Assert.Same(outer, AiDotNetEngine.Current);
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    // ── gradient offload runtime tests ───────────────────────────────────
    //
    // The observable contract of CpuOffloadGradients is: before the sharded reduce, any
    // deferred GPU download registered against the gradient vector's backing array must be
    // drained (so the reduce sees live values, not the uninitialized bytes
    // GC.AllocateUninitializedArray left in place — see AiDotNet.Tensors PR #604 FinishGpuOp).

    [Fact(Timeout = 60000)]
    public async Task GradientOffload_DrainsDeferredDownload_BeforeReduce()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = new ShardingConfiguration<double>(backend)
        {
            AutoSyncGradients = true,
            CpuOffloadGradients = true,
        };

        var gradients = new Vector<double>(new double[8]);
        // Register against the vector's ACTUAL backing array — the same reference
        // OffloadGradientsToCpu materializes via gradients.GetDataArray() — so the test is
        // robust to whether the Vector ctor wraps or copies.
        var backing = gradients.GetDataArray();
        bool materializerFired = false;
        // Register a deferred materializer that flips the flag when the array is requested.
        // This mimics what FinishGpuOp does after a GPU op — the array is registered as
        // deferred until the first host read.
        DeferredArrayMaterializer.Register(backing, arr =>
        {
            var typed = (double[])arr;
            for (int i = 0; i < typed.Length; i++) typed[i] = i + 1; // "download"
            materializerFired = true;
        });
        try
        {
            var probe = new GradientProbeOptimizer<double, Matrix<double>, Vector<double>>(config);
            probe.CallOffload(gradients);

            Assert.True(materializerFired,
                "OffloadGradientsToCpu must drain any pending deferred download so the reduce reads live values.");
        }
        finally
        {
            // Remove the registration even if the assertion fails, so a leaked deferred
            // entry can't pollute later tests that reuse array identities.
            DeferredArrayMaterializer.Remove(backing);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task GradientOffload_IsNoOp_WhenFlagOff()
    {
        await Task.Yield();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        var config = new ShardingConfiguration<double>(backend); // CpuOffloadGradients=false

        var gradients = new Vector<double>(new double[8]);
        var backing = gradients.GetDataArray();
        bool materializerFired = false;
        DeferredArrayMaterializer.Register(backing, _ => materializerFired = true);
        try
        {
            var probe = new GradientProbeOptimizer<double, Matrix<double>, Vector<double>>(config);
            probe.CallOffload(gradients);

            Assert.False(materializerFired,
                "Flag off must NOT trigger materialization (pass-through path).");
        }
        finally
        {
            DeferredArrayMaterializer.Remove(backing);
        }
    }

    // ── test-only helpers ────────────────────────────────────────────────

    /// <summary>
    /// A non-CPU <see cref="IEngine"/> used purely as an identity marker for the engine-swap
    /// tests. It is NOT a <see cref="CpuEngine"/>, so the offload scope performs a real swap;
    /// and it throws on any actual tensor op, because the wrapped step must run on the
    /// swapped-in CpuEngine and must never touch this outer engine.
    /// </summary>
    private class ThrowingEngineProxy : DispatchProxy
    {
        public static IEngine Create() => Create<IEngine, ThrowingEngineProxy>();

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
            => throw new NotSupportedException(
                $"ThrowingEngineProxy is a non-CpuEngine identity marker for engine-swap tests; " +
                $"'{targetMethod?.Name}' must never be called — the wrapped step runs on the swapped-in CpuEngine.");
    }

    /// <summary>Test-only sharded optimizer that exposes the protected
    /// OffloadGradientsToCpu helper so the gradient-offload tests can drive it
    /// without spinning up a full IOptimizer + Adam step.</summary>
    private sealed class GradientProbeOptimizer<T, TInput, TOutput>
        : ShardedOptimizerBase<T, TInput, TOutput>
    {
        public GradientProbeOptimizer(IShardingConfiguration<T> config)
            : base(new EngineCapturingOptimizer<T, TInput, TOutput>(_ => { }), config) { }

        // The one supported member: it forwards to the protected helper under test.
        public void CallOffload(Vector<T> gradients) => OffloadGradientsToCpu(gradients);

        // Everything below is outside this probe's contract — fail fast if the runtime
        // ever routes through it so a silent placeholder can't hide an unexpected call.
        public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
            => throw new NotSupportedException();
        public override void SynchronizeOptimizerState() => throw new NotSupportedException();
        public override byte[] Serialize() => throw new NotSupportedException();
        public override void Deserialize(byte[] data) => throw new NotSupportedException();
    }

    /// <summary>
    /// Strict gradient-based-optimizer test double. It implements IGradientBasedOptimizer so
    /// DDPOptimizer accepts it, but only <see cref="Optimize"/> — which records the engine active
    /// inside the wrapped step — is a real contract member. Because the double is not an
    /// OptimizerBase and exposes no InitialSolution, DDP's Optimize invokes nothing else on it, so
    /// every other member throws NotSupportedException: a strict mock that surfaces any unexpected
    /// call instead of silently returning a placeholder default.
    /// </summary>
    private sealed class EngineCapturingOptimizer<T, TInput, TOutput> : IGradientBasedOptimizer<T, TInput, TOutput>
    {
        private readonly Action<IEngine> _capture;
        public EngineCapturingOptimizer(Action<IEngine> capture) { _capture = capture; }

        public OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
        {
            _capture(AiDotNetEngine.Current);
            return new OptimizationResult<T, TInput, TOutput>();
        }

        // ── IOptimizer members outside this double's contract ──
        public OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions() => throw new NotSupportedException();
        public bool ShouldEarlyStop() => throw new NotSupportedException();
        public void Reset() => throw new NotSupportedException();
        public void SetModel(IFullModel<T, TInput, TOutput> model) => throw new NotSupportedException();
        public byte[] Serialize() => throw new NotSupportedException();
        public void Deserialize(byte[] data) => throw new NotSupportedException();
        public void SaveModel(string filePath) => throw new NotSupportedException();
        public void LoadModel(string filePath) => throw new NotSupportedException();

        // ── IGradientBasedOptimizer members outside this double's contract ──
        public Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient) => throw new NotSupportedException();
        public Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient) => throw new NotSupportedException();
        public void UpdateParameters(List<ILayer<T>> layers) => throw new NotSupportedException();
        public double GetCurrentLearningRate() => throw new NotSupportedException();
        public void Step(AiDotNet.Tensors.Engines.Autodiff.TapeStepContext<T> context) => throw new NotSupportedException();
        public Vector<T> LastComputedGradients => throw new NotSupportedException();
        public IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> gradients, IFullModel<T, TInput, TOutput> model) => throw new NotSupportedException();
        public IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> originalParameters, Vector<T> gradients, IFullModel<T, TInput, TOutput> model) => throw new NotSupportedException();
        public Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients) => throw new NotSupportedException();
        public bool SupportsGpuUpdate => throw new NotSupportedException();
        public void UpdateParametersGpu(AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer parameters, AiDotNet.Tensors.Engines.DirectGpu.IGpuBuffer gradients, int parameterCount, AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend backend) => throw new NotSupportedException();
        public void InitializeGpuState(int parameterCount, AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend backend) => throw new NotSupportedException();
        public void DisposeGpuState() => throw new NotSupportedException();
    }
}
