using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.TimeSeries;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Arena;

/// <summary>
/// Regression + invariant suite for the TensorArena "zero-alloc training"
/// mechanism at the MODEL level. The core guarantee: the arena is a pure
/// allocation optimization, so training arena-ON must produce results
/// bit-identical to arena-OFF for the same seed. The clean toggle is
/// <see cref="TensorPool.ForceFreshAllocations"/> — true = arena bypassed
/// (fresh allocs), false = arena engaged.
///
/// These guard the model zoo against the class of bug fixed in #1804 (the
/// arena ring re-issued a pooled buffer under the wrong shape). If any
/// equivalence test fails beyond fp-noise, that is a real correctness bug in
/// the arena, not the model.
/// </summary>
[CollectionDefinition(nameof(TensorArenaTrainingEquivalenceTests), DisableParallelization = true)]
public sealed class TensorArenaTrainingEquivalenceTestsCollection { }

[Collection(nameof(TensorArenaTrainingEquivalenceTests))]
public class TensorArenaTrainingEquivalenceTests
{
    // <=1e-4 required by the guarantee; we expect bit-exact (0).
    private const double EquivTol = 1e-4;

    // ---------------------------------------------------------------
    // Determinism plumbing — mirrors NeuralNetworkModelTestBase so two
    // sequential runs on the same thread are bit-reproducible: pinned
    // weight-init seed, single-threaded BLAS reductions, CPU engine, and
    // a cleared fused-training-plan cache (which otherwise carries Adam
    // moment state across runs).
    // ---------------------------------------------------------------
    private static void SetDeterminism()
    {
        BlasProvider.SetDeterministicMode(true);
        NeuralNetworkArchitecture<double>.DefaultRandomSeedOverride = 1234;
        NeuralNetworkArchitecture<float>.DefaultRandomSeedOverride = 1234;
        if (AiDotNetEngine.Current is not CpuEngine)
            AiDotNetEngine.ResetToCpu();
    }

    private static void ResetGlobalTrainingState()
    {
        try { AiDotNet.Training.CompiledTapeTrainingStep<double>.Invalidate(); } catch { }
        try { AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate(); } catch { }
        try { WeightRegistry.Reset(); } catch { }
        try { InferenceWeightCache.InvalidateAll(); } catch { }
        // Drop any cross-arena persistent buffers so a prior run's pool
        // state can't leak into the next run's warmup.
        TensorArena.ClearPersistentPool();
    }

    private static double MaxAbsDelta(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double m = 0;
        for (int i = 0; i < a.Length; i++)
            m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static double MaxAbsDelta(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double m = 0;
        for (int i = 0; i < a.Length; i++)
            m = Math.Max(m, Math.Abs((double)a[i] - b[i]));
        return m;
    }

    // Deterministic fixed input/target (no RNG) so data is never a variable.
    private static Tensor<float> FixedTensor(int[] shape, int salt)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++)
            t[i] = (float)(0.5 + 0.25 * Math.Sin((i + salt) * 0.137));
        return t;
    }

    private static (Matrix<double> X, Vector<double> Y) FixedSeries(int length)
    {
        var rng = new Random(7); // local + fixed -> identical across runs
        var x = new Matrix<double>(length, 1);
        var y = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            x[i, 0] = i;
            y[i] = 0.5 * i + 3.0 * Math.Sin(2.0 * Math.PI * i / 20.0)
                   + (rng.NextDouble() - 0.5) * 0.1;
        }
        return (x, y);
    }

    // =====================================================================
    // A) MODEL-LEVEL EQUIVALENCE: arena-ON must equal arena-OFF, same seed
    // =====================================================================

    /// <summary>N-BEATS — the known-good case. TimeSeriesModelBase opens a
    /// training arena; #1804's permute reshape lives on this path.</summary>
    [Fact]
    public void NBeats_ArenaOnEqualsOff()
    {
        static (double[] pred, double[] parms) Run(bool forceFresh)
        {
            ResetGlobalTrainingState();
            SetDeterminism();
            TensorPool.ForceFreshAllocations = forceFresh;

            var (x, y) = FixedSeries(60);
            using var model = new NBEATSModel<double>(new NBEATSModelOptions<double>
            {
                NumStacks = 2,
                NumBlocksPerStack = 1,
                LookbackWindow = 10,
                ForecastHorizon = 5,
                HiddenLayerSize = 16,
                NumHiddenLayers = 1,
                Epochs = 15,
                BatchSize = 16,
                LearningRate = 0.001,
                // MaxTrainingTimeSeconds left 0 -> honor Epochs EXACTLY
                // (time-bounded mode makes the epoch count wall-clock dependent).
            });
            model.Train(x, y);
            var pred = model.Predict(x);
            return (pred.ToArray(), model.GetParameters().ToArray());
        }

        var off = Run(true);
        var on = Run(false);
        try
        {
            double predDelta = MaxAbsDelta(off.pred, on.pred);
            double parmDelta = MaxAbsDelta(off.parms, on.parms);
            Assert.True(predDelta <= EquivTol,
                $"N-BEATS arena-ON != arena-OFF: max pred delta = {predDelta:E3}");
            Assert.True(parmDelta <= EquivTol,
                $"N-BEATS arena-ON != arena-OFF: max param delta = {parmDelta:E3}");
        }
        finally { TensorPool.ForceFreshAllocations = false; }
    }

    /// <summary>Feed-forward MLP — exercises NeuralNetworkBase.TrainWithTape's
    /// top-level per-step arena (the #478 Optimize/tape path).</summary>
    [Fact]
    public void FeedForward_ArenaOnEqualsOff()
    {
        static (float[] pred, float[] parms) Run(bool forceFresh)
        {
            ResetGlobalTrainingState();
            SetDeterminism();
            TensorPool.ForceFreshAllocations = forceFresh;

            using var net = new FeedForwardNeuralNetwork<float>();
            var input = FixedTensor(new[] { 128 }, 0);
            var target = FixedTensor(new[] { 1 }, 99);
            for (int i = 0; i < 20; i++) net.Train(input, target);
            var pred = net.Predict(input);
            return (pred.ToArray(), net.GetParameters().ToArray());
        }

        var off = Run(true);
        var on = Run(false);
        try
        {
            double predDelta = MaxAbsDelta(off.pred, on.pred);
            double parmDelta = MaxAbsDelta(off.parms, on.parms);
            Assert.True(predDelta <= EquivTol,
                $"FFNN arena-ON != arena-OFF: max pred delta = {predDelta:E3}");
            Assert.True(parmDelta <= EquivTol,
                $"FFNN arena-ON != arena-OFF: max param delta = {parmDelta:E3}");
        }
        finally { TensorPool.ForceFreshAllocations = false; }
    }

    /// <summary>GRU recurrent net — exercises recurrent tape reuse across
    /// unrolled timesteps under the arena.</summary>
    [Fact]
    public void Gru_ArenaOnEqualsOff()
    {
        static (float[] pred, float[] parms) Run(bool forceFresh)
        {
            ResetGlobalTrainingState();
            SetDeterminism();
            TensorPool.ForceFreshAllocations = forceFresh;

            using var net = new GRUNeuralNetwork<float>();
            var input = FixedTensor(new[] { 128 }, 3);
            var target = FixedTensor(new[] { 1 }, 42);
            for (int i = 0; i < 15; i++) net.Train(input, target);
            var pred = net.Predict(input);
            return (pred.ToArray(), net.GetParameters().ToArray());
        }

        var off = Run(true);
        var on = Run(false);
        try
        {
            double predDelta = MaxAbsDelta(off.pred, on.pred);
            double parmDelta = MaxAbsDelta(off.parms, on.parms);
            Assert.True(predDelta <= EquivTol,
                $"GRU arena-ON != arena-OFF: max pred delta = {predDelta:E3}");
            Assert.True(parmDelta <= EquivTol,
                $"GRU arena-ON != arena-OFF: max param delta = {parmDelta:E3}");
        }
        finally { TensorPool.ForceFreshAllocations = false; }
    }

    // =====================================================================
    // B) INVARIANTS
    // =====================================================================

    /// <summary>#5 Determinism: same seed, arena ON, run twice -> identical.
    /// If this fails, the whole equivalence comparison is meaningless, so it
    /// guards the harness itself.</summary>
    [Fact]
    public void Determinism_ArenaOn_TwiceIdentical()
    {
        static float[] Run()
        {
            ResetGlobalTrainingState();
            SetDeterminism();
            TensorPool.ForceFreshAllocations = false; // arena ON
            using var net = new FeedForwardNeuralNetwork<float>();
            var input = FixedTensor(new[] { 128 }, 0);
            var target = FixedTensor(new[] { 1 }, 99);
            for (int i = 0; i < 20; i++) net.Train(input, target);
            return net.Predict(input).ToArray();
        }

        var a = Run();
        var b = Run();
        double delta = MaxAbsDelta(a, b);
        Assert.True(delta == 0.0,
            $"Arena-ON training is non-deterministic across runs: max delta = {delta:E3}");
    }

    /// <summary>#6 Convergence: arena ON, training loss (MSE to target) must
    /// improve over epochs — the arena Reset between steps must not corrupt
    /// the optimizer trajectory.</summary>
    [Fact]
    public void Convergence_ArenaOn_LossImproves()
    {
        ResetGlobalTrainingState();
        SetDeterminism();
        TensorPool.ForceFreshAllocations = false; // arena ON

        using var net = new FeedForwardNeuralNetwork<float>();
        var input = FixedTensor(new[] { 128 }, 0);
        var target = FixedTensor(new[] { 1 }, 99);

        double Mse()
        {
            var p = net.Predict(input).ToArray();
            var t = target.ToArray();
            double s = 0; for (int i = 0; i < p.Length; i++) { double d = p[i] - t[i]; s += d * d; }
            return s / p.Length;
        }

        double initial = Mse();
        for (int i = 0; i < 40; i++) net.Train(input, target);
        double final = Mse();

        Assert.True(final < initial,
            $"Arena-ON training did not reduce loss: initial={initial:E4}, final={final:E4}");
    }

    /// <summary>#7 Nested arena: wrap training in an OUTER arena (Train opens
    /// its own inner arena -> nested). Result must equal arena-OFF.</summary>
    [Fact]
    public void NestedArena_Training_EqualsArenaOff()
    {
        static float[] RunNested()
        {
            ResetGlobalTrainingState();
            SetDeterminism();
            TensorPool.ForceFreshAllocations = false; // arena ON
            using var outer = TensorArena.Create();   // OUTER arena -> Train nests inside
            using var net = new FeedForwardNeuralNetwork<float>();
            var input = FixedTensor(new[] { 128 }, 0);
            var target = FixedTensor(new[] { 1 }, 99);
            for (int i = 0; i < 20; i++) net.Train(input, target);
            return net.Predict(input).ToArray();
        }

        static float[] RunOff()
        {
            ResetGlobalTrainingState();
            SetDeterminism();
            TensorPool.ForceFreshAllocations = true; // arena OFF
            using var net = new FeedForwardNeuralNetwork<float>();
            var input = FixedTensor(new[] { 128 }, 0);
            var target = FixedTensor(new[] { 1 }, 99);
            for (int i = 0; i < 20; i++) net.Train(input, target);
            return net.Predict(input).ToArray();
        }

        var nested = RunNested();
        var off = RunOff();
        try
        {
            double delta = MaxAbsDelta(nested, off);
            Assert.True(delta <= EquivTol,
                $"Nested-arena training != arena-OFF: max delta = {delta:E3}");
        }
        finally { TensorPool.ForceFreshAllocations = false; }
    }

    /// <summary>#8 Thread isolation: arena is [ThreadStatic]. Train two models
    /// concurrently on two threads; each must equal its single-threaded
    /// arena-OFF reference (no cross-thread buffer bleed).</summary>
    [Fact]
    public void ThreadIsolation_ConcurrentTraining_MatchesReference()
    {
        static float[] TrainOne(bool forceFresh)
        {
            SetDeterminism();
            TensorPool.ForceFreshAllocations = forceFresh;
            using var net = new FeedForwardNeuralNetwork<float>();
            var input = FixedTensor(new[] { 128 }, 0);
            var target = FixedTensor(new[] { 1 }, 99);
            for (int i = 0; i < 20; i++) net.Train(input, target);
            return net.Predict(input).ToArray();
        }

        // Single-threaded arena-OFF reference.
        ResetGlobalTrainingState();
        var reference = TrainOne(true);

        // Two concurrent arena-ON trainings on separate threads. ForceFreshAllocations
        // is a process-wide static, so set it ON once up front and keep it fixed for
        // the duration (the arena itself is thread-static — that is what we test).
        ResetGlobalTrainingState();
        SetDeterminism();
        TensorPool.ForceFreshAllocations = false;

        float[]? r1 = null, r2 = null;
        Exception? e1 = null, e2 = null;
        var t1 = new Thread(() =>
        {
            try
            {
                TensorArena.ClearPersistentPool();
                using var net = new FeedForwardNeuralNetwork<float>();
                var input = FixedTensor(new[] { 128 }, 0);
                var target = FixedTensor(new[] { 1 }, 99);
                for (int i = 0; i < 20; i++) net.Train(input, target);
                r1 = net.Predict(input).ToArray();
            }
            catch (Exception ex) { e1 = ex; }
        });
        var t2 = new Thread(() =>
        {
            try
            {
                TensorArena.ClearPersistentPool();
                using var net = new FeedForwardNeuralNetwork<float>();
                var input = FixedTensor(new[] { 128 }, 0);
                var target = FixedTensor(new[] { 1 }, 99);
                for (int i = 0; i < 20; i++) net.Train(input, target);
                r2 = net.Predict(input).ToArray();
            }
            catch (Exception ex) { e2 = ex; }
        });
        t1.Start(); t2.Start();
        t1.Join(); t2.Join();

        TensorPool.ForceFreshAllocations = false;
        Assert.Null(e1);
        Assert.Null(e2);
        Assert.NotNull(r1);
        Assert.NotNull(r2);

        double d1 = MaxAbsDelta(reference, r1!);
        double d2 = MaxAbsDelta(reference, r2!);
        Assert.True(d1 <= EquivTol, $"Thread 1 arena result diverged from reference: {d1:E3}");
        Assert.True(d2 <= EquivTol, $"Thread 2 arena result diverged from reference: {d2:E3}");
    }
}
