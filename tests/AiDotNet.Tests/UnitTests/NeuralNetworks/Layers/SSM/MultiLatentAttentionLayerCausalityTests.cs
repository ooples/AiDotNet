using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Pins the attention semantics of <see cref="MultiLatentAttentionLayer{T}"/> across engine attention kernels: strict
/// causality (position i reads only positions j &lt;= i), agreement between the fused host float32 kernel and the
/// decomposed chain every other element type runs, and a gradient that reaches the input.
/// </summary>
[Collection("NonParallelIntegration")]
public class MultiLatentAttentionLayerCausalityTests
{
    private const int Batch = 2;
    private const int Seq = 6;
    private const int Dim = 8;
    private const int Heads = 2;
    private const int Latent = 4;

    private static Tensor<double> MakeInput(int seq, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var x = new Tensor<double>(new[] { Batch, seq, Dim });
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, rng.NextDouble() * 2.0 - 1.0);
        return x;
    }

    private static Tensor<float> ToFloat(Tensor<double> x)
    {
        var f = new Tensor<float>(x.Shape.ToArray());
        for (int i = 0; i < x.Length; i++) f.SetFlat(i, (float)x.GetFlat(i));
        return f;
    }

    /// <summary>A double layer and a float layer holding the same (float-representable) parameters.</summary>
    private static (MultiLatentAttentionLayer<double> D, MultiLatentAttentionLayer<float> F) MakePair()
    {
        var d = new MultiLatentAttentionLayer<double>(Seq, Dim, Heads, Latent);
        var f = new MultiLatentAttentionLayer<float>(Seq, Dim, Heads, Latent);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(4242);
        var pf = f.GetParameters();
        var valuesF = new float[pf.Length];
        var valuesD = new double[pf.Length];
        for (int i = 0; i < valuesF.Length; i++)
        {
            valuesF[i] = (float)(rng.NextDouble() * 1.2 - 0.6);
            valuesD[i] = valuesF[i];
        }
        f.SetParameters(new Vector<float>(valuesF));
        d.SetParameters(new Vector<double>(valuesD));
        return (d, f);
    }

    [Fact]
    public void Forward_IsStrictlyCausal_InBothAttentionKernels()
    {
        var (d, f) = MakePair();
        var x = MakeInput(Seq, 11);
        var baseD = d.Forward(x);
        var baseF = f.Forward(ToFloat(x));

        const int changed = 3;
        var x2 = MakeInput(Seq, 11);
        for (int b = 0; b < Batch; b++)
            for (int c = 0; c < Dim; c++)
                x2.SetFlat((b * Seq + changed) * Dim + c, x2.GetFlat((b * Seq + changed) * Dim + c) + 0.75);

        var movedD = d.Forward(x2);
        var movedF = f.Forward(ToFloat(x2));

        double laterDeltaD = 0.0, laterDeltaF = 0.0;
        for (int b = 0; b < Batch; b++)
        {
            for (int s = 0; s < Seq; s++)
            {
                for (int c = 0; c < Dim; c++)
                {
                    int i = (b * Seq + s) * Dim + c;
                    double dd = Math.Abs(baseD.GetFlat(i) - movedD.GetFlat(i));
                    double df = Math.Abs(baseF.GetFlat(i) - movedF.GetFlat(i));
                    if (s < changed)
                    {
                        Assert.True(dd <= 1e-12, $"double: position {s} read future position {changed} (delta {dd})");
                        Assert.True(df <= 1e-6, $"float: position {s} read future position {changed} (delta {df})");
                    }
                    else
                    {
                        laterDeltaD = Math.Max(laterDeltaD, dd);
                        laterDeltaF = Math.Max(laterDeltaF, df);
                    }
                }
            }
        }

        // The perturbation has to matter somewhere, or the assertions above prove nothing.
        Assert.True(laterDeltaD > 1e-4, $"double: perturbing position {changed} changed nothing at or after it");
        Assert.True(laterDeltaF > 1e-4, $"float: perturbing position {changed} changed nothing at or after it");
    }

    [Fact]
    public void Forward_PrefixOfSequence_MatchesFullSequencePrefix()
    {
        var (d, f) = MakePair();
        var full = MakeInput(Seq, 23);
        const int prefix = 4;
        var head = new Tensor<double>(new[] { Batch, prefix, Dim });
        for (int b = 0; b < Batch; b++)
            for (int s = 0; s < prefix; s++)
                for (int c = 0; c < Dim; c++)
                    head.SetFlat((b * prefix + s) * Dim + c, full.GetFlat((b * Seq + s) * Dim + c));

        var fullD = d.Forward(full);
        var headD = d.Forward(head);
        var fullF = f.Forward(ToFloat(full));
        var headF = f.Forward(ToFloat(head));

        for (int b = 0; b < Batch; b++)
        {
            for (int s = 0; s < prefix; s++)
            {
                for (int c = 0; c < Dim; c++)
                {
                    int iFull = (b * Seq + s) * Dim + c;
                    int iHead = (b * prefix + s) * Dim + c;
                    Assert.True(Math.Abs(fullD.GetFlat(iFull) - headD.GetFlat(iHead)) <= 1e-12,
                        $"double: prefix output differs at b={b}, s={s}, c={c}");
                    Assert.True(Math.Abs(fullF.GetFlat(iFull) - headF.GetFlat(iHead)) <= 1e-5,
                        $"float: prefix output differs at b={b}, s={s}, c={c}");
                }
            }
        }
    }

    [Fact]
    public void Forward_FusedFloatKernel_AgreesWithDecomposedDoubleChain()
    {
        var (d, f) = MakePair();
        var x = MakeInput(Seq, 37);
        var yd = d.Forward(x);
        var yf = f.Forward(ToFloat(x));
        Assert.Equal(yd.Shape.ToArray(), yf.Shape.ToArray());
        for (int i = 0; i < yd.Length; i++)
        {
            Assert.True(Math.Abs(yd.GetFlat(i) - yf.GetFlat(i)) <= 1e-4 * Math.Max(1.0, Math.Abs(yd.GetFlat(i))),
                $"float and double attention disagree at {i}: double={yd.GetFlat(i)}, float={yf.GetFlat(i)}");
        }
    }

    [Fact]
    public void Backward_InputGradient_MatchesFiniteDifferences_AndFloatMatchesDouble()
    {
        var (d, f) = MakePair();
        var x = MakeInput(Seq, 51);
        var xf = ToFloat(x);
        var engine = AiDotNetEngine.Current;

        Tensor<double> gradD;
        using (var tape = new GradientTape<double>())
        {
            var y = d.Forward(x);
            var loss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(y, y), null), 0.5);
            gradD = tape.ComputeGradients(loss, sources: new[] { x })[x];
        }

        Tensor<float> gradF;
        using (var tape = new GradientTape<float>())
        {
            var y = f.Forward(xf);
            var loss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(y, y), null), 0.5f);
            gradF = tape.ComputeGradients(loss, sources: new[] { xf })[xf];
        }

        double Loss()
        {
            var y = d.Forward(x);
            double sum = 0.0;
            for (int i = 0; i < y.Length; i++) sum += 0.5 * y.GetFlat(i) * y.GetFlat(i);
            return sum;
        }

        const double eps = 1e-6;
        double gradScale = 0.0;
        for (int i = 0; i < x.Length; i++)
        {
            double orig = x.GetFlat(i);
            x.SetFlat(i, orig + eps);
            double lp = Loss();
            x.SetFlat(i, orig - eps);
            double lm = Loss();
            x.SetFlat(i, orig);
            double numeric = (lp - lm) / (2.0 * eps);
            gradScale = Math.Max(gradScale, Math.Abs(numeric));
            Assert.True(Math.Abs(gradD.GetFlat(i) - numeric) <= 1e-6 * Math.Max(1.0, Math.Abs(numeric)),
                $"dL/dx mismatch at {i}: analytic={gradD.GetFlat(i)}, numeric={numeric}");
            Assert.True(Math.Abs(gradF.GetFlat(i) - numeric) <= 2e-3 * Math.Max(1.0, Math.Abs(numeric)),
                $"float dL/dx mismatch at {i}: analytic={gradF.GetFlat(i)}, numeric={numeric}");
        }

        Assert.True(gradScale > 1e-6, "the loss does not depend on the input: the gradient check is vacuous");
    }

    /// <summary>A model of one multi-latent attention layer over [batch, Seq, Dim], with seeded parameters.</summary>
    private class MlaModel<T> : NeuralNetwork<T>
    {
        public MlaModel(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null, int seq = Seq, int dim = Dim)
            : base(new NeuralNetworkArchitecture<T>(
                inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputHeight: seq, inputWidth: dim, outputSize: seq * dim,
                layers: new List<ILayer<T>> { new MultiLatentAttentionLayer<T>(seq, dim, Heads, Latent) }), optimizer)
        {
        }
    }

    /// <summary>The same model, kept on the eager tape step.</summary>
    private sealed class EagerMlaModel<T> : MlaModel<T>
    {
        public EagerMlaModel(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, int seq = Seq, int dim = Dim)
            : base(optimizer, seq, dim)
        {
        }

        protected override bool SupportsFusedCompiledTraining => false;
    }

    private static Tensor<T> Values<T>(int[] shape, int seed, double scale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, ops.FromDouble((rng.NextDouble() * 2.0 - 1.0) * scale));
        return t;
    }

    private static void Seed<T>(NeuralNetworkBase<T> model)
    {
        model.Predict(Values<T>(new[] { Batch, Seq, Dim }, 1, 1.0));   // materialize any lazily shaped state
        int count = model.GetParameters().Length;
        var flat = Values<T>(new[] { count }, 4242, 0.6);
        var vector = new Vector<T>(count);
        for (int i = 0; i < count; i++) vector[i] = flat.GetFlat(i);
        model.UpdateParameters(vector);
    }

    private static double[] ToDoubles<T>(Tensor<T> t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new double[t.Length];
        for (int i = 0; i < result.Length; i++) result[i] = ops.ToDouble(t.GetFlat(i));
        return result;
    }

    private static double[] ToDoubles<T>(Vector<T> v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new double[v.Length];
        for (int i = 0; i < result.Length; i++) result[i] = ops.ToDouble(v[i]);
        return result;
    }

    private static CompiledModelHost<T> CompileHostOf<T>(NeuralNetworkBase<T> model)
        => (CompiledModelHost<T>)(typeof(NeuralNetworkBase<T>)
            .GetField("_compileHost", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?.GetValue(model) ?? throw new InvalidOperationException("NeuralNetworkBase has no _compileHost field."));

    /// <summary>
    /// A model holding this layer keeps its compiled inference plan for every element type. Only float32 has the fused
    /// attention kernel; the engine runs every other type through a masked attention op that refuses to be traced for
    /// inference, so the host dropped the plan and ran such a model eagerly on every call, without saying so. The plan
    /// must be the thing that answers (the host's replay counter moves) and it must answer what the eager forward does,
    /// on values other than the ones it was traced with.
    /// </summary>
    private static void AssertCompiledInferenceKeepsItsPlan<T>(double tolerance, IEngine? engine = null)
    {
        var previousEngine = AiDotNetEngine.Current;
        var options = TensorCodecOptions.Current;
        AiDotNetEngine.Current = engine ?? new CpuEngine();
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = false });
            var model = new MlaModel<T>();
            model.SetTrainingMode(false);
            Seed(model);
            var traced = Values<T>(new[] { Batch, Seq, Dim }, 61, 1.0);
            var replayed = Values<T>(new[] { Batch, Seq, Dim }, 62, 1.0);
            var eager = ToDoubles(model.Predict(replayed));

            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            Assert.True(model.CompileForward(traced), "CompileForward reported a failure");
            var host = CompileHostOf(model);
            long before = host.HotPathHits;
            var compiled = ToDoubles(model.PredictCompiled(replayed));
            Assert.True(host.HotPathHits > before,
                $"{typeof(T).Name}: the compiled plan did not answer (replays {before} -> {host.HotPathHits}, "
                + $"slow-path calls {host.SlowPathCalls}); the host fell back to the eager forward");

            Assert.Equal(eager.Length, compiled.Length);
            double largest = 0.0;
            foreach (double v in eager) largest = Math.Max(largest, Math.Abs(v));
            Assert.True(largest > 1e-3, "the eager output is all zero, so the comparison proves nothing");
            for (int i = 0; i < eager.Length; i++)
                Assert.True(Math.Abs(eager[i] - compiled[i]) <= tolerance * Math.Max(1.0, largest),
                    $"{typeof(T).Name}: compiled output {compiled[i]:G9} differs from eager {eager[i]:G9} at {i}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(options);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    [Fact]
    public void CompiledInference_KeepsItsPlan_AndMatchesEager_Float() => AssertCompiledInferenceKeepsItsPlan<float>(1e-5);

    [Fact]
    public void CompiledInference_KeepsItsPlan_AndMatchesEager_Double() => AssertCompiledInferenceKeepsItsPlan<double>(1e-12);

    [Fact]
    public void CompiledInference_KeepsItsPlan_AndMatchesEager_Half() => AssertCompiledInferenceKeepsItsPlan<Half>(1e-2);

    /// <summary>The float32 plan is kept on a GPU engine as well.</summary>
    [SkippableFact]
    public void CompiledInference_KeepsItsPlan_AndMatchesEager_FloatOnTheGpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.SupportsGpu && gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        AssertCompiledInferenceKeepsItsPlan<float>(1e-4, gpu);
    }

    private const int TrainingSteps = 6;

    private static AdamWOptimizer<float, Tensor<float>, Tensor<float>> Optimizer()
        => new(null, new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 });

    /// <summary>Trains the model <see cref="TrainingSteps"/> steps from the seeded parameters; returns each step's loss and the final parameters.</summary>
    private static (double[] Losses, double[] Parameters, bool Fused, string? Miss) Train(IEngine engine, bool fused)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            MlaModel<float> model = fused ? new MlaModel<float>(Optimizer()) : new EagerMlaModel<float>(Optimizer());
            Seed(model);
            model.SetTrainingMode(true);
            var x = Values<float>(new[] { Batch, Seq, Dim }, 71, 1.0);
            var y = Values<float>(new[] { Batch, Seq, Dim }, 72, 1.0);
            var losses = new double[TrainingSteps];
            for (int s = 0; s < TrainingSteps; s++)
            {
                model.Train(x, y);
                losses[s] = model.GetLastLoss();
            }
            return (losses, ToDoubles(model.GetParameters()), model.FusedSession.IsCommitted, model.FusedSession.LastMissReason);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// The compiled training plan of a model holding this layer trains it as the eager tape does: the same loss at
    /// every step and the same parameters after them. The plan has to be the thing that ran, or the two arms are one.
    /// </summary>
    private static void AssertFusedTrainingMatchesEager(IEngine engine, double tolerance)
    {
        var eager = Train(engine, fused: false);
        var fused = Train(engine, fused: true);
        Assert.False(eager.Fused, "the eager arm committed a compiled plan");
        Assert.True(fused.Fused, "the compiled training plan did not commit: " + (fused.Miss ?? "no reason recorded"));

        Assert.True(eager.Losses[0] > 1e-3, "the first loss is zero, so nothing is being trained");
        Assert.True(eager.Losses[TrainingSteps - 1] < 0.98 * eager.Losses[0],
            $"the eager loss did not fall ({eager.Losses[0]:G6} -> {eager.Losses[TrainingSteps - 1]:G6}), so the steps prove nothing");
        for (int s = 0; s < TrainingSteps; s++)
            Assert.True(Math.Abs(eager.Losses[s] - fused.Losses[s]) <= tolerance * Math.Max(1.0, Math.Abs(eager.Losses[s])),
                $"step {s}: compiled loss {fused.Losses[s]:G9}, eager loss {eager.Losses[s]:G9}");

        Assert.Equal(eager.Parameters.Length, fused.Parameters.Length);
        double moved = 0.0;
        var seeded = Values<float>(new[] { eager.Parameters.Length }, 4242, 0.6);
        for (int i = 0; i < eager.Parameters.Length; i++)
        {
            moved = Math.Max(moved, Math.Abs(eager.Parameters[i] - seeded.GetFlat(i)));
            Assert.True(Math.Abs(eager.Parameters[i] - fused.Parameters[i]) <= tolerance * Math.Max(1.0, Math.Abs(eager.Parameters[i])),
                $"parameter {i}: compiled {fused.Parameters[i]:G9}, eager {eager.Parameters[i]:G9}");
        }
        Assert.True(moved > 1e-3, "training moved no parameter");
    }

    /// <summary>
    /// An eager GPU step through this layer keeps its attention on the device. On a backend with no fused attention
    /// kernel the engine's attention core runs its decomposed chain, whose attention runs on the host under the tape:
    /// it read the query, key and value back from the device on every forward. A step may read scalars (the loss,
    /// the clip norms), never a tensor the size of an attention operand.
    /// </summary>
    [SkippableFact]
    public void EagerGpuStep_ReadsNoAttentionOperandBack()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.SupportsGpu && gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");

        const int seq = 32, dim = 32;
        const long operandBytes = (long)Batch * seq * dim * sizeof(float);
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var model = new EagerMlaModel<float>(Optimizer(), seq, dim);
            model.SetTrainingMode(true);
            var x = Values<float>(new[] { Batch, seq, dim }, 81, 1.0);
            var y = Values<float>(new[] { Batch, seq, dim }, 82, 1.0);
            for (int s = 0; s < 3; s++) model.Train(x, y);

            // The least of three steps: readbacks of a test running beside this one can only add to a step's count.
            long least = long.MaxValue;
            for (int s = 0; s < 3; s++)
            {
                long before = AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.ReadbackBytes;
                model.Train(x, y);
                least = Math.Min(least, AiDotNet.Tensors.Engines.DirectGpu.GpuLaunchProbe.ReadbackBytes - before);
            }
            Assert.False(model.FusedSession.IsCommitted, "the step ran on a compiled plan, not the eager tape");
            Assert.True(least < operandBytes,
                $"an eager GPU step read {least:N0} B back from the device; one attention operand is {operandBytes:N0} B");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [Fact]
    public void CompiledTrainingPlan_MatchesTheEagerTape_OnTheCpuEngine()
        => AssertFusedTrainingMatchesEager(new CpuEngine(), 1e-4);

    [SkippableFact]
    public void CompiledTrainingPlan_MatchesTheEagerTape_OnTheGpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.SupportsGpu && gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        AssertFusedTrainingMatchesEager(gpu, 2e-3);
    }
}
