using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Training;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// Deterministic fused-vs-eager parity for every optimizer that maps onto a Tensors fused kernel.
/// </summary>
/// <remarks>
/// <para>
/// Each case takes the optimizer's own fused configuration (its <c>IFusedOptimizerSpec</c>), configures a compiled
/// training plan with it exactly as the training loop does, and drives that plan and the eager tape
/// <c>Step</c> with the same fixed sequence of gradients from the same starting weights. The loss is
/// <c>sum(w * G)</c>, so the gradient the plan computes is exactly the <c>G</c> fed to the eager step, and the two
/// runs differ only in the optimizer update. That isolates the thing a mapping can get wrong: the kernel's formula,
/// its hyperparameters, and its state across steps.
/// </para>
/// <para>
/// A network-level comparison (<see cref="FusedOptimizerParityTests"/>) cannot draw this line. Adam-family updates
/// move an element by about lr * sign(g) when its gradient is near zero, so float-reordering in the forward and
/// backward pass flips a few such elements and the maximum divergence is decided by whether one flipped: measured on
/// that probe, correctly-mapped AdamW ranged from 1.7e-5 to 1.8e-3 across initialisations while its single step
/// agrees with the eager step to 6e-8.
/// </para>
/// </remarks>
public class FusedKernelParityTests
{
    private const int Length = 48;
    private const int Length2 = 20;
    private const int Total = Length + Length2;
    // A rank-2 weight and a rank-1 bias, so rank-dependent behaviour (LAMB's no-decay group) is exercised.
    private static readonly int[] Shape1 = { 6, 8 };
    private const int Steps = 40;
    private readonly ITestOutputHelper _output;

    public FusedKernelParityTests(ITestOutputHelper output) => _output = output;


    public static IEnumerable<object[]> Cases()
    {
        foreach (var name in new[]
        {
            "Adam", "AdaMax", "Nadam", "RMSprop", "Adagrad", "Lion", "AdaDelta", "AMSGrad", "AdamW", "RAdam",
            "Rprop", "GradientDescent", "MiniBatchGradientDescent", "StochasticGradientDescent",
            "NesterovAcceleratedGradient", "CoordinateDescent", "Momentum", "TrustRegion", "LBFGS",
            "ProximalGradientDescentL2", "ProximalGradientDescentL1", "FTRL", "ASGD", "Adam8BitBf16",
            "LAMB", "LAMBUnclamped", "Adam8BitInt8", "Adam8BitInt8Mixed",
        })
        {
            yield return new object[] { name };
        }
    }

    private static IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> Create(string name) => name switch
    {
        "Adam" => new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "AdaMax" => new AdaMaxOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdaMaxOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "Nadam" => new NadamOptimizer<float, Tensor<float>, Tensor<float>>(null, new NadamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "RMSprop" => new RootMeanSquarePropagationOptimizer<float, Tensor<float>, Tensor<float>>(null, new RootMeanSquarePropagationOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "Adagrad" => new AdagradOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdagradOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "Lion" => new LionOptimizer<float, Tensor<float>, Tensor<float>>(null, new LionOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 }),
        "AdaDelta" => new AdaDeltaOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdaDeltaOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1.0 }),
        "AMSGrad" => new AMSGradOptimizer<float, Tensor<float>, Tensor<float>>(null, new AMSGradOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "AdamW" => new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "RAdam" => new RAdamOptimizer<float, Tensor<float>, Tensor<float>>(null!, new RAdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "Rprop" => new RpropOptimizer<float, Tensor<float>, Tensor<float>>(null!, new RpropOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "GradientDescent" => new GradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null!, new GradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-1 }),
        "MiniBatchGradientDescent" => new MiniBatchGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null!, new MiniBatchGradientDescentOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-1 }),
        "StochasticGradientDescent" => new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null!, new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-1 }),
        "NesterovAcceleratedGradient" => new NesterovAcceleratedGradientOptimizer<float, Tensor<float>, Tensor<float>>(null!, new NesterovAcceleratedGradientOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "CoordinateDescent" => new CoordinateDescentOptimizer<float, Tensor<float>, Tensor<float>>(null!, new CoordinateDescentOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "Momentum" => new MomentumOptimizer<float, Tensor<float>, Tensor<float>>(null!, new MomentumOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2, UseAdaptiveMomentum = false }),
        "TrustRegion" => new TrustRegionOptimizer<float, Tensor<float>, Tensor<float>>(null!, new TrustRegionOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2, AdaptTrustRegionRadius = false }),
        "LBFGS" => new LBFGSOptimizer<float, Tensor<float>, Tensor<float>>(null!, new LBFGSOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2, UseLineSearch = false }),
        "ProximalGradientDescentL2" => new ProximalGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null!, new ProximalGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-1,
            Regularization = new L2Regularization<float, Tensor<float>, Tensor<float>>(new RegularizationOptions { Strength = 1e-2 }),
        }),
        "ProximalGradientDescentL1" => new ProximalGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null!, new ProximalGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-1,
            Regularization = new L1Regularization<float, Tensor<float>, Tensor<float>>(new RegularizationOptions { Strength = 1e-3 }),
        }),
        "FTRL" => new FTRLOptimizer<float, Tensor<float>, Tensor<float>>(null!, new FTRLOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-1 }),
        "ASGD" => new ASGDOptimizer<float, Tensor<float>, Tensor<float>>(null!, new ASGDOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "Adam8BitBf16" => new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null, new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2, UseBFloat16MomentStorage = true }),
        // Defaults: trust ratio clipped at 10, bias correction on, rank <= 1 parameters excluded from weight decay.
        "LAMB" => new LAMBOptimizer<float, Tensor<float>, Tensor<float>>(null!, new LAMBOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 }),
        "LAMBUnclamped" => new LAMBOptimizer<float, Tensor<float>, Tensor<float>>(null!, new LAMBOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-2,
            ClipTrustRatio = false,
            UseBiasCorrection = false,
            ExcludeBiasFromWeightDecay = false,
        }),
        // Every tensor quantized, in several blocks each.
        "Adam8BitInt8" => new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null, new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2, BlockSize = 16, Min8BitSize = 0 }),
        // The 48-element weight quantized, the 20-element bias below Min8BitSize and full precision.
        "Adam8BitInt8Mixed" => new Adam8BitOptimizer<float, Tensor<float>, Tensor<float>>(null, new Adam8BitOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2, BlockSize = 16, Min8BitSize = 30 }),
        _ => throw new ArgumentException(name),
    };

    private static float[][] Gradients()
    {
        // Seeded, with a spread of magnitudes; the global norm stays well under the eager default clip of 1.
        var rng = new Random(20260927);
        var grads = new float[Steps][];
        for (int t = 0; t < Steps; t++)
        {
            grads[t] = new float[Total];
            for (int i = 0; i < Total; i++)
                grads[t][i] = (float)((rng.NextDouble() - 0.5) * 0.1 * Math.Pow(10, -2 * rng.NextDouble()));
        }
        return grads;
    }

    private static float[] InitialWeights()
    {
        var rng = new Random(7);
        var w = new float[Total];
        for (int i = 0; i < Total; i++) w[i] = (float)(rng.NextDouble() - 0.5);
        return w;
    }

    private static float[] RunFused(AiDotNet.Optimizers.Fused.FusedOptimizerConfig config, float[] w0, float[][] grads)
    {
        // Two parameter tensors of different sizes, so a whole-vector quantity (L-BFGS history, a trust radius) and a
        // per-tensor one (LAMB's trust ratio) cannot be confused without the comparison noticing.
        var engine = new CpuEngine();
        var w1 = new Tensor<float>(Shape1, new Vector<float>(w0[..Length]));
        var w2 = new Tensor<float>(new[] { Length2 }, new Vector<float>(w0[Length..]));
        var g1 = new Tensor<float>(Shape1);
        var g2 = new Tensor<float>(new[] { Length2 });
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(w1, g1), null), engine.ReduceSum(engine.TensorMultiply(w2, g2), null));
            plan = scope.CompileTraining(new[] { w1, w2 });
        }

        using (plan)
        {
            CompiledTapeTrainingStep<float>.ConfigureFusedOptimizer(plan, new[] { w1, w2 }, config.Type, config.LearningRate,
                config.Beta1, config.Beta2, config.Epsilon, config.WeightDecay, config.Schedule, config.UseBf16Moments,
                config.Extras, config.Int8MomentBlockSize, config.Int8MinQuantizedLength, config.DecayOnlyRankTwoAndAbove);

            foreach (var step in grads)
            {
                for (int i = 0; i < Length; i++) g1[i] = step[i];
                for (int i = 0; i < Length2; i++) g2[i] = step[Length + i];
                plan.Step();
            }
        }
        var result = new float[Total];
        w1.GetDataArray().AsSpan(0, Length).CopyTo(result);
        w2.GetDataArray().AsSpan(0, Length2).CopyTo(result.AsSpan(Length));
        return result;
    }

    private static float[] RunEager(IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer, float[] w0, float[][] grads)
    {
        var w1 = new Tensor<float>(Shape1, new Vector<float>(w0[..Length]));
        var w2 = new Tensor<float>(new[] { Length2 }, new Vector<float>(w0[Length..]));
        foreach (var step in grads)
        {
            var g1 = new Tensor<float>(Shape1, new Vector<float>(step[..Length]));
            var g2 = new Tensor<float>(new[] { Length2 }, new Vector<float>(step[Length..]));
            optimizer.Step(new TapeStepContext<float>(
                new[] { w1, w2 }, new Dictionary<Tensor<float>, Tensor<float>> { [w1] = g1, [w2] = g2 }, 0f));
        }
        var result = new float[Total];
        w1.GetDataArray().AsSpan(0, Length).CopyTo(result);
        w2.GetDataArray().AsSpan(0, Length2).CopyTo(result.AsSpan(Length));
        return result;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FusedKernel_MatchesTheEagerStep_OverAFixedGradientSequence(string name)
    {
        var optimizer = Create(name);
        Assert.True(NeuralNetworkBase<float>.TryMapToFusedOptimizerConfig(optimizer, out var config),
            $"{name}: the configuration under test must map onto a fused kernel.");

        var w0 = InitialWeights();
        var grads = Gradients();
        var fused = RunFused(config, w0, grads);
        var eager = RunEager(Create(name), w0, grads);

        double maxDiff = 0, maxMove = 0;
        for (int i = 0; i < Total; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs((double)fused[i] - eager[i]));
            maxMove = Math.Max(maxMove, Math.Abs((double)eager[i] - w0[i]));
        }
        _output.WriteLine($"{name} ({config.Type}): max |fused - eager| = {maxDiff:E3}, eager moved {maxMove:E3}");
        System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), "zzkernel.txt"),
            $"{name} ({config.Type}): diff {maxDiff:E3} move {maxMove:E3}{Environment.NewLine}");

        Assert.True(maxMove > 1e-4, $"{name}: the eager run barely moved ({maxMove:E3}); the comparison would be vacuous.");
        Assert.True(maxDiff <= 1e-5 + 1e-4 * maxMove,
            $"{name}: fused and eager updates differ by {maxDiff:E3} after {Steps} steps (eager moved {maxMove:E3}).");
    }
}
