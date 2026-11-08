using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.Safety;
using AiDotNet.Training;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// A model whose declared objectives train different parameters of the same network with different optimizers
/// (StegaStamp after its secret-only phase: the encoder and decoder's Adam step, then the WGAN critic's RMSProp step,
/// every batch). Each objective's optimizer keeps its own fused plan and moments. Through one shared session every
/// switch between the objectives dropped the other's, so each fused Adam step was Adam's first (an update of
/// lr * sign(g)) and fused training left the eager trajectory from step 2. The critic's RMSProp starts its mean square
/// at 1 (TensorFlow 1's), which the fused kernel cannot, so it steps eagerly; the switch still reset the shared session.
/// The learning-rate schedule advances once per step on either path.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class SameNetworkObjectivesFusedTrainingTests
{
    private const int Size = 32, Bits = 16, Batch = 4;
    // Measured: at most 1.5% apart over the eight steps (the chaotic drift below); 33% apart at step 2 before each
    // objective kept its own moments (fused 6.49e-4, eager 4.89e-4).
    private const double RelativeTolerance = 0.1;

    private readonly ITestOutputHelper _output;

    public SameNetworkObjectivesFusedTrainingTests(ITestOutputHelper output) => _output = output;

    // noImageLossSteps = 0: both objectives from the first step.
    private static StegaStampWatermarker<float> Build(int warmupSteps = 0, int noImageLossSteps = 0) => new(new StegaStampOptions
    {
        ImageSize = Size, SecretBits = Bits, UseErrorCorrection = false, Seed = 2, LearningRate = 1e-3,
        LearningRateWarmupSteps = warmupSteps, NoImageLossSteps = noImageLossSteps,
    });

    // The encoder and decoder optimizer that trains from step 0, which carries the warmup.
    private static GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> FirstOptimizer(StegaStampWatermarker<float> model, bool secretPhase)
    {
        string name = secretPhase ? "_secretOptimizer" : "_optimizer";
        var field = typeof(StegaStampWatermarker<float>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.True(field is not null, $"StegaStampWatermarker has no field {name}; update this test's reflection.");
        return Assert.IsAssignableFrom<GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>>>(field!.GetValue(model));
    }

    private static void Run(StegaStampWatermarker<float> model, bool compile, int steps, Action<int>? afterStep = null)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<float>.Invalidate();
            var rng = RandomHelper.CreateSeededRandom(5);
            for (int step = 1; step <= steps; step++)
            {
                var secrets = new Tensor<float>(new[] { Batch, Bits });
                for (int i = 0; i < secrets.Length; i++) secrets.SetFlat(i, rng.Next(2));
                model.Train(StegaStampWatermarkerTests.Images(rng, Batch), secrets);
                afterStep?.Invoke(step);
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<float>.Invalidate();
        }
    }

    // The mean |update| of the encoder and decoder at each step, which Adam's moments shrink as the run goes on.
    private static double[] MeanUpdates(StegaStampWatermarker<float> model, bool compile, int steps)
    {
        var updates = new double[steps];
        var before = EncoderDecoder(model);
        Run(model, compile, steps, step =>
        {
            var now = EncoderDecoder(model);
            double sum = 0;
            for (int i = 0; i < now.Length; i++) sum += Math.Abs(now[i] - before[i]);
            updates[step - 1] = sum / now.Length;
            before = now;
        });
        return updates;
    }

    /// <summary>
    /// Each fused step is the Adam step the eager path takes. The weights themselves are not compared past the first
    /// steps: the distortions round (JPEG) and the critic's weights are clipped, so the two arms' rounding-order
    /// differences (3e-6 after step 2) grow chaotically. The size of each step is what the lost moments changed: with
    /// them dropped every step, every fused update was lr * sign(g): the mean |update| grew to 6.5e-4 at step 2 while the
    /// eager one fell to 4.9e-4.
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async System.Threading.Tasks.Task FusedSteps_KeepEachObjectivesAdamMoments()
    {
        await System.Threading.Tasks.Task.Yield();
        const int steps = 8;
        var fused = Build();
        var eager = Build();
        Assert.Equal(EncoderDecoder(fused), EncoderDecoder(eager));

        var f = MeanUpdates(fused, compile: true, steps);
        var e = MeanUpdates(eager, compile: false, steps);
        Assert.Equal(steps, fused.CustomFusedStepCount);   // the encoder and decoder
        Assert.Equal(steps, fused.CustomEagerStepCount);   // the critic

        for (int step = 0; step < steps; step++)
        {
            _output.WriteLine($"step {step + 1}: mean |update| fused {f[step]:E4}, eager {e[step]:E4}");
            Assert.True(e[step] > 0 && double.IsFinite(e[step]) && double.IsFinite(f[step]),
                $"step {step + 1}: the ratio needs a positive, finite eager update; fused {f[step]:E4}, eager {e[step]:E4}");
            Assert.True(Math.Abs(f[step] / e[step] - 1) <= RelativeTolerance,
                $"step {step + 1}: the fused mean |update| is {f[step]:E4}, the eager {e[step]:E4}");
        }
    }

    private static float[] EncoderDecoder(StegaStampWatermarker<float> model)
    {
        var flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var layers = new[] { "_encoderLayers", "_decoderLayers" }
            .SelectMany(name => (List<ILayer<float>>)typeof(StegaStampWatermarker<float>).GetField(name, flags)!.GetValue(model)!);
        return layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();
    }

    [Theory(Timeout = 600_000)]
    [InlineData(true)]
    [InlineData(false)]
    public async System.Threading.Tasks.Task LearningRateSchedule_AdvancesOncePerStep(bool compile)
    {
        await System.Threading.Tasks.Task.Yield();
        const int warmup = 4, steps = 6;
        var model = Build(warmupSteps: warmup, noImageLossSteps: steps);
        var optimizer = FirstOptimizer(model, secretPhase: true);
        var reference = new LinearWarmupScheduler(1e-3, warmup);
        Run(model, compile, steps, step =>
        {
            double expected = reference.Step();
            // The schedule must move: during warmup the rate is below the base rate.
            if (step < warmup)
                Assert.True(expected < 1e-3, $"the reference schedule gives {expected:R} after warmup step {step}");
            Assert.True(Math.Abs(optimizer.GetCurrentLearningRate() - expected) <= 1e-12,
                $"after step {step} the learning rate is {optimizer.GetCurrentLearningRate():R}, the schedule's {expected:R}");
        });
    }
}
