using System;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Training;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// The fused compiled step must train exactly like the eager tape step for two optimizer settings that
/// live outside the kernel's own hyperparameters: the optimizer's global-norm gradient clip, and a
/// learning-rate scheduler stepped per epoch.
/// </summary>
/// <remarks>
/// Both diverged silently. The eager Adam step clips by global norm inside its own Step, which the fused
/// kernel never runs, so the compiled plan trained unclipped. A per-epoch scheduler was mapped to a
/// per-step schedule shape, so the plan annealed over tMax batches while the eager optimizer held its rate
/// for the epoch; the rate change at OnEpochEnd then read as configuration drift and stopped fused training.
/// Each parity check is measured against the same optimizer with the feature off, run the same way.
/// </remarks>
[Collection("FusedTrainingSerial")]
public class FusedOptimizerScheduleAndClipParityTests
{
    /// <summary>
    /// How far a fused run may drift from eager, as a multiple of the drift of the same optimizer with the
    /// feature under test turned off. The fused kernel takes the learning rate, betas and epsilon as float
    /// (FusedOptimizerConfig), so even plain Adam differs from eager in double by that rounding (measured
    /// 3.1e-8 over 9 steps here). The defects these tests guard differ by 1e-3 or throw.
    /// </summary>
    private const double ControlFloorMultiple = 4.0;

    private static NeuralNetworkArchitecture<double> MakeArch() =>
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 3);

    private static (Tensor<double> x, Tensor<double> y) MakeData()
    {
        var x = new Tensor<double>(new[] { 4, 8 });
        var y = new Tensor<double>(new[] { 4, 3 });
        for (int b = 0; b < 4; b++)
        {
            for (int f = 0; f < 8; f++) x[b, f] = (((b * 8 + f) % 7) - 3) * 0.1;
            for (int o = 0; o < 3; o++) y[b, o] = (((b * 3 + o) % 5) - 2) * 0.2;
        }
        return (x, y);
    }

    private sealed record RunResult(Vector<double> Parameters, long FusedSteps);

    /// <summary>
    /// Trains <paramref name="epochs"/> epochs of <paramref name="stepsPerEpoch"/> batches, calling the
    /// optimizer's OnEpochEnd between epochs as an epoch loop does.
    /// </summary>
    private static RunResult Run(
        Func<AdamOptimizer<double, Tensor<double>, Tensor<double>>> optimizerFactory,
        Vector<double>? initialParameters,
        bool compiled,
        int epochs,
        int stepsPerEpoch)
    {
        var optimizer = optimizerFactory();
        var model = new FeedForwardNeuralNetwork<double>(MakeArch(), optimizer, new MeanSquaredErrorLoss<double>());
        if (initialParameters is not null)
            model.UpdateParameters(initialParameters);
        var (x, y) = MakeData();

        var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = options.EnableCompilation;
        try
        {
            options.EnableCompilation = compiled;
            CompiledTapeTrainingStep<double>.Invalidate();
            CompiledTapeTrainingStep<double>.ResetFusedStepCount();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int step = 0; step < stepsPerEpoch; step++)
                    model.Train(x, y);
                optimizer.OnEpochEnd();
            }
            return new RunResult(model.GetParameters(), CompiledTapeTrainingStep<double>.GetFusedStepCount());
        }
        finally
        {
            options.EnableCompilation = saved;
            CompiledTapeTrainingStep<double>.Invalidate();
            CompiledTapeTrainingStep<double>.ResetFusedStepCount();
        }
    }

    private static Vector<double> InitialParameters() =>
        new FeedForwardNeuralNetwork<double>(MakeArch(), lossFunction: new MeanSquaredErrorLoss<double>()).GetParameters();

    private static double MaxAbsDifference(Vector<double> a, Vector<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0.0;
        for (int i = 0; i < a.Length; i++)
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Fact]
    public void PerEpochScheduler_FusedHoldsTheEpochRateAndFollowsOnEpochEnd()
    {
        AdamOptimizer<double, Tensor<double>, Tensor<double>> Factory() =>
            new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-2,
                EnableGradientClipping = false,
                // StepPerEpoch is the default; stated so the test does not depend on it.
                SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
                LearningRateScheduler = new CosineAnnealingLRScheduler(baseLearningRate: 1e-2, tMax: 4, etaMin: 0.0),
            });

        // Positive control: the scheduler really moves the rate at an epoch boundary, so a plan that held
        // the first rate forever would fail the comparison below.
        var probe = Factory();
        double before = probe.GetCurrentLearningRate();
        probe.OnEpochEnd();
        Assert.True(probe.GetCurrentLearningRate() < before * 0.9,
            $"Cosine scheduler did not lower the rate at OnEpochEnd ({before} -> {probe.GetCurrentLearningRate()}).");

        AdamOptimizer<double, Tensor<double>, Tensor<double>> Unscheduled() =>
            new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-2,
                EnableGradientClipping = false,
            });

        var init = InitialParameters();
        var fused = Run(Factory, init, compiled: true, epochs: 3, stepsPerEpoch: 3);
        var eager = Run(Factory, init, compiled: false, epochs: 3, stepsPerEpoch: 3);
        double floor = MaxAbsDifference(
            Run(Unscheduled, init, compiled: true, epochs: 3, stepsPerEpoch: 3).Parameters,
            Run(Unscheduled, init, compiled: false, epochs: 3, stepsPerEpoch: 3).Parameters);

        Assert.Equal(9, fused.FusedSteps);
        Assert.Equal(0, eager.FusedSteps);
        Assert.True(MaxAbsDifference(eager.Parameters, init) > 1e-4, "Training did not move the parameters.");
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence <= ControlFloorMultiple * floor,
            $"Fused training with a per-epoch scheduler diverged from eager by {divergence:E3}; " +
            $"unscheduled Adam drifts {floor:E3}.");
    }

    [Fact]
    public void OptimizerGlobalNormClip_IsAppliedByTheFusedStep()
    {
        // Epsilon = 1 makes Adam's update lr * g / (|g| + 1), proportional to g for small gradients, so a
        // clip that scales g changes the update instead of being normalised away.
        AdamOptimizer<double, Tensor<double>, Tensor<double>> Factory(bool clip) =>
            new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-2,
                Epsilon = 1.0,
                EnableGradientClipping = clip,
                GradientClippingMethod = GradientClippingMethod.ByNorm,
                MaxGradientNorm = 1e-3,
            });

        var init = InitialParameters();
        var fused = Run(() => Factory(clip: true), init, compiled: true, epochs: 1, stepsPerEpoch: 4);
        var eager = Run(() => Factory(clip: true), init, compiled: false, epochs: 1, stepsPerEpoch: 4);
        var fusedUnclipped = Run(() => Factory(clip: false), init, compiled: true, epochs: 1, stepsPerEpoch: 4);
        var eagerUnclipped = Run(() => Factory(clip: false), init, compiled: false, epochs: 1, stepsPerEpoch: 4);
        double floor = MaxAbsDifference(fusedUnclipped.Parameters, eagerUnclipped.Parameters);

        Assert.Equal(4, fused.FusedSteps);
        // Positive control: the clip binds at this bound, so a fused step that skipped it would differ.
        Assert.True(MaxAbsDifference(eager.Parameters, eagerUnclipped.Parameters) > 1e-6,
            "MaxGradientNorm did not bind; the clip comparison would be vacuous.");
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence <= ControlFloorMultiple * floor,
            $"Fused training ignored the optimizer's gradient clip: diverged from eager by {divergence:E3}; " +
            $"unclipped Adam drifts {floor:E3}.");
    }

    [Fact]
    public void WarmupThenEpochScheduler_StaysOnTheEagerPath()
    {
        // The fused path never calls OnBatchEnd, so it cannot advance a per-batch warmup; the optimizer
        // must decline fusion rather than train on a schedule it cannot follow.
        AdamOptimizer<double, Tensor<double>, Tensor<double>> Factory() =>
            new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-2,
                SchedulerStepMode = SchedulerStepMode.WarmupThenEpoch,
                LearningRateScheduler = new LinearWarmupScheduler(baseLearningRate: 1e-2, warmupSteps: 4, totalSteps: 20),
            });

        var run = Run(Factory, InitialParameters(), compiled: true, epochs: 1, stepsPerEpoch: 2);

        Assert.Equal(0, run.FusedSteps);
    }
}
