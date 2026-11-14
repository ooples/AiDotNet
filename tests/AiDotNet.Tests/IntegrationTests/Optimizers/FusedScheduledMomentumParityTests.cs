using System;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// SGD momentum with a learning-rate schedule on the fused compiled step. MomentumOptimizer's velocity carries the
/// learning rate (v = mu*v + lr_t*g; p -= v); the classical fused kernel applies the current rate to the whole
/// accumulated velocity (v = mu*v + g; p -= lr_t*v), which differs as soon as the rate moves. So every scheduled
/// momentum optimizer declined the fused path: ResNet, VGG, YOLOv9 and the other paper-scheduled models trained
/// eagerly. The fused step now uses the learning-rate-in-velocity form.
/// </summary>
[Collection("FusedTrainingSerial")]
public class FusedScheduledMomentumParityTests
{
    private const int Steps = 12;

    private static NeuralNetworkArchitecture<float> MakeArch() =>
        new(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple, inputSize: 8, outputSize: 3);

    private static (Tensor<float> x, Tensor<float> y) MakeData()
    {
        var x = new Tensor<float>(new[] { 4, 8 });
        var y = new Tensor<float>(new[] { 4, 3 });
        for (int b = 0; b < 4; b++)
        {
            for (int f = 0; f < 8; f++) x[b, f] = (((b * 8 + f) % 7) - 3) * 0.1f;
            for (int o = 0; o < 3; o++) y[b, o] = (((b * 3 + o) % 5) - 2) * 0.2f;
        }
        return (x, y);
    }

    private static MomentumOptimizer<float, Tensor<float>, Tensor<float>> Momentum(ILearningRateScheduler? scheduler) =>
        new(null, new MomentumOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.05,
            InitialMomentum = 0.9,
            UseAdaptiveLearningRate = false,
            UseAdaptiveMomentum = false,
            LearningRateScheduler = scheduler,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });

    private static (Vector<float> Parameters, long FusedSteps) Run(
        Func<MomentumOptimizer<float, Tensor<float>, Tensor<float>>> factory, Vector<float> init, bool compiled)
    {
        var optimizer = factory();
        var model = new FeedForwardNeuralNetwork<float>(MakeArch(), optimizer, new MeanSquaredErrorLoss<float>());
        model.UpdateParameters(init);
        var (x, y) = MakeData();
        var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = options.EnableCompilation;
        try
        {
            options.EnableCompilation = compiled;
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            for (int s = 0; s < Steps; s++) model.Train(x, y);
            return (model.GetParameters(), CompiledTapeTrainingStep<float>.GetFusedStepCount());
        }
        finally
        {
            options.EnableCompilation = saved;
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        }
    }

    private static double MaxAbsDifference(Vector<float> a, Vector<float> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
        return max;
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ScheduledMomentum_TrainsFused_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // A rate that moves every step: warm up over 4 steps, then cosine decay over the next 8.
        Func<MomentumOptimizer<float, Tensor<float>, Tensor<float>>> scheduled =
            () => Momentum(new WarmupStableDecayScheduler(0.05, 4, 4, 8, WarmupStableDecayScheduler.DecayShape.Cosine));
        Func<MomentumOptimizer<float, Tensor<float>, Tensor<float>>> constant = () => Momentum(null);
        var init = new FeedForwardNeuralNetwork<float>(MakeArch(), lossFunction: new MeanSquaredErrorLoss<float>()).GetParameters();

        var fused = Run(scheduled, init, compiled: true);
        var eager = Run(scheduled, init, compiled: false);
        // Control: the same momentum at a constant rate, where both forms agree; its drift is float round-off.
        double floor = MaxAbsDifference(Run(constant, init, compiled: true).Parameters, Run(constant, init, compiled: false).Parameters);

        Assert.Equal(Steps, fused.FusedSteps);
        Assert.True(MaxAbsDifference(eager.Parameters, init) > 1e-3, "training did not move the parameters.");
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence <= 4 * Math.Max(floor, 1e-6),
            $"scheduled momentum: fused diverged from eager by {divergence:E3}; constant-rate momentum drifts {floor:E3}.");
    }
}
