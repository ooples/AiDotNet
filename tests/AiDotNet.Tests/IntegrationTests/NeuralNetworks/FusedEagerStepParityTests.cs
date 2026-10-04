using System;
using System.Threading.Tasks;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;
using static AiDotNet.Tests.IntegrationTests.NeuralNetworks.FusedOptimizerIntegrationTests;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// A fused step must take the step the eager tape takes. These cover two settings the fused path used to read
/// differently from the eager step: the optimizer's own gradient clip, and a schedule held for a whole epoch.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class FusedEagerStepParityTests
{
    // Relative to the largest parameter update. The fused plan takes beta1, beta2 and epsilon as float, so double
    // training differs from the eager step by a few 1e-8 of the update. The mismatches these tests guard are far
    // larger: the clip was 2e-3 of the update, and a schedule advanced per batch is a different learning rate.
    private const double RelativeTolerance = 1e-6;

    [Fact(Timeout = 120000)]
    public async Task FusedAdam_AppliesTheOptimizersOwnClip_WhenTheNetworkDoesNotClip()
    {
        await Task.CompletedTask;

        // The network clips nothing; Adam keeps its default MaxGradientNorm of 1.0, which its eager step applies.
        // Before the fix the plan received only the network's threshold and trained unclipped.
        var (fused, eager) = Run(
            network => network.SetMaxGradNormForTest(0.0),
            () => new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-3 },
            epochs: 1, stepsPerEpoch: 3, targetScale: 100.0);

        AssertSameSteps(fused, eager, expectedFusedSteps: 3);
    }

    [Fact(Timeout = 120000)]
    public async Task FusedAdam_HoldsAPerEpochSchedule_ForTheWholeEpoch()
    {
        await Task.CompletedTask;

        // StepPerEpoch is the default cadence: the rate changes only at OnEpochEnd. The plan evaluates its schedule
        // every step, so mapping the cosine to the plan advanced it every batch.
        var (fused, eager) = Run(
            _ => { },
            () => new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 1e-2,
                LearningRateScheduler = new CosineAnnealingLRScheduler(1e-2, tMax: 4, etaMin: 1e-4),
                SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
            },
            epochs: 3, stepsPerEpoch: 3, targetScale: 1.0);

        AssertSameSteps(fused, eager, expectedFusedSteps: 9);
    }

    private sealed record RunResult(double[] Initial, double[] Final, long FusedSteps);

    private static (RunResult Fused, RunResult Eager) Run(
        Action<FusedTrainingTestNetworkDouble> configure,
        Func<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>> options,
        int epochs,
        int stepsPerEpoch,
        double targetScale)
    {
        var input = RandomTensor(new[] { 16, 4 }, seed: 42, scale: 1.0);
        var target = RandomTensor(new[] { 16, 2 }, seed: 43, scale: targetScale);
        double[]? initial = null;

        RunResult Train(bool compile)
        {
            var network = BuildNetwork();
            network.Predict(RandomTensor(new[] { 1, 4 }, seed: 99, scale: 1.0));
            if (initial is null) initial = network.GetParameters().ToArray();
            else network.UpdateParameters(new Vector<double>(initial));
            configure(network);
            // A fresh scheduler per run: the options copy constructor shares the scheduler instance.
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(network, options());

            var originalOptions = TensorCodecOptions.Current;
            try
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                CompiledTapeTrainingStep<double>.ResetFusedStepCount();
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    for (int step = 0; step < stepsPerEpoch; step++)
                        network.TrainPublic(input, target, optimizer);
                    optimizer.OnEpochEnd();
                }

                return new RunResult(initial, network.GetParameters().ToArray(),
                    CompiledTapeTrainingStep<double>.GetFusedStepCount());
            }
            finally
            {
                TensorCodecOptions.SetCurrent(originalOptions);
                CompiledTapeTrainingStep<double>.Invalidate();
                CompiledTapeTrainingStep<double>.ResetFusedStepCount();
            }
        }

        var fused = Train(compile: true);
        var eager = Train(compile: false);
        return (fused, eager);
    }

    private static void AssertSameSteps(RunResult fused, RunResult eager, long expectedFusedSteps)
    {
        Assert.Equal(expectedFusedSteps, fused.FusedSteps);
        Assert.Equal(0, eager.FusedSteps);

        double maxUpdate = 0.0, maxDivergence = 0.0;
        for (int i = 0; i < fused.Final.Length; i++)
        {
            maxUpdate = Math.Max(maxUpdate, Math.Abs(eager.Final[i] - eager.Initial[i]));
            maxDivergence = Math.Max(maxDivergence, Math.Abs(fused.Final[i] - eager.Final[i]));
        }

        Assert.True(maxUpdate > 1e-6, $"training did not move the parameters (max update {maxUpdate:E3}).");
        Assert.True(maxDivergence < RelativeTolerance * maxUpdate,
            $"fused training diverged from the eager step by {maxDivergence:E3} (max update {maxUpdate:E3}).");
    }

    private static FusedTrainingTestNetworkDouble BuildNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);
        var network = new FusedTrainingTestNetworkDouble(architecture);
        network.AddLayer(new DenseLayer<double>(8));
        network.AddLayer(new DenseLayer<double>(2));
        return network;
    }

    private static Tensor<double> RandomTensor(int[] shape, int seed, double scale)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (random.NextDouble() * 2.0 - 1.0) * scale;
        return tensor;
    }
}
