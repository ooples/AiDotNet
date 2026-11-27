using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Gradient accumulation in one process and data-parallel training in a world of one rank run the same step: the
/// same accumulated gradient, unscaled, clipped and applied by the same code. They must therefore train bit for bit
/// alike. The reported gradient norm is the norm before clipping on both paths, as torch.nn.utils.clip_grad_norm_
/// returns it.
/// </summary>
public class GradientAccumulationStepParityTests
{
    private const int MaxSteps = 6;
    private const int AccumulationSteps = 2;

    private readonly ITestOutputHelper _output;

    public GradientAccumulationStepParityTests(ITestOutputHelper output) => _output = output;

    private static async Task<(Vector<float> Weights, double[] GradNorms)> Train(double maxGradNorm, bool worldOfOne)
    {
        // Only the network clips (the optimizer's own clip is off), so the clip under test is the step's only one.
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            MaxIterations = 1000,
            UseAdaptiveLearningRate = false,
            UseEarlyStopping = false,
            Tolerance = 0.0,
            EnableGradientClipping = false,
            FitnessCalculator = new MeanSquaredErrorFitnessCalculator<float, Tensor<float>, Tensor<float>>(),
        });
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(4),
            new DenseLayer<float>(8, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 1, layers: layers);
        var model = new FeedForwardNeuralNetwork<float>(arch, optimizer: optimizer, lossFunction: new MeanSquaredErrorLoss<float>(),
            options: new FeedForwardNeuralNetworkOptions { MaxGradNorm = maxGradNorm, Seed = 17 });
        var monitor = new StreamingStepMonitorTests.RecordingMonitor();
        var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureDataLoader(StreamingStepMonitorTests.Loader())
            .ConfigureStreamingTraining(new StreamingTrainingOptions<float, Tensor<float>, Tensor<float>>
            {
                Seed = 5,
                MaxSteps = MaxSteps,
                GradientAccumulationSteps = AccumulationSteps,
                MonitorEveryNSteps = 1,
                MonitorGradNorm = true,
            })
            .ConfigureTrainingMonitor(monitor);
        InMemoryCommunicationBackend<float>? backend = null;
        if (worldOfOne)
        {
            backend = new InMemoryCommunicationBackend<float>(0, 1, "accum_parity_" + Guid.NewGuid().ToString("N"));
            builder = builder.ConfigureDistributedTraining(backend, DistributedStrategy.DDP);
        }

        try
        {
            await builder.BuildAsync();
        }
        finally
        {
            if (backend is { IsInitialized: true }) backend.Shutdown();
        }

        return (model.GetParameters(), monitor.Values("grad_norm").ToArray());
    }

    [Theory(Timeout = 120000)]
    [InlineData(0.05)] // the clip scales every step
    [InlineData(1e9)] // the clip computes the norm and never scales
    public async Task OneProcessAccumulation_TrainsBitwiseLikeAWorldOfOne_AndBothReportThePreClipNorm(double maxGradNorm)
    {
        var (local, localNorms) = await Train(maxGradNorm, worldOfOne: false);
        var (world, worldNorms) = await Train(maxGradNorm, worldOfOne: true);
        // The step-1 gradient does not depend on the clip (same initial weights, same batch), so the step-1 norm of
        // the run that never scales is the pre-clip norm the clipped run must report at step 1.
        var (_, unclippedNorms) = await Train(1e9, worldOfOne: false);

        double maxDiff = 0;
        int differ = 0;
        for (int i = 0; i < local.Length; i++)
        {
            double d = Math.Abs(local[i] - world[i]);
            if (BitConverter.SingleToInt32Bits(local[i]) != BitConverter.SingleToInt32Bits(world[i])) differ++;
            maxDiff = Math.Max(maxDiff, d);
        }

        _output.WriteLine($"max grad norm {maxGradNorm}: {differ} of {local.Length} weights differ, max |d| {maxDiff:E3}");
        _output.WriteLine($"grad_norm one process {string.Join(", ", localNorms.Select(v => v.ToString("R")))}");
        _output.WriteLine($"grad_norm world of 1  {string.Join(", ", worldNorms.Select(v => v.ToString("R")))}");
        _output.WriteLine($"step-1 pre-clip norm  {unclippedNorms[0]:R}");

        Assert.Equal(0, differ);
        Assert.Equal(MaxSteps, localNorms.Length);
        Assert.Equal(localNorms, worldNorms);
        Assert.Equal(unclippedNorms[0], localNorms[0]);
        Assert.Equal(unclippedNorms[0], worldNorms[0]);
        // The tiny regression model's step-1 norm is far above 0.05, so a post-clip norm would read about 0.05.
        Assert.True(unclippedNorms[0] > 10 * 0.05, $"the step-1 norm {unclippedNorms[0]} does not exercise the clip");
    }
}
