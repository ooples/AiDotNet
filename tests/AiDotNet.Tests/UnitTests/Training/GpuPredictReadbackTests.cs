using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// #2385: a warm small-model Predict on a GPU engine paid ~1 ms at every batch size. After GPU training the dense
/// weights are device-owned, so the host-array CompiledMlp tier downloaded a snapshot of every weight on each call,
/// rejected it as not live, and fell through to MlpForward; Predict then downloaded the result to detach it from an
/// arena that had rented nothing. A warm Predict must not build the host plan, and its result must stay on the device.
/// </summary>
[Collection("NonParallelIntegration")]
public class GpuPredictReadbackTests
{
    private const int Features = 32;
    private const int Classes = 4;
    private const int Batch = 3;

    private static void RequireDevice(OpenClBackend backend)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(backend.IsAvailable, "The required OpenCL lane has no usable device.");
        Skip.IfNot(backend.IsAvailable, "No OpenCL device is available.");
    }

    private static Tensor<float> Seeded(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static FeedForwardNeuralNetwork<float> BuildMlp()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(16, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: Features,
            outputSize: Classes,
            layers: layers);
        return new FeedForwardNeuralNetwork<float>(arch, lossFunction: new CrossEntropyWithLogitsLoss<float>());
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void Warm_mlp_predict_after_gpu_training_reads_nothing_back_and_matches_the_cpu()
    {
        using var backend = new OpenClBackend(); RequireDevice(backend);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var network = BuildMlp();
            var input = Seeded(7, Batch, Features);
            var labels = new Tensor<float>(new[] { Batch, Classes });
            for (int b = 0; b < Batch; b++) labels[b, b % Classes] = 1f;

            // Enough GPU steps for the resident training step to take the weights: they end device-owned.
            for (int step = 0; step < 10; step++) network.Train(input, labels);
            network.Predict(input);         // warm: chain validation, uploads, kernels
            network.Predict(input);

            Tensor<float> output;
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                output = network.Predict(input);
                Assert.True(scope.Downloads == 0,
                    $"A warm Predict downloaded {scope.Downloads} time(s):\n" + string.Join("\n", scope.Events));
            }

            // The host-array CompiledMlp plan is the CPU tier; on a GPU engine it snapshotted every weight per call.
            var plan = typeof(FeedForwardNeuralNetwork<float>)
                .GetField("_compiledMlpPlan", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                ?? throw new InvalidOperationException("FeedForwardNeuralNetwork no longer has _compiledMlpPlan; update this test.");
            Assert.Null(plan.GetValue(network));
            // The result stays on the device until read, instead of being copied into a fresh host tensor per call.
            Assert.True(output.HasPendingGpuData, "Predict copied its GPU result to the host.");

            var actual = output.ToArray();
            AiDotNetEngine.Current = new CpuEngine();
            var expected = network.Predict(input).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, Math.Abs(expected[i])),
                    $"output[{i}]: gpu {actual[i]} vs cpu {expected[i]}");
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
