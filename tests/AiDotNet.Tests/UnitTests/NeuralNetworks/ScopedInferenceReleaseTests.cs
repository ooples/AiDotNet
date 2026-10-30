using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The CNNs whose generated stacks keep nothing between Predict calls free their GPU intermediates per call
/// (<c>ReleasesInferenceIntermediates</c>). #2383: DenseNet without it allocated a fresh device buffer per op and grew
/// to 4.6-7.4 GB at b1-b16. Caller-supplied layers may carry a tensor between calls, so they must keep the default.
/// </summary>
/// <remarks>
/// The release is observed through an enclosing <see cref="TensorScope"/> around one Predict. Predict's own scope frees
/// the call's GPU results when it ends and hands only the kept output to the enclosing scope, so that scope then tracks
/// at most the output. Without the release every GPU result of the call lands in the enclosing scope instead. Results
/// on the CPU are never tracked, which is why these run on a GPU.
/// </remarks>
[Collection("NonParallelIntegration")]
public class ScopedInferenceReleaseTests
{
    private const int Size = 32;
    private const int Classes = 10;

    public enum CnnFamily
    {
        DenseNet,
        ResNet,
        Vgg,
    }

    private static NeuralNetworkArchitecture<float> Arch(List<ILayer<float>>? layers) => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputHeight: Size, inputWidth: Size, inputDepth: 3, outputSize: Classes,
        layers: layers);

    // Two convolutions before the head, so a stack that does not release leaves several GPU results behind.
    private static List<ILayer<float>> CustomLayers() => new()
    {
        new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1,
            activationFunction: new ReLUActivation<float>()),
        new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1,
            activationFunction: new ReLUActivation<float>()),
        new FlattenLayer<float>(),
        new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()),
    };

    private static NeuralNetworkBase<float> Build(CnnFamily family, bool custom)
    {
        var arch = Arch(custom ? CustomLayers() : null);
        return family switch
        {
            CnnFamily.DenseNet => new DenseNetNetwork<float>(arch, new DenseNetConfiguration(
                variant: DenseNetVariant.Custom, numClasses: Classes, inputHeight: Size, inputWidth: Size,
                inputChannels: 3, growthRate: 12, compressionFactor: 0.5, customBlockLayers: [2, 2])),
            CnnFamily.ResNet => new ResNetNetwork<float>(arch, new ResNetConfiguration(ResNetVariant.ResNet18, Classes, Size, Size)),
            CnnFamily.Vgg => new VGGNetwork<float>(arch, new VGGConfiguration(VGGVariant.VGG11, Classes, Size, Size)),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
    }

    private static Tensor<float> Input()
    {
        var rng = new Random(5);
        var input = new Tensor<float>(new[] { 2, 3, Size, Size });
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        return input;
    }

    private static OpenClBackend RequireDevice()
    {
        var backend = new OpenClBackend();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(backend.IsAvailable, "The required OpenCL lane has no usable device.");
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            Skip.If(true, "No OpenCL device is available.");
        }

        return backend;
    }

    /// <summary>
    /// Runs one Predict inside an enclosing scope and returns its values and the number of GPU results the enclosing
    /// scope was left holding.
    /// </summary>
    private static (float[] Values, int ResultsLeft) PredictInEnclosingScope(NeuralNetworkBase<float> network, Tensor<float> input)
    {
        using var enclosing = TensorScope.Begin();
        var output = network.Predict(input);
        int left = enclosing.TrackedCount;
        return (output.ToArray(), left);
    }

    // A fresh model's first GPU Predict creates its lazy weights INSIDE the per-call scope. The scope must free only
    // that call's intermediates, never the weights it saw created, or the second call reads freed buffers.
    [SkippableTheory]
    [InlineData(CnnFamily.DenseNet)]
    [InlineData(CnnFamily.ResNet)]
    [InlineData(CnnFamily.Vgg)]
    [Trait("Category", "GPU")]
    public void Fresh_model_frees_each_predicts_intermediates_and_still_matches_the_cpu(CnnFamily family)
    {
        using var backend = RequireDevice();
        using var direct = new DirectGpuEngine(backend);
        using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var network = Build(family, custom: false);
            var input = Input();

            var first = network.Predict(input).ToArray();
            var (second, left) = PredictInEnclosingScope(network, input);
            // The call's convolution results were freed when Predict ended; at most the output it kept remains.
            Assert.True(left <= 1, $"{family}: Predict left {left} GPU results alive after it returned.");

            AiDotNetEngine.Current = new CpuEngine();
            var expected = network.Predict(input).ToArray();
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(first[i] - second[i]) <= 1e-5f * Math.Max(1f, Math.Abs(first[i])),
                    $"{family}: second GPU Predict differs at {i}: {second[i]} vs {first[i]}");
                Assert.True(Math.Abs(expected[i] - first[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])),
                    $"{family}: GPU {first[i]} vs CPU {expected[i]} at {i}");
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    // Caller-supplied layers may carry a tensor between calls, so Predict must not free what they produced.
    [SkippableTheory]
    [InlineData(CnnFamily.DenseNet)]
    [InlineData(CnnFamily.ResNet)]
    [InlineData(CnnFamily.Vgg)]
    [Trait("Category", "GPU")]
    public void Custom_layers_keep_their_predict_results(CnnFamily family)
    {
        using var backend = RequireDevice();
        using var direct = new DirectGpuEngine(backend);
        using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var network = Build(family, custom: true);
            var input = Input();
            network.Predict(input);   // warm: lazy weights, chain validation

            var (_, left) = PredictInEnclosingScope(network, input);
            Assert.True(left > 1, $"{family}: Predict freed the custom layers' results ({left} left to the caller).");
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
