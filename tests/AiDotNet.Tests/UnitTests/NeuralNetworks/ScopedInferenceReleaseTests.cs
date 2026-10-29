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
[Collection("NonParallelIntegration")]
public class ScopedInferenceReleaseTests
{
    private const int Size = 32;
    private const int Classes = 10;

    private static bool Releases(NeuralNetworkBase<float> network)
    {
        var property = typeof(NeuralNetworkBase<float>).GetProperty("ReleasesInferenceIntermediates",
                           System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                       ?? throw new InvalidOperationException("NeuralNetworkBase no longer has ReleasesInferenceIntermediates.");
        return (bool)(property.GetValue(network) ?? throw new InvalidOperationException("Null property value."));
    }

    private static NeuralNetworkArchitecture<float> Arch(List<ILayer<float>>? layers) => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputHeight: Size, inputWidth: Size, inputDepth: 3, outputSize: Classes,
        layers: layers);

    private static List<ILayer<float>> CustomLayers() => new()
    {
        new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1,
            activationFunction: new ReLUActivation<float>()),
        new FlattenLayer<float>(),
        new DenseLayer<float>(Classes, (IActivationFunction<float>)new IdentityActivation<float>()),
    };

    private static NeuralNetworkBase<float> Build(string family, bool custom)
    {
        var arch = Arch(custom ? CustomLayers() : null);
        return family switch
        {
            "densenet" => new DenseNetNetwork<float>(arch, new DenseNetConfiguration(
                variant: DenseNetVariant.Custom, numClasses: Classes, inputHeight: Size, inputWidth: Size,
                inputChannels: 3, growthRate: 12, compressionFactor: 0.5, customBlockLayers: [2, 2])),
            "resnet" => new ResNetNetwork<float>(arch, new ResNetConfiguration(ResNetVariant.ResNet18, Classes, Size, Size)),
            "vgg" => new VGGNetwork<float>(arch, new VGGConfiguration(VGGVariant.VGG11, Classes, Size, Size)),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
    }

    [Theory]
    [InlineData("densenet")]
    [InlineData("resnet")]
    [InlineData("vgg")]
    public void Generated_stack_releases_and_custom_layers_do_not(string family)
    {
        Assert.True(Releases(Build(family, custom: false)), $"{family}: the generated stack should release per call.");
        Assert.False(Releases(Build(family, custom: true)), $"{family}: caller-supplied layers must keep the default.");
    }

    // A fresh model's first GPU Predict creates its lazy weights INSIDE the per-call scope. The scope must free only
    // that call's intermediates, never the weights it saw created, or the second call reads freed buffers.
    [SkippableTheory]
    [InlineData("densenet")]
    [InlineData("resnet")]
    [InlineData("vgg")]
    [Trait("Category", "GPU")]
    public void Fresh_model_predicts_twice_on_the_gpu_and_matches_the_cpu(string family)
    {
        using var backend = new OpenClBackend();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(backend.IsAvailable, "The required OpenCL lane has no usable device.");
        Skip.IfNot(backend.IsAvailable, "No OpenCL device is available.");
        using var direct = new DirectGpuEngine(backend);
        using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;
            var network = Build(family, custom: false);
            var rng = new Random(5);
            var input = new Tensor<float>(new[] { 2, 3, Size, Size });
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            var first = network.Predict(input).ToArray();
            var second = network.Predict(input).ToArray();
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
}