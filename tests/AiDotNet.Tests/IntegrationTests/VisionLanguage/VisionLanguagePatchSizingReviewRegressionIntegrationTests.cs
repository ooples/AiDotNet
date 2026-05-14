#nullable disable
using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.InstructionTuned;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.VisionLanguage;

public class VisionLanguagePatchSizingReviewRegressionIntegrationTests
{
    [Fact]
    public void InstructionTunedModels_WithNonDivisibleTokenBudgets_RoundPatchSizeUp()
    {
        var architecture = CreateArchitectureWithCustomLayers();

        Assert.Equal(8, InvokeComputePatchSize(new DeepSeekVL<double>(architecture, CreateOptions<DeepSeekVLOptions>())));
        Assert.Equal(8, InvokeComputePatchSize(new DeepSeekVL2<double>(architecture, CreateOptions<DeepSeekVL2Options>())));
        Assert.Equal(8, InvokeComputePatchSize(new Gemma3<double>(architecture, CreateOptions<Gemma3Options>())));
        Assert.Equal(8, InvokeComputePatchSize(new InternVL<double>(architecture, CreateOptions<InternVLOptions>())));
        Assert.Equal(8, InvokeComputePatchSize(new InternVL2<double>(architecture, CreateOptions<InternVL2Options>())));
        Assert.Equal(8, InvokeComputePatchSize(new InternVL25<double>(architecture, CreateOptions<InternVL25Options>())));
        Assert.Equal(8, InvokeComputePatchSize(new InternVL3<double>(architecture, CreateOptions<InternVL3Options>())));
        Assert.Equal(8, InvokeComputePatchSize(new Llama32Vision<double>(architecture, CreateOptions<Llama32VisionOptions>())));
        Assert.Equal(8, InvokeComputePatchSize(new Phi3Vision<double>(architecture, CreateOptions<Phi3VisionOptions>())));
        Assert.Equal(8, InvokeComputePatchSize(new Phi4Multimodal<double>(architecture, CreateOptions<Phi4MultimodalOptions>())));
    }

    [Fact]
    public void InstructionTunedModels_WithInvalidVisualSizingOptions_RejectBeforeLayerInitialization()
    {
        var architecture = CreateArchitectureWithCustomLayers();

        AssertInvalidSizingRejected(() => new DeepSeekVL<double>(architecture, CreateOptions<DeepSeekVLOptions>(imageSize: 0)), "ImageSize");
        AssertInvalidSizingRejected(() => new DeepSeekVL2<double>(architecture, CreateOptions<DeepSeekVL2Options>(imageSize: 0)), "ImageSize");
        AssertInvalidSizingRejected(() => new Gemma3<double>(architecture, CreateOptions<Gemma3Options>(imageSize: 0)), "ImageSize");
        AssertInvalidSizingRejected(() => new InternVL<double>(architecture, CreateOptions<InternVLOptions>(imageSize: 0)), "ImageSize");
        AssertInvalidSizingRejected(() => new InternVL2<double>(architecture, CreateOptions<InternVL2Options>(imageSize: 0)), "ImageSize");
        AssertInvalidSizingRejected(() => new InternVL25<double>(architecture, CreateOptions<InternVL25Options>(maxVisualTokens: 0)), "MaxVisualTokens");
        AssertInvalidSizingRejected(() => new InternVL3<double>(architecture, CreateOptions<InternVL3Options>(maxVisualTokens: 0)), "MaxVisualTokens");
        AssertInvalidSizingRejected(() => new Llama32Vision<double>(architecture, CreateOptions<Llama32VisionOptions>(maxVisualTokens: 0)), "MaxVisualTokens");
        AssertInvalidSizingRejected(() => new Phi3Vision<double>(architecture, CreateOptions<Phi3VisionOptions>(maxVisualTokens: 0)), "MaxVisualTokens");
        AssertInvalidSizingRejected(() => new Phi4Multimodal<double>(architecture, CreateOptions<Phi4MultimodalOptions>(maxVisualTokens: 0)), "MaxVisualTokens");
    }

    private static TOptions CreateOptions<TOptions>(int imageSize = 31, int maxVisualTokens = 16)
        where TOptions : InstructionTunedVLMOptions, new()
    {
        return new TOptions
        {
            ImageSize = imageSize,
            MaxVisualTokens = maxVisualTokens,
            VisionDim = 8,
            DecoderDim = 8,
            ProjectionDim = 8,
            NumVisionLayers = 1,
            NumDecoderLayers = 1,
            NumHeads = 1,
            VocabSize = 128,
            MaxSequenceLength = 16
        };
    }

    private static NeuralNetworkArchitecture<double> CreateArchitectureWithCustomLayers()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(outputSize: 8),
            new DenseLayer<double>(outputSize: 8)
        };

        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 31,
            inputWidth: 31,
            inputDepth: 3,
            outputSize: 8,
            layers: layers);
    }

    private static int InvokeComputePatchSize(object model)
    {
        var method = model.GetType().GetMethod("ComputePatchSize", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return (int)method!.Invoke(model, Array.Empty<object>())!;
    }

    private static void AssertInvalidSizingRejected(Action createModel, string expectedMessage)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(createModel);
        Assert.Contains(expectedMessage, ex.Message, StringComparison.Ordinal);
    }
}
