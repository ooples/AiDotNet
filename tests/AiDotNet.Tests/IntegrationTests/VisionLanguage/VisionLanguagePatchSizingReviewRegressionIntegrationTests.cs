#nullable disable
using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.InstructionTuned;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.VisionLanguage;

public class VisionLanguagePatchSizingReviewRegressionIntegrationTests
{
    [Fact(Timeout = 120000)]
    public async Task InstructionTunedModels_WithNonDivisibleTokenBudgets_RoundPatchSizeUp()
    {
        await Task.Yield();
        var architecture = CreateArchitectureWithCustomLayers();

        AssertPatchSizeIsEight(() => new DeepSeekVL<double>(architecture, CreateOptions<DeepSeekVLOptions>()));
        AssertPatchSizeIsEight(() => new DeepSeekVL2<double>(architecture, CreateOptions<DeepSeekVL2Options>()));
        AssertPatchSizeIsEight(() => new Gemma3<double>(architecture, CreateOptions<Gemma3Options>()));
        AssertPatchSizeIsEight(() => new InternVL<double>(architecture, CreateOptions<InternVLOptions>()));
        AssertPatchSizeIsEight(() => new InternVL2<double>(architecture, CreateOptions<InternVL2Options>()));
        AssertPatchSizeIsEight(() => new InternVL25<double>(architecture, CreateOptions<InternVL25Options>()));
        AssertPatchSizeIsEight(() => new InternVL3<double>(architecture, CreateOptions<InternVL3Options>()));
        AssertPatchSizeIsEight(() => new Llama32Vision<double>(architecture, CreateOptions<Llama32VisionOptions>()));
        AssertPatchSizeIsEight(() => new Phi3Vision<double>(architecture, CreateOptions<Phi3VisionOptions>()));
        AssertPatchSizeIsEight(() => new Phi4Multimodal<double>(architecture, CreateOptions<Phi4MultimodalOptions>()));
    }

    private static void AssertPatchSizeIsEight<TModel>(Func<TModel> factory) where TModel : IDisposable
    {
        using var model = factory();
        Assert.Equal(8, InvokeComputePatchSize(model));
    }

    [Fact(Timeout = 120000)]
    public async Task InstructionTunedModels_WithInvalidVisualSizingOptions_RejectBeforeLayerInitialization()
    {
        await Task.Yield();
        var architecture = CreateArchitectureWithCustomLayers();

        AssertInvalidSizingRejected(() => new DeepSeekVL<double>(architecture, CreateOptions<DeepSeekVLOptions>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(() => new DeepSeekVL2<double>(architecture, CreateOptions<DeepSeekVL2Options>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(() => new Gemma3<double>(architecture, CreateOptions<Gemma3Options>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(() => new InternVL<double>(architecture, CreateOptions<InternVLOptions>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(() => new InternVL2<double>(architecture, CreateOptions<InternVL2Options>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(() => new InternVL25<double>(architecture, CreateOptions<InternVL25Options>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(() => new InternVL3<double>(architecture, CreateOptions<InternVL3Options>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(() => new Llama32Vision<double>(architecture, CreateOptions<Llama32VisionOptions>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(() => new Phi3Vision<double>(architecture, CreateOptions<Phi3VisionOptions>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(() => new Phi4Multimodal<double>(architecture, CreateOptions<Phi4MultimodalOptions>(maxVisualTokens: 0)), "maxVisualTokens");
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

    private static void AssertInvalidSizingRejected(Action createModel, string expectedParamName)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(createModel);
        Assert.Equal(expectedParamName, ex.ParamName);
    }
}
