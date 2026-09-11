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
        AssertPatchSizeIsEight(architecture => new DeepSeekVL<double>(architecture, CreateOptions<DeepSeekVLOptions>()));
        AssertPatchSizeIsEight(architecture => new DeepSeekVL2<double>(architecture, CreateOptions<DeepSeekVL2Options>()));
        AssertPatchSizeIsEight(architecture => new Gemma3<double>(architecture, CreateOptions<Gemma3Options>()));
        AssertPatchSizeIsEight(architecture => new InternVL<double>(architecture, CreateOptions<InternVLOptions>()));
        AssertPatchSizeIsEight(architecture => new InternVL2<double>(architecture, CreateOptions<InternVL2Options>()));
        AssertPatchSizeIsEight(architecture => new InternVL25<double>(architecture, CreateOptions<InternVL25Options>()));
        AssertPatchSizeIsEight(architecture => new InternVL3<double>(architecture, CreateOptions<InternVL3Options>()));
        AssertPatchSizeIsEight(architecture => new Llama32Vision<double>(architecture, CreateOptions<Llama32VisionOptions>()));
        AssertPatchSizeIsEight(architecture => new Phi3Vision<double>(architecture, CreateOptions<Phi3VisionOptions>()));
        AssertPatchSizeIsEight(architecture => new Phi4Multimodal<double>(architecture, CreateOptions<Phi4MultimodalOptions>()));
    }

    private static void AssertPatchSizeIsEight<TModel>(Func<NeuralNetworkArchitecture<double>, TModel> factory)
        where TModel : IDisposable
    {
        // An architecture carrying explicit layers is a single-owner blueprint. Build a fresh graph for
        // every model under test; reusing one here would either alias mutable layers or reuse layers that
        // the preceding model disposed, neither of which is part of the patch-size behavior under test.
        using var model = factory(CreateArchitectureWithCustomLayers());
        Assert.Equal(8, InvokeComputePatchSize(model));
    }

    [Fact(Timeout = 120000)]
    public async Task InstructionTunedModels_WithInvalidVisualSizingOptions_RejectBeforeLayerInitialization()
    {
        await Task.Yield();
        AssertInvalidSizingRejected(architecture => new DeepSeekVL<double>(architecture, CreateOptions<DeepSeekVLOptions>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(architecture => new DeepSeekVL2<double>(architecture, CreateOptions<DeepSeekVL2Options>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(architecture => new Gemma3<double>(architecture, CreateOptions<Gemma3Options>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(architecture => new InternVL<double>(architecture, CreateOptions<InternVLOptions>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(architecture => new InternVL2<double>(architecture, CreateOptions<InternVL2Options>(imageSize: 0)), "imageSize");
        AssertInvalidSizingRejected(architecture => new InternVL25<double>(architecture, CreateOptions<InternVL25Options>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(architecture => new InternVL3<double>(architecture, CreateOptions<InternVL3Options>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(architecture => new Llama32Vision<double>(architecture, CreateOptions<Llama32VisionOptions>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(architecture => new Phi3Vision<double>(architecture, CreateOptions<Phi3VisionOptions>(maxVisualTokens: 0)), "maxVisualTokens");
        AssertInvalidSizingRejected(architecture => new Phi4Multimodal<double>(architecture, CreateOptions<Phi4MultimodalOptions>(maxVisualTokens: 0)), "maxVisualTokens");
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
        return (int)method.Invoke(model, Array.Empty<object>());
    }

    private static void AssertInvalidSizingRejected(
        Action<NeuralNetworkArchitecture<double>> createModel,
        string expectedParamName)
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(
            () => createModel(CreateArchitectureWithCustomLayers()));
        Assert.Equal(expectedParamName, ex.ParamName);
    }
}
