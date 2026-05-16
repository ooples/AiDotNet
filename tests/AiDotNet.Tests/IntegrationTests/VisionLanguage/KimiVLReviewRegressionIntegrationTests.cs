#nullable disable
using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.VisionLanguage.Reasoning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.VisionLanguage;

public class KimiVLReviewRegressionIntegrationTests
{
    [Fact(Timeout = 120000)]
    public void Constructor_WithInvalidImageSize_ThrowsBeforeLayerInitialization()
    {
        var options = CreateOptions();
        options.ImageSize = 0;

        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new KimiVL<double>(CreateArchitectureWithCustomLayers(), options));

        // Assert on the parameter-name contract rather than message text — the validator
        // throws with `nameof(imageSize)` which produces lowercase "imageSize", and the
        // message wording isn't part of the public contract.
        Assert.Equal("imageSize", ex.ParamName);
    }

    [Fact(Timeout = 120000)]
    public void Constructor_WithTinyNativeConfiguration_UsesSharedBoundaryHelpers()
    {
        var options = CreateOptions();
        var model = new KimiVL<double>(CreateDefaultLayerArchitecture(), options);

        Assert.NotEmpty(model.Layers);
        int encoderLayerEnd = GetPrivateIntField(model, "_encoderLayerEnd");
        int transformerBlockLayerCount = options.DropoutRate > 0 ? 6 : 5;
        int resamplerBlockLayerCount = options.DropoutRate > 0 ? 8 : 7;
        int expectedBoundary = 2 + options.NumVisionLayers * transformerBlockLayerCount + 4 * resamplerBlockLayerCount + 1;
        Assert.Equal(expectedBoundary, encoderLayerEnd);
        Assert.InRange(encoderLayerEnd, 1, model.Layers.Count - 1);
    }

    [Fact(Timeout = 120000)]
    public void Constructor_WithNonDivisibleTokenBudget_RoundsPatchSizeUp()
    {
        var options = CreateOptions();
        options.ImageSize = 31;
        options.MaxVisualTokens = 16;
        var model = new KimiVL<double>(CreateArchitectureWithCustomLayers(), options);

        int patchSize = InvokePrivateIntMethod(model, "ComputeKimiPatchSize");

        Assert.Equal(8, patchSize);
    }

    private static KimiVLOptions CreateOptions()
    {
        return new KimiVLOptions
        {
            ImageSize = 32,
            MaxVisualTokens = 16,
            VisionDim = 8,
            DecoderDim = 8,
            NumVisionLayers = 1,
            NumDecoderLayers = 1,
            NumHeads = 1,
            VocabSize = 128,
            MaxSequenceLength = 16
        };
    }

    private static NeuralNetworkArchitecture<double> CreateDefaultLayerArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 8);
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
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 8,
            layers: layers);
    }

    private static int GetPrivateIntField(object instance, string fieldName)
    {
        var field = instance.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return (int)field!.GetValue(instance)!;
    }

    private static int InvokePrivateIntMethod(object instance, string methodName)
    {
        var method = instance.GetType().GetMethod(methodName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return (int)method!.Invoke(instance, Array.Empty<object>())!;
    }
}
