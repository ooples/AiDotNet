using System;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>Tests the loaded vision stage and constructor contract, not a missing ONNX perceiver or generation path.</summary>
public sealed class FlamingoOnnxContractTests
{
    public enum NativeSetting { LmHiddenDim, VisionLayers, NumLmLayers, NumHeads, PatchSize, VocabSize, Channels, NumPerceiverLayers, NumPerceiverTokens, LearningRate }
    public enum GraphConflict { VisionWidth, ImageSize, Batch, RequiredInput }

    public FlamingoOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(NativeSetting.LmHiddenDim)]
    [InlineData(NativeSetting.VisionLayers)]
    [InlineData(NativeSetting.NumLmLayers)]
    [InlineData(NativeSetting.NumHeads)]
    [InlineData(NativeSetting.PatchSize)]
    [InlineData(NativeSetting.VocabSize)]
    [InlineData(NativeSetting.Channels)]
    [InlineData(NativeSetting.NumPerceiverLayers)]
    [InlineData(NativeSetting.NumPerceiverTokens)]
    [InlineData(NativeSetting.LearningRate)]
    public void NativeOnlyOverridesAreRejectedRatherThanIgnored(NativeSetting setting)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (setting)
        {
            case NativeSetting.LmHiddenDim: options.LmHiddenDim = 16; break;
            case NativeSetting.VisionLayers: options.VisionLayers = 2; break;
            case NativeSetting.NumLmLayers: options.NumLmLayers = 4; break;
            case NativeSetting.NumHeads: options.NumHeads = 2; break;
            case NativeSetting.PatchSize: options.PatchSize = 8; break;
            case NativeSetting.VocabSize: options.VocabSize = 512; break;
            case NativeSetting.Channels: options.Channels = 4; break;
            case NativeSetting.NumPerceiverLayers: options.NumPerceiverLayers = 2; break;
            case NativeSetting.NumPerceiverTokens: options.NumPerceiverTokens = 2; break;
            case NativeSetting.LearningRate: options.LearningRate = 0.02; break;
            default: throw new ArgumentOutOfRangeException(nameof(setting));
        }
        var error = Assert.Throws<ArgumentException>(() => { using var model = Create(fixture, options); });
        Assert.Contains(setting.ToString(), error.Message);
    }

    [Theory]
    [InlineData(GraphConflict.VisionWidth)]
    [InlineData(GraphConflict.ImageSize)]
    [InlineData(GraphConflict.Batch)]
    [InlineData(GraphConflict.RequiredInput)]
    public void TheLoadedVisionStageMustMatchItsConfiguredFeatureBoundary(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new FlamingoNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image, embedding: conflict == GraphConflict.VisionWidth ? 6 : 4,
                    image: conflict == GraphConflict.ImageSize ? 32 : 16,
                    outputKind: conflict == GraphConflict.Batch ? OutputKind.BatchedTokenFeatures : OutputKind.FirstTokenEmbedding,
                    extraRequiredInput: conflict == GraphConflict.RequiredInput),
                fixture.WriteEmbeddedLanguageModel(), ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        });
        Assert.Contains("ONNX", error.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ActualVisionStageReadsGraphTokensWithoutNativePadding(bool dynamicInput)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInput);
        using var image = new Tensor<float>(new[] { 3, 16, 16 });
        image.Fill(2f);
        // This isolates the real existing ONNX vision stage. It deliberately does not
        // claim that the two-file wrapper supplies an ONNX perceiver or text generation.
        MethodInfo method = typeof(FlamingoNeuralNetwork<float>).GetMethod("ExtractVisionFeaturesOnnx", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("The actual ONNX vision boundary is missing.");
        using var features = Assert.IsType<Tensor<float>>(method.Invoke(model, new object[] { image }));
        Assert.Equal(new[] { 2, 4 }, features.Shape);
        for (int index = 0; index < features.Length; index++) Assert.Equal(image.Length * 2 + index + 1, features[index]);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Equal(2, configuration.Graphs.Count);
        Assert.Same(configuration, model.GetModelMetadata().AdditionalInfo[nameof(model.OnnxConfiguration)]);
        Assert.False(model.GetModelMetadata().AdditionalInfo.ContainsKey("NumVisionLayers"));
    }

    [Fact]
    public void GraphImageGeometryDoesNotRequireAnUnobservablePatchSize()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 7;
        using var model = Create(fixture, options);
        Assert.Equal(7, model.ImageSize);
        Assert.NotNull(model.OnnxConfiguration);
    }

    private static FlamingoNeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture, FlamingoOptions options, bool dynamicInput = false)
        => new(Architecture(), fixture.WriteEncoder(EncoderKind.Image, embedding: options.VisionDim, image: options.ImageSize,
                dynamicInputs: dynamicInput, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteEmbeddedLanguageModel(), ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), options);

    private static FlamingoOptions Options() => new() { ImageSize = 16, EmbeddingDimension = 4, VisionDim = 4, MaxSequenceLength = 8 };
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
}
