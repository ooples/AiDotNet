using System;
using System.Linq;
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

public sealed class LLaVAOnnxContractTests
{
    public enum NativeSetting { VisionDim, VisionLayers, NumLmLayers, NumHeads, PatchSize, VocabSize, Channels }
    public enum GraphConflict { VisionWidth, ImageSize, VisionBatch, LanguageWidth, RequiredLanguageInput }

    public LLaVAOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(NativeSetting.VisionDim)]
    [InlineData(NativeSetting.VisionLayers)]
    [InlineData(NativeSetting.NumLmLayers)]
    [InlineData(NativeSetting.NumHeads)]
    [InlineData(NativeSetting.PatchSize)]
    [InlineData(NativeSetting.VocabSize)]
    [InlineData(NativeSetting.Channels)]
    public void NativeOnlyOverridesAreRejectedRatherThanIgnored(NativeSetting setting)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (setting)
        {
            case NativeSetting.VisionDim: options.VisionDim = 16; break;
            case NativeSetting.VisionLayers: options.VisionLayers = 2; break;
            case NativeSetting.NumLmLayers: options.NumLmLayers = 2; break;
            case NativeSetting.NumHeads: options.NumHeads = 2; break;
            case NativeSetting.PatchSize: options.PatchSize = 8; break;
            case NativeSetting.VocabSize: options.VocabSize = 512; break;
            case NativeSetting.Channels: options.Channels = 4; break;
            default: throw new ArgumentOutOfRangeException(nameof(setting));
        }
        var error = Assert.Throws<ArgumentException>(() => { using var model = Create(fixture, options); });
        Assert.Contains(setting.ToString(), error.Message);
    }

    [Theory]
    [InlineData(GraphConflict.VisionWidth)]
    [InlineData(GraphConflict.ImageSize)]
    [InlineData(GraphConflict.VisionBatch)]
    [InlineData(GraphConflict.LanguageWidth)]
    [InlineData(GraphConflict.RequiredLanguageInput)]
    public void GraphsMustMatchTheWrappersActualEmbeddingBoundary(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new LLaVANeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image, embedding: conflict == GraphConflict.VisionWidth ? 6 : 4,
                    image: conflict == GraphConflict.ImageSize ? 32 : 16,
                    outputKind: conflict == GraphConflict.VisionBatch ? OutputKind.BatchedTokenFeatures : OutputKind.FirstTokenEmbedding),
                fixture.WriteEmbeddedLanguageModel(width: conflict == GraphConflict.LanguageWidth ? 6 : 4,
                    extraRequiredInput: conflict == GraphConflict.RequiredLanguageInput),
                ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        });
        Assert.Contains("ONNX", error.Message);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActualGraphFeaturesAndMeanPooledEmbeddingUseLoadedTokenGeometry(bool dynamicInput, bool batchedInput)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInput);
        using var image = new Tensor<float>(batchedInput ? new[] { 1, 3, 16, 16 } : new[] { 3, 16, 16 });
        image.Fill(2f);
        using var features = model.ExtractVisualFeatures(image);
        Assert.Equal(new[] { 2, 4 }, features.Shape);
        double sum = image.Length * 2;
        for (int token = 0; token < 2; token++)
            for (int column = 0; column < 4; column++)
                Assert.Equal(sum + token * 4 + column + 1, features[token, column], 4);
        var embedding = model.GetImageEmbedding(image);
        double norm = Math.Sqrt(Enumerable.Range(3, 4).Sum(offset => (sum + offset) * (sum + offset)));
        Assert.Equal(4, embedding.Length);
        for (int column = 0; column < 4; column++)
            Assert.Equal((sum + column + 3) / norm, embedding[column], 4);
        Assert.Equal(2, model.NumVisualTokens);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Equal(2, configuration.Graphs.Count);
        Assert.Same(configuration, model.GetModelMetadata().AdditionalInfo[nameof(model.OnnxConfiguration)]);
        Assert.False(model.GetModelMetadata().AdditionalInfo.ContainsKey("VisionHiddenDim"));
    }

    [Fact]
    public void OnnxImageSizeDoesNotRequireAnUnobservableNativePatch()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 7;
        using var model = Create(fixture, options);
        using var image = new Tensor<float>(new[] { 3, 7, 7 });
        Assert.Equal(4, model.GetImageEmbedding(image).Length);
        Assert.Equal(2, model.NumVisualTokens);
    }

    [Fact]
    public void BatchTwoCannotSilentlyLoseTheSecondImage()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInput: true);
        using var image = new Tensor<float>(new[] { 2, 3, 16, 16 });
        var error = Assert.Throws<ArgumentException>(() => model.ExtractVisualFeatures(image));
        Assert.Equal("image", error.ParamName);
    }

    [Fact]
    public void SymbolicTokenCountIsNotInventedFromTheNativePatchSize()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = new LLaVANeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, dynamicInputs: true, outputKind: OutputKind.SpatialTokenFeatures),
            fixture.WriteEmbeddedLanguageModel(), ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        Assert.Throws<InvalidOperationException>(() => model.NumVisualTokens);
        using var image = new Tensor<float>(new[] { 3, 16, 16 });
        image.Fill(2f);
        using var features = model.ExtractVisualFeatures(image);
        Assert.Equal(new[] { 16, 4 }, features.Shape);
        for (int token = 0; token < 16; token++)
            for (int column = 0; column < 4; column++)
                Assert.Equal(96 + column + 1, features[token, column]);
    }

    [Fact]
    public void SingleVectorGraphPreservesOneTokenWithoutFabricatedPatches()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = new LLaVANeuralNetwork<float>(Architecture(), fixture.WriteEncoder(EncoderKind.Image),
            fixture.WriteEmbeddedLanguageModel(), ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        using var image = new Tensor<float>(new[] { 3, 16, 16 });
        using var features = model.ExtractVisualFeatures(image);
        Assert.Equal(new[] { 1, 4 }, features.Shape);
        Assert.Equal(1, model.NumVisualTokens);
        Assert.Equal(4, model.GetImageEmbedding(image).Length);
        Assert.NotNull(model.OnnxConfiguration);
    }

    private static LLaVANeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture, LLaVAOptions options, bool dynamicInput = false)
        => new(Architecture(), fixture.WriteEncoder(EncoderKind.Image, image: options.ImageSize,
                dynamicInputs: dynamicInput, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteEmbeddedLanguageModel(), ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), options);

    private static LLaVAOptions Options() => new() { ImageSize = 16, EmbeddingDimension = 4, MaxSequenceLength = 8 };
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
}
