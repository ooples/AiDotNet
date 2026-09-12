using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Onnx.Protobuf;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Models;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class BlipOnnxContractTests
{
    public enum NativeSetting { HiddenDim, NumEncoderLayers, NumDecoderLayers, NumHeads, MlpDim, PatchSize, VocabSize, Channels }
    public enum GraphConflict { ImageWidth, TextWidth, ImageSize, Context, TokenType, RequiredInput, OutputBatch }

    public BlipOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(NativeSetting.HiddenDim)]
    [InlineData(NativeSetting.NumEncoderLayers)]
    [InlineData(NativeSetting.NumDecoderLayers)]
    [InlineData(NativeSetting.NumHeads)]
    [InlineData(NativeSetting.MlpDim)]
    [InlineData(NativeSetting.PatchSize)]
    [InlineData(NativeSetting.VocabSize)]
    [InlineData(NativeSetting.Channels)]
    public void NativeOnlyOverridesAreRejectedRatherThanIgnored(NativeSetting setting)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (setting)
        {
            case NativeSetting.HiddenDim: options.HiddenDim = 16; break;
            case NativeSetting.NumEncoderLayers: options.NumEncoderLayers = 2; break;
            case NativeSetting.NumDecoderLayers: options.NumDecoderLayers = 2; break;
            case NativeSetting.NumHeads: options.NumHeads = 2; break;
            case NativeSetting.MlpDim: options.MlpDim = 32; break;
            case NativeSetting.PatchSize: options.PatchSize = 8; break;
            case NativeSetting.VocabSize: options.VocabSize = 512; break;
            case NativeSetting.Channels: options.Channels = 4; break;
            default: throw new ArgumentOutOfRangeException(nameof(setting));
        }
        var error = Assert.Throws<ArgumentException>(() => { using var model = Create(fixture, options); });
        Assert.Contains(setting.ToString(), error.Message);
    }

    [Theory]
    [InlineData(GraphConflict.ImageWidth)]
    [InlineData(GraphConflict.TextWidth)]
    [InlineData(GraphConflict.ImageSize)]
    [InlineData(GraphConflict.Context)]
    [InlineData(GraphConflict.TokenType)]
    [InlineData(GraphConflict.RequiredInput)]
    [InlineData(GraphConflict.OutputBatch)]
    public void FixedGraphConflictsAreRejectedBeforeConstructionCompletes(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new BlipNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image, embedding: conflict == GraphConflict.ImageWidth ? 6 : 4,
                    image: conflict == GraphConflict.ImageSize ? 32 : 16,
                    outputKind: conflict == GraphConflict.OutputBatch ? OutputKind.BatchedEmbedding : OutputKind.FixedEmbedding),
                fixture.WriteEncoder(EncoderKind.Text, embedding: conflict == GraphConflict.TextWidth ? 6 : 4,
                    context: conflict == GraphConflict.Context ? 9 : 8,
                    tokenType: conflict == GraphConflict.TokenType ? TensorProto.Types.DataType.Float : TensorProto.Types.DataType.Int64,
                    extraRequiredInput: conflict == GraphConflict.RequiredInput),
                fixture.WriteEncoder(EncoderKind.Text), tokenizer, Options());
        });
        Assert.Contains("ONNX", error.Message);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActualImageAndTextEncodersKeepTheirNormalizedFirstTokenContract(bool dynamicInputs, bool tokenFeatures)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs, tokenFeatures);
        using var image = new Tensor<float>(new[] { 3, 16, 16 });
        image.AsWritableSpan().Fill(2f);
        AssertNormalized(model.GetImageEmbedding(image), image.Length * 2);
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var encoded = tokenizer.Encode("a", new EncodingOptions
        {
            MaxLength = 8, Padding = true, Truncation = true, AddSpecialTokens = true
        });
        double tokenSum = encoded.TokenIds.Select((value, index) =>
            (double)value * (encoded.AttentionMask is null ? 1 : encoded.AttentionMask[index])).Sum();
        AssertNormalized(model.GetTextEmbedding("a"), tokenSum);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Equal(3, configuration.Graphs.Count);
        Assert.True(configuration.Graphs.ContainsKey(OnnxModelRole.TextDecoder));
        var metadata = model.GetModelMetadata();
        Assert.Same(configuration, metadata.AdditionalInfo[nameof(model.OnnxConfiguration)]);
        Assert.False(metadata.AdditionalInfo.ContainsKey("HiddenDimension"));
        Assert.False(metadata.AdditionalInfo.ContainsKey("VocabularySize"));
    }

    [Fact]
    public void OnnxInputDoesNotNeedANativePatch()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 7;
        using var model = Create(fixture, options);
        using var image = new Tensor<float>(new[] { 3, 7, 7 });
        image.AsWritableSpan().Fill(2f);
        AssertNormalized(model.GetImageEmbedding(image), image.Length * 2);
    }

    [Fact]
    public void ImageBatchCannotBeSilentlyReducedToItsFirstExample()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: true);
        using var image = new Tensor<float>(new[] { 2, 3, 16, 16 });
        Assert.Throws<ArgumentException>(() => model.GetImageEmbedding(image));
    }

    [Theory]
    [InlineData(2, 16, 16)]
    [InlineData(3, 15, 16)]
    [InlineData(3, 16, 15)]
    public void DynamicGraphDoesNotOverrideHostImageContract(int channels, int height, int width)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: true);
        using var image = new Tensor<float>(new[] { channels, height, width });
        Assert.Throws<ArgumentException>(() => model.GetImageEmbedding(image));
    }

    [Fact]
    public void SymbolicTextOutputWidthIsCheckedAfterExecutingTheGraph()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = new BlipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true, outputKind: OutputKind.TokenSequence),
            fixture.WriteEncoder(EncoderKind.Text),
            ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), Options());
        var error = Assert.Throws<InvalidOperationException>(() => model.GetTextEmbedding("a"));
        Assert.Contains("embedding width 8", error.Message);
    }

    [Fact]
    public void ExplicitSingleImageBatchRetainsAllPixels()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options());
        using var image = new Tensor<float>(new[] { 1, 3, 16, 16 });
        image.AsWritableSpan().Fill(3f);
        AssertNormalized(model.GetImageEmbedding(image), image.Length * 3);
    }

    [Fact]
    public void CallerMutationCannotChangeTheLoadedGraphConfiguration()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        using var model = Create(fixture, options);
        options.ImageSize = 32;
        options.EmbeddingDimension = 6;
        options.MaxSequenceLength = 9;
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Equal(16, configuration.ImageSize);
        Assert.Equal(4, configuration.EmbeddingDimension);
        Assert.Equal(8, configuration.MaxSequenceLength);
        using var image = new Tensor<float>(new[] { 3, 16, 16 });
        image.AsWritableSpan().Fill(2f);
        AssertNormalized(model.GetImageEmbedding(image), image.Length * 2);
    }

    private static void AssertNormalized(Vector<float> actual, double sum)
    {
        Assert.Equal(4, actual.Length);
        double norm = Math.Sqrt(Enumerable.Range(1, 4).Sum(offset => (sum + offset) * (sum + offset)));
        for (int index = 0; index < 4; index++) Assert.Equal((sum + index + 1) / norm, actual[index], 5);
    }

    private static BlipNeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture, BlipOptions options,
        bool dynamicInputs = false, bool tokenFeatures = false)
    {
        var output = tokenFeatures ? OutputKind.FirstTokenEmbedding : OutputKind.FixedEmbedding;
        return new BlipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: options.ImageSize, dynamicInputs: dynamicInputs, outputKind: output),
            fixture.WriteEncoder(EncoderKind.Text, context: options.MaxSequenceLength, dynamicInputs: dynamicInputs, outputKind: output),
            fixture.WriteEncoder(EncoderKind.Text),
            ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), options);
    }

    private static BlipOptions Options() => new() { ImageSize = 16, EmbeddingDimension = 4, MaxSequenceLength = 8 };
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
}
