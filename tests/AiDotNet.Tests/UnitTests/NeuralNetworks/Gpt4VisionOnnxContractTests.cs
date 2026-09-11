using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Onnx.Protobuf;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;
using TextInputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.TextInputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class Gpt4VisionOnnxContractTests
{
    public Gpt4VisionOnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    public enum GraphConflict { VisionWidth, TextWidth, Context, Image, Tokens, OutputRank, RequiredInput }
    public enum NativeSetting { HiddenDim, VisionLayers, NumLmLayers, NumHeads, PatchSize, VocabSize, Channels }

    [Theory]
    [InlineData(GraphConflict.VisionWidth)]
    [InlineData(GraphConflict.TextWidth)]
    [InlineData(GraphConflict.Context)]
    [InlineData(GraphConflict.Image)]
    [InlineData(GraphConflict.Tokens)]
    [InlineData(GraphConflict.OutputRank)]
    [InlineData(GraphConflict.RequiredInput)]
    public void GraphConflictsAreRejectedBeforeReturningAModel(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new Gpt4VisionNeuralNetwork<float>(Architecture(), tokenizer: tokenizer,
                visionEncoderPath: fixture.WriteEncoder(EncoderKind.Image,
                    embedding: conflict == GraphConflict.VisionWidth ? 6 : 4,
                    image: conflict == GraphConflict.Image ? 28 : 14,
                    outputKind: conflict == GraphConflict.OutputRank ? OutputKind.FixedEmbedding : OutputKind.FirstTokenEmbedding),
                languageModelPath: fixture.WriteEncoder(EncoderKind.Text,
                    embedding: conflict == GraphConflict.TextWidth ? 6 : 4,
                    context: conflict == GraphConflict.Context ? 9 : 8,
                    tokenType: conflict == GraphConflict.Tokens ? TensorProto.Types.DataType.Float : TensorProto.Types.DataType.Int64,
                    outputKind: OutputKind.FirstTokenEmbedding,
                    extraRequiredInput: conflict == GraphConflict.RequiredInput,
                    textInputKind: TextInputKind.TokensOnly), options: Options());
        });
        Assert.Contains("ONNX", error.Message);
    }

    [Theory]
    [InlineData(NativeSetting.HiddenDim)]
    [InlineData(NativeSetting.VisionLayers)]
    [InlineData(NativeSetting.NumLmLayers)]
    [InlineData(NativeSetting.NumHeads)]
    [InlineData(NativeSetting.PatchSize)]
    [InlineData(NativeSetting.VocabSize)]
    [InlineData(NativeSetting.Channels)]
    public void NativeTopologyOverridesAreNotSilentlyIgnored(NativeSetting setting)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (setting)
        {
            case NativeSetting.HiddenDim: options.HiddenDim = 16; break;
            case NativeSetting.VisionLayers: options.VisionLayers = 2; break;
            case NativeSetting.NumLmLayers: options.NumLmLayers = 2; break;
            case NativeSetting.NumHeads: options.NumHeads = 2; break;
            case NativeSetting.PatchSize: options.PatchSize = 7; break;
            case NativeSetting.VocabSize: options.VocabSize = 512; break;
            case NativeSetting.Channels: options.Channels = 4; break;
            default: throw new ArgumentOutOfRangeException(nameof(setting));
        }
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = Create(fixture, options, dynamicInputs: true);
        });
        Assert.Contains(setting.ToString(), error.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RealGraphEncodersPreserveSeparateVisionAndTextWidths(bool dynamicInputs)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.VisionDim = 6;
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        if (!dynamicInputs) options.MaxSequenceLength = tokenizer.Encode("a").TokenIds.Count;
        using var model = Create(fixture, options, dynamicInputs);
        using var image = new Tensor<float>(new[] { 3, 14, 14 });
        image.AsWritableSpan().Fill(2f);
        var encoded = model.GetImageEmbedding(image);
        Assert.Equal(6, encoded.Length);
        for (int index = 0; index < 6; index++) Assert.Equal(image.Length * 2 + index + 4, encoded[index]);
        double tokenSum = tokenizer.Encode("a").TokenIds.Sum(value => (double)value);
        var text = model.GetTextEmbedding("a");
        Assert.Equal(4, text.Length);
        for (int index = 0; index < 4; index++) Assert.Equal(tokenSum + index + 3, text[index], 5);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Equal(6, configuration.Graphs[OnnxModelRole.ImageEncoder].Outputs.Values.First().Dimensions[2]);
        Assert.Same(configuration, model.GetModelMetadata().AdditionalInfo[nameof(model.OnnxConfiguration)]);
        Assert.False(model.GetModelMetadata().AdditionalInfo.ContainsKey("num_vision_layers"));
        Assert.False(model.GetModelMetadata().AdditionalInfo.ContainsKey("vocabulary_size"));
        options.VisionDim = 123;
        options.ImageSize = 99;
        Assert.Equal(14, configuration.ImageSize);
        Assert.Equal(6, model.GetImageEmbedding(image).Length);
    }

    [Fact]
    public void OnnxImageInputCannotBeSilentlyCroppedOrPadded()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: true);
        using var wrongImage = new Tensor<float>(new[] { 3, 13, 14 });
        Assert.Throws<ArgumentException>(() => model.GetImageEmbedding(wrongImage));
    }

    [Fact]
    public void OnnxTokenInputCannotExceedItsConfiguredContext()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: true);
        Assert.Throws<ArgumentException>(() => model.GetTextEmbedding(string.Join(" ", Enumerable.Repeat("a", 20))));
    }

    [Fact]
    public void OnnxImageSizeDoesNotRequireANativePatch()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 7;
        using var model = Create(fixture, options, dynamicInputs: false);
        using var image = new Tensor<float>(new[] { 3, 7, 7 });
        image.AsWritableSpan().Fill(2f);
        var embedding = model.GetImageEmbedding(image);
        for (int index = 0; index < 4; index++) Assert.Equal(image.Length * 2 + index + 3, embedding[index]);
    }

    [Fact]
    public void StaticTextContextRejectsShortInputsWithoutInventedPadding()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: false);
        var error = Assert.Throws<ArgumentException>(() => model.GetTextEmbedding("a"));
        Assert.Contains("input_ids", error.Message);
    }

    [Theory]
    [InlineData("a")]
    [InlineData("a a a a a a")]
    public void DynamicTextOutputWidthIsCheckedAfterExecution(string text)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        Assert.NotEqual(4, tokenizer.Encode(text).TokenIds.Count);
        using var model = new Gpt4VisionNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: 14, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true,
                outputKind: OutputKind.ContextTokenFeatures, textInputKind: TextInputKind.TokensOnly),
            tokenizer, Options());
        var error = Assert.Throws<InvalidOperationException>(() => model.GetTextEmbedding(text));
        Assert.Contains("embedding width", error.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TokenFeaturesRejectAdditionalBatchOrEmptySequence(bool empty)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new Gpt4VisionNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image, image: 14,
                    outputKind: empty ? OutputKind.EmptyTokenFeatures : OutputKind.BatchedTokenFeatures),
                fixture.WriteEncoder(EncoderKind.Text, outputKind: OutputKind.FirstTokenEmbedding,
                    textInputKind: TextInputKind.TokensOnly), tokenizer, Options());
        });
        Assert.Contains(empty ? "no token" : "batch dimension", error.Message);
    }

    private static Gpt4VisionNeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture,
        Gpt4VisionOptions options, bool dynamicInputs) => new(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: options.ImageSize, embedding: options.VisionDim,
                dynamicInputs: dynamicInputs, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteEncoder(EncoderKind.Text, context: options.MaxSequenceLength, dynamicInputs: dynamicInputs,
                outputKind: OutputKind.FirstTokenEmbedding, textInputKind: TextInputKind.TokensOnly),
            ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }), options);

    private static Gpt4VisionOptions Options() => new()
    {
        EmbeddingDimension = 4, VisionDim = 4, MaxSequenceLength = 8, ImageSize = 14
    };

    private static NeuralNetworkArchitecture<float> Architecture() => new(
        InputType.ThreeDimensional, NeuralNetworkTaskType.ImageClassification,
        inputDepth: 3, inputHeight: 14, inputWidth: 14, outputSize: 4);
}
