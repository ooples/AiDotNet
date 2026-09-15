using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Onnx.Protobuf;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class BlipOnnxContractTests
{
    public enum NativeSetting { HiddenDim, NumEncoderLayers, NumDecoderLayers, NumHeads, MlpDim, PatchSize, VocabSize, Channels }
    public enum GraphConflict
    {
        ImageWidth, TextWidth, ImageSize, Context, TokenType, RequiredInput, OutputBatch,
        VisionWithoutHiddenStates, DecoderWithoutEncoderStates, DecoderStateWidth, DecoderVocabulary
    }

    private const int Vocabulary = 512;

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
    [InlineData(GraphConflict.VisionWithoutHiddenStates)]
    [InlineData(GraphConflict.DecoderWithoutEncoderStates)]
    [InlineData(GraphConflict.DecoderStateWidth)]
    [InlineData(GraphConflict.DecoderVocabulary)]
    public void FixedGraphConflictsAreRejectedBeforeConstructionCompletes(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = Tokenizer();
        var ids = SpecialIds(tokenizer);
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new BlipNeuralNetwork<float>(Architecture(),
                fixture.WriteEncoder(EncoderKind.Image, embedding: conflict == GraphConflict.ImageWidth ? 6 : 4,
                    image: conflict == GraphConflict.ImageSize ? 32 : 16,
                    outputKind: conflict == GraphConflict.OutputBatch ? OutputKind.BatchedEmbedding : OutputKind.FixedEmbedding,
                    hiddenStatesWidth: conflict == GraphConflict.VisionWithoutHiddenStates ? 0 : 4),
                fixture.WriteEncoder(EncoderKind.Text, embedding: conflict == GraphConflict.TextWidth ? 6 : 4,
                    context: conflict == GraphConflict.Context ? 9 : 8,
                    tokenType: conflict == GraphConflict.TokenType ? TensorProto.Types.DataType.Float : TensorProto.Types.DataType.Int64,
                    extraRequiredInput: conflict == GraphConflict.RequiredInput),
                fixture.WriteTextDecoder(Vocabulary, ids.Word, ids.Eos,
                    hiddenWidth: conflict == GraphConflict.DecoderStateWidth ? 6 : 4,
                    omitEncoderStates: conflict == GraphConflict.DecoderWithoutEncoderStates,
                    declaredVocabulary: conflict == GraphConflict.DecoderVocabulary ? Vocabulary - 1 : null),
                tokenizer, Options());
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
        var tokenizer = Tokenizer();
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
        Assert.Equal(nameof(BlipNeuralNetwork<float>), metadata.AdditionalInfo["ModelType"]);
        Assert.Equal(new[] { 3, 16, 16 }, Assert.IsType<int[]>(metadata.AdditionalInfo["InputShape"]));
        Assert.Equal(new[] { 4 }, Assert.IsType<int[]>(metadata.AdditionalInfo["OutputShape"]));
        Assert.False(metadata.AdditionalInfo.ContainsKey("HiddenDimension"));
        Assert.False(metadata.AdditionalInfo.ContainsKey("VocabularySize"));
    }

    [Fact]
    public void CaptionIsGreedilyDecodedFromTheImageStatesAndStopsAtEos()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = Tokenizer();
        var ids = SpecialIds(tokenizer);
        using var model = Create(fixture, Options(), tokenizer: tokenizer);

        // Bright pixels make the decoder's state sum positive, so every step emits the word token.
        using var bright = Image(2f);
        Assert.Equal(tokenizer.Decode(new List<int> { ids.Bos, ids.Word, ids.Word, ids.Word }),
            model.GenerateCaption(bright, maxLength: 3, numBeams: 1));

        // Dark pixels make it negative, so the first step emits EOS and the caption holds only BOS.
        using var dark = Image(-2f);
        Assert.Equal(tokenizer.Decode(new List<int> { ids.Bos }), model.GenerateCaption(dark, maxLength: 3, numBeams: 1));
    }

    [Fact]
    public void CaptionLengthIsBoundedByTheConfiguredContext()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = Tokenizer();
        var ids = SpecialIds(tokenizer);
        using var model = Create(fixture, Options(), tokenizer: tokenizer);
        using var bright = Image(2f);
        var expected = new List<int> { ids.Bos };
        expected.AddRange(Enumerable.Repeat(ids.Word, Options().MaxSequenceLength - 1));
        Assert.Equal(tokenizer.Decode(expected), model.GenerateCaption(bright, maxLength: 50, numBeams: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.GenerateCaption(bright, maxLength: 0, numBeams: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.GenerateCaption(bright, maxLength: 3, numBeams: 0));
    }

    [Fact]
    public void AnswerContinuesTheUnpaddedQuestionAndReturnsOnlyTheContinuation()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var tokenizer = Tokenizer();
        var ids = SpecialIds(tokenizer);
        using var model = Create(fixture, Options(), tokenizer: tokenizer);
        using var bright = Image(2f);
        var question = tokenizer.Encode("a", new EncodingOptions
        {
            MaxLength = Options().MaxSequenceLength, Padding = false, Truncation = true, AddSpecialTokens = true
        });
        int answerLength = Math.Min(2, Options().MaxSequenceLength - question.TokenIds.Count);
        Assert.True(answerLength > 0, "The fixture question must leave room for an answer.");
        Assert.Equal(tokenizer.Decode(Enumerable.Repeat(ids.Word, answerLength).ToList()),
            model.AnswerQuestion(bright, "a", maxLength: 2));
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
        Assert.Throws<ArgumentException>(() => model.GenerateCaption(image, maxLength: 3, numBeams: 1));
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
        var tokenizer = Tokenizer();
        var ids = SpecialIds(tokenizer);
        using var model = new BlipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, hiddenStatesWidth: 4),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true, outputKind: OutputKind.TokenSequence),
            fixture.WriteTextDecoder(Vocabulary, ids.Word, ids.Eos),
            tokenizer, Options());
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

    private static Tensor<float> Image(float fill)
    {
        var image = new Tensor<float>(new[] { 3, 16, 16 });
        image.AsWritableSpan().Fill(fill);
        return image;
    }

    private static BpeTokenizer Tokenizer() => ClipTokenizerFactory.CreateShapeCompatibleForTesting(Vocabulary, new[] { "a" });

    /// <summary>The BOS and EOS ids the model resolves, and a distinct ordinary token for the decoder fixture.</summary>
    private static (int Bos, int Eos, int Word) SpecialIds(BpeTokenizer tokenizer)
    {
        var special = tokenizer.SpecialTokens;
        int Resolve(string? primary, string? secondary, int fallback)
            => !string.IsNullOrEmpty(primary) ? tokenizer.Vocabulary.GetTokenId(primary)
                : !string.IsNullOrEmpty(secondary) ? tokenizer.Vocabulary.GetTokenId(secondary)
                : fallback;
        int bos = Resolve(special?.BosToken, special?.ClsToken, 101);
        int eos = Resolve(special?.EosToken, special?.SepToken, 102);
        int word = Enumerable.Range(1, Vocabulary - 1).First(id => id != bos && id != eos);
        return (bos, eos, word);
    }

    private static BlipNeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture, BlipOptions options,
        bool dynamicInputs = false, bool tokenFeatures = false, BpeTokenizer? tokenizer = null)
    {
        tokenizer ??= Tokenizer();
        var ids = SpecialIds(tokenizer);
        var output = tokenFeatures ? OutputKind.FirstTokenEmbedding : OutputKind.FixedEmbedding;
        return new BlipNeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: options.ImageSize, dynamicInputs: dynamicInputs, outputKind: output,
                hiddenStatesWidth: 4),
            fixture.WriteEncoder(EncoderKind.Text, context: options.MaxSequenceLength, dynamicInputs: dynamicInputs, outputKind: output),
            fixture.WriteTextDecoder(Vocabulary, ids.Word, ids.Eos),
            tokenizer, options);
    }

    private static BlipOptions Options() => new() { ImageSize = 16, EmbeddingDimension = 4, MaxSequenceLength = 8 };
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 16, inputWidth: 16, outputSize: 4);
}
