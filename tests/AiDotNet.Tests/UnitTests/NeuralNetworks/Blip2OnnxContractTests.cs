using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Onnx;
using AiDotNet.Tests.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Models;
using Xunit;
using EncoderKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.EncoderKind;
using OutputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.OutputKind;
using QueryInputKind = AiDotNet.Tests.Helpers.OnnxVisionLanguageFixture.QueryInputKind;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class Blip2OnnxContractTests
{
    public enum NativeSetting { QformerHiddenDim, LmHiddenDim, NumQformerLayers, NumHeads, NumLmDecoderLayers, PatchSize, VocabSize, Channels }
    public enum GraphConflict { VisionWidth, QueryInputWidth, QueryOutputWidth, QueryCount, ImageSize, QueryBatch, VisionBatch, RequiredInput, RequiredBoth }

    public Blip2OnnxContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(NativeSetting.QformerHiddenDim)]
    [InlineData(NativeSetting.LmHiddenDim)]
    [InlineData(NativeSetting.NumQformerLayers)]
    [InlineData(NativeSetting.NumHeads)]
    [InlineData(NativeSetting.NumLmDecoderLayers)]
    [InlineData(NativeSetting.PatchSize)]
    [InlineData(NativeSetting.VocabSize)]
    [InlineData(NativeSetting.Channels)]
    public void NativeOnlyOverridesCannotSilentlyConfigureAnOpaqueGraph(NativeSetting setting)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        switch (setting)
        {
            case NativeSetting.QformerHiddenDim: options.QformerHiddenDim = 16; break;
            case NativeSetting.LmHiddenDim: options.LmHiddenDim = 16; break;
            case NativeSetting.NumQformerLayers: options.NumQformerLayers = 2; break;
            case NativeSetting.NumHeads: options.NumHeads = 2; break;
            case NativeSetting.NumLmDecoderLayers: options.NumLmDecoderLayers = 2; break;
            case NativeSetting.PatchSize: options.PatchSize = 7; break;
            case NativeSetting.VocabSize: options.VocabSize = 512; break;
            case NativeSetting.Channels: options.Channels = 4; break;
            default: throw new ArgumentOutOfRangeException(nameof(setting));
        }
        var error = Assert.Throws<ArgumentException>(() => { using var model = Create(fixture, options); });
        Assert.Contains(setting.ToString(), error.Message);
    }

    [Theory]
    [InlineData(GraphConflict.VisionWidth)]
    [InlineData(GraphConflict.QueryInputWidth)]
    [InlineData(GraphConflict.QueryOutputWidth)]
    [InlineData(GraphConflict.QueryCount)]
    [InlineData(GraphConflict.ImageSize)]
    [InlineData(GraphConflict.QueryBatch)]
    [InlineData(GraphConflict.VisionBatch)]
    [InlineData(GraphConflict.RequiredInput)]
    [InlineData(GraphConflict.RequiredBoth)]
    public void LoadedGraphContractsRejectIncompatibleBoundaries(GraphConflict conflict)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        string vision = fixture.WriteEncoder(EncoderKind.Image, embedding: conflict == GraphConflict.VisionWidth ? 6 : 4,
            image: conflict == GraphConflict.ImageSize ? 28 : 14,
            outputKind: conflict == GraphConflict.VisionBatch ? OutputKind.BatchedTokenFeatures : OutputKind.FirstTokenEmbedding);
        string query = fixture.WriteQueryTransformer(
            kind: conflict == GraphConflict.RequiredBoth ? QueryInputKind.RequiredBoth : QueryInputKind.ImageOnly,
            width: conflict == GraphConflict.QueryOutputWidth ? 6 : 4,
            visionWidth: conflict == GraphConflict.QueryInputWidth ? 6 : 4,
            queries: conflict == GraphConflict.QueryCount ? 3 : 2,
            batch: conflict == GraphConflict.QueryBatch ? 2 : 1,
            extraRequiredInput: conflict == GraphConflict.RequiredInput);
        var error = Assert.Throws<ArgumentException>(() =>
        {
            using var model = new Blip2NeuralNetwork<float>(Architecture(), vision, query,
                fixture.WriteEmbeddedLanguageModel(), Tokenizer(), Options());
        });
        Assert.Contains("ONNX", error.Message);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ImageQueriesUseActualConfiguredCountAndBothGraphNumerics(bool dynamicInputs)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), dynamicInputs: dynamicInputs);
        using var image = Image(14, 2);
        using var features = model.ExtractQFormerFeatures(image);
        Assert.Equal(new[] { 2, 4 }, features.Shape);
        double visionSum = 8 * image.Length * 2 + 36;
        for (int index = 0; index < features.Length; index++) Assert.Equal(visionSum + index + 1, features[index]);
        AssertNormalized(model.GetImageEmbedding(image), Enumerable.Range(0, 4).Select(i => visionSum + i + 3).ToArray());
        Assert.Equal(2, model.NumQueryTokens);
        Assert.Equal(2, Assert.IsType<Blip2Options>(model.GetOptions()).NumQueryTokens);
        var info = model.GetModelMetadata().AdditionalInfo;
        Assert.Equal(2, info["NumQueryTokens"]);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.Same(configuration, info[nameof(model.OnnxConfiguration)]);
        Assert.Equal(3, configuration.Graphs.Count);
        Assert.False(info.ContainsKey("NumLmDecoderLayers"));
        Assert.False(info.ContainsKey("QFormerHiddenDim"));
    }

    [Fact]
    public void TextOnlyExportPreservesMeanPoolingAndDoesNotInventVisualQueries()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.NumQueryTokens = new Blip2Options().NumQueryTokens;
        using var model = Create(fixture, options, QueryInputKind.TextOnly);
        AssertTextEmbedding(model);
        Assert.Throws<NotSupportedException>(() => _ = model.NumQueryTokens);
        using var image = Image(14, 2);
        Assert.Throws<NotSupportedException>(() => model.ExtractQFormerFeatures(image));
        Assert.False(model.GetModelMetadata().AdditionalInfo.ContainsKey("NumQueryTokens"));
    }

    [Fact]
    public void DefaultedInputsSupportBothActualOperationsWithoutRequiringAllInputs()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options(), QueryInputKind.DefaultedBoth);
        AssertTextEmbedding(model);
        using var image = Image(14, 2);
        using var features = model.ExtractQFormerFeatures(image);
        Assert.Equal(new[] { 2, 4 }, features.Shape);
        Assert.Equal(8 * image.Length * 2 + 37, features[0]);
        var configuration = Assert.IsType<OnnxMultimodalConfiguration>(model.OnnxConfiguration);
        Assert.True(configuration.Graphs[OnnxModelRole.QueryTransformer].Inputs["encoder_hidden_states"].HasDefaultValue);
    }

    [Fact]
    public void ImageOnlyExportRejectsTextBeforeInvokingAnIncompatibleGraph()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options());
        Assert.Throws<NotSupportedException>(() => model.GetTextEmbedding("a"));
    }

    [Fact]
    public void ATextOnlyGraphCannotSilentlyIgnoreAnExplicitVisualQueryOverride()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var error = Assert.Throws<ArgumentException>(() => { using var model = Create(fixture, Options(), QueryInputKind.TextOnly); });
        Assert.Contains(nameof(Blip2Options.NumQueryTokens), error.Message);
    }

    [Fact]
    public void SymbolicVisualQueryCountIsCheckedAtExecution()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = new Blip2NeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: 14, dynamicInputs: true, outputKind: OutputKind.SpatialTokenFeatures),
            fixture.WriteQueryTransformer(dynamicInputs: true, preserveVisionTokens: true),
            fixture.WriteEmbeddedLanguageModel(), Tokenizer(), Options());
        using var image = Image(14, 2);
        var error = Assert.Throws<InvalidOperationException>(() => model.ExtractQFormerFeatures(image));
        Assert.Contains(nameof(Blip2Options.NumQueryTokens), error.Message);
    }

    [Fact]
    public void ImageBatchIsNotSilentlyDiscarded()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        using var model = Create(fixture, Options());
        using var image = new Tensor<float>(new[] { 2, 3, 14, 14 });
        Assert.Throws<ArgumentException>(() => model.ExtractQFormerFeatures(image));
    }

    [Fact]
    public void OnnxImageSizeIsNotConstrainedByAnUnobservableNativePatch()
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.ImageSize = 7;
        using var model = Create(fixture, options);
        using var image = Image(7, 1);
        using var features = model.ExtractQFormerFeatures(image);
        Assert.Equal(new[] { 2, 4 }, features.Shape);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExistingThirtyTwoQueryExportRetainsItsImageEmbedding(bool batchedImage)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.NumQueryTokens = 32;
        using var model = new Blip2NeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: 14, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteQueryTransformer(queries: 32), fixture.WriteEmbeddedLanguageModel(), Tokenizer(), options);
        using var image = new Tensor<float>(batchedImage ? new[] { 1, 3, 14, 14 } : new[] { 3, 14, 14 });
        for (int i = 0; i < image.Length; i++) image[i] = 2;
        double visionSum = 8 * image.Length * 2 + 36;
        AssertNormalized(model.GetImageEmbedding(image), Enumerable.Range(0, 4).Select(i => visionSum + i + 63).ToArray());
        Assert.Equal(32, model.NumQueryTokens);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExistingDynamicTextExportsRetainVectorAndMeanPooledEmbeddings(bool tokenFeatures)
    {
        using var fixture = new OnnxVisionLanguageFixture();
        var options = Options();
        options.NumQueryTokens = 32;
        using var model = new Blip2NeuralNetwork<float>(Architecture(),
            fixture.WriteEncoder(EncoderKind.Image, image: 14, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteEncoder(EncoderKind.Text, dynamicInputs: true,
                outputKind: tokenFeatures ? OutputKind.FirstTokenEmbedding : OutputKind.FixedEmbedding),
            fixture.WriteEmbeddedLanguageModel(), Tokenizer(), options);
        var encoded = Tokenizer().Encode("a");
        var mask = encoded.AttentionMask ?? Enumerable.Repeat(1, encoded.TokenIds.Count).ToList();
        double sum = encoded.TokenIds.Select((id, i) => (double)id * mask[i]).Sum();
        AssertNormalized(model.GetTextEmbedding("a"), Enumerable.Range(0, 4).Select(i => sum + i + (tokenFeatures ? 3 : 1)).ToArray());
    }

    private static void AssertTextEmbedding(Blip2NeuralNetwork<float> model)
    {
        var encoded = Tokenizer().Encode("a", new EncodingOptions { MaxLength = 8, Padding = true, Truncation = true, AddSpecialTokens = true });
        var mask = encoded.AttentionMask ?? Enumerable.Repeat(1, encoded.TokenIds.Count).ToList();
        double textSum = encoded.TokenIds.Select((id, i) => (double)id * mask[i]).Sum();
        AssertNormalized(model.GetTextEmbedding("a"), Enumerable.Range(0, 4).Select(i => textSum + i + 3).ToArray());
    }

    private static void AssertNormalized(Vector<float> actual, double[] expected)
    {
        double norm = Math.Sqrt(expected.Sum(value => value * value));
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(actual[i] - expected[i] / norm), 0, 1e-6);
    }

    private static Blip2NeuralNetwork<float> Create(OnnxVisionLanguageFixture fixture, Blip2Options options,
        QueryInputKind kind = QueryInputKind.ImageOnly, bool dynamicInputs = false)
        => new(Architecture(), fixture.WriteEncoder(EncoderKind.Image, embedding: options.VisionDim, image: options.ImageSize,
                dynamicInputs: dynamicInputs, outputKind: OutputKind.FirstTokenEmbedding),
            fixture.WriteQueryTransformer(kind, dynamicInputs: dynamicInputs), fixture.WriteEmbeddedLanguageModel(), Tokenizer(), options);

    private static AiDotNet.Tokenization.Interfaces.ITokenizer Tokenizer() => ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
    private static Blip2Options Options() => new() { ImageSize = 14, EmbeddingDimension = 4, VisionDim = 4, NumQueryTokens = 2, MaxSequenceLength = 8 };
    private static Tensor<float> Image(int size, float value)
    {
        var image = new Tensor<float>(new[] { 3, size, size });
        for (int i = 0; i < image.Length; i++) image[i] = value;
        return image;
    }
    private static NeuralNetworkArchitecture<float> Architecture() => new(InputType.ThreeDimensional,
        NeuralNetworkTaskType.ImageClassification, inputDepth: 3, inputHeight: 14, inputWidth: 14, outputSize: 4);
}
