using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tokenization;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Conditioning;

/// <summary>
/// Tests for text conditioning modules: CLIP, T5, Dual, and Triple.
/// Each conditioner ctor now requires an explicit <c>ITokenizer</c> (PyTorch-style:
/// model construction and tokenizer loading are separate concerns). Tests pass
/// the existing small-vocab factory tokenizers, which produce real (minimal-vocab)
/// BPE / SentencePiece output without any network I/O.
/// </summary>
public class ConditioningModuleTests
{
    private static CLIPTextConditioner<double> NewClip(CLIPVariant variant = CLIPVariant.ViTL14) =>
        new CLIPTextConditioner<double>(ClipTokenizerFactory.CreateSimple(), variant);

    private static T5TextConditioner<double> NewT5(T5Variant variant = T5Variant.Base) =>
        new T5TextConditioner<double>(
            LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5),
            variant);

    #region CLIP Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_DefaultVariant_Creates768DimEmbedding()
    {
        var clip = NewClip();

        Assert.Equal(768, clip.EmbeddingDimension);
        Assert.Equal(77, clip.MaxSequenceLength);
        Assert.True(clip.ProducesPooledOutput);
        Assert.Equal(ConditioningType.Text, clip.ConditioningType);
    }

    [Theory]
    [InlineData(CLIPVariant.ViTL14, 768)]
    [InlineData(CLIPVariant.ViTH14, 1024)]
    [InlineData(CLIPVariant.ViTBigG14, 1280)]
    public void CLIPConditioner_Variants_HaveCorrectDimensions(CLIPVariant variant, int expectedDim)
    {
        var clip = NewClip(variant);

        Assert.Equal(expectedDim, clip.EmbeddingDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_Tokenize_ReturnsCorrectShape()
    {
        var clip = NewClip();

        var tokens = clip.Tokenize("a cat sitting on a mat");

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(1, tokens.Shape[0]);
        Assert.Equal(77, tokens.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task CLIPConditioner_TokenizeBatch_ReturnsCorrectShape()
    {
        var clip = NewClip();

        var tokens = clip.TokenizeBatch(new[] { "a cat", "a dog", "a bird" });

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(3, tokens.Shape[0]);
        Assert.Equal(77, tokens.Shape[1]);
    }

    #endregion

    #region T5 Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task T5Conditioner_DefaultVariant_HasCorrectDimensions()
    {
        var t5 = NewT5();

        Assert.Equal(768, t5.EmbeddingDimension);
        Assert.Equal(512, t5.MaxSequenceLength);
        Assert.False(t5.ProducesPooledOutput);
        Assert.Equal(ConditioningType.Text, t5.ConditioningType);
    }

    [Theory]
    [InlineData(T5Variant.Small, 512)]
    [InlineData(T5Variant.Base, 768)]
    [InlineData(T5Variant.Large, 1024)]
    [InlineData(T5Variant.XL, 2048)]
    [InlineData(T5Variant.XXL, 4096)]
    public void T5Conditioner_Variants_HaveCorrectDimensions(T5Variant variant, int expectedDim)
    {
        var t5 = NewT5(variant);

        Assert.Equal(expectedDim, t5.EmbeddingDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task T5Conditioner_Tokenize_ReturnsCorrectShape()
    {
        var t5 = NewT5();

        var tokens = t5.Tokenize("a cat sitting on a mat");

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(1, tokens.Shape[0]);
        Assert.Equal(512, tokens.Shape[1]);
    }

    #endregion

    #region Dual Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task DualConditioner_FromEncoders_HasCorrectProperties()
    {
        var dual = new DualTextConditioner<double>(
            clipEncoder: NewClip(),
            t5Encoder: NewT5());

        Assert.True(dual.EmbeddingDimension > 0);
        Assert.True(dual.MaxSequenceLength > 0);
        Assert.Equal(ConditioningType.MultiModal, dual.ConditioningType);
    }

    #endregion

    #region Triple Text Conditioner Tests

    [Fact(Timeout = 120000)]
    public async Task TripleConditioner_FromEncoders_HasCorrectProperties()
    {
        var triple = new TripleTextConditioner<double>(
            clipLEncoder: NewClip(CLIPVariant.ViTL14),
            clipGEncoder: NewClip(CLIPVariant.ViTBigG14),
            t5Encoder: NewT5(T5Variant.XXL));

        Assert.True(triple.EmbeddingDimension > 0);
        Assert.Equal(ConditioningType.MultiModal, triple.ConditioningType);
    }

    [Fact(Timeout = 120000)]
    public async Task TripleConditioner_CustomVariants_DimensionsMatchSelection()
    {
        var triple = new TripleTextConditioner<double>(
            clipLEncoder: NewClip(CLIPVariant.ViTL14),
            clipGEncoder: NewClip(CLIPVariant.ViTH14),
            t5Encoder: NewT5(T5Variant.XL));

        Assert.Equal(768, triple.CLIPLEmbeddingDimension);
        Assert.Equal(1024, triple.CLIPGEmbeddingDimension);
        Assert.Equal(2048, triple.T5EmbeddingDimension);
        Assert.Equal(1792, triple.CombinedPooledDimension);
    }

    #endregion

    #region IConditioningModule Interface Compliance Tests

    [Fact(Timeout = 120000)]
    public async Task AllConditioners_ImplementIConditioningModule()
    {
        var clip = NewClip();
        var t5 = NewT5();
        var conditioners = new IConditioningModule<double>[]
        {
            clip,
            t5,
            new DualTextConditioner<double>(NewClip(), NewT5()),
            new TripleTextConditioner<double>(NewClip(CLIPVariant.ViTL14), NewClip(CLIPVariant.ViTBigG14), NewT5(T5Variant.XXL)),
        };

        foreach (var conditioner in conditioners)
        {
            Assert.True(conditioner.EmbeddingDimension > 0,
                $"{conditioner.GetType().Name} should have positive EmbeddingDimension");
            Assert.True(conditioner.MaxSequenceLength > 0,
                $"{conditioner.GetType().Name} should have positive MaxSequenceLength");

            var tokens = conditioner.Tokenize("test");
            Assert.NotNull(tokens);

            var batchTokens = conditioner.TokenizeBatch(new[] { "test1", "test2" });
            Assert.NotNull(batchTokens);
            Assert.Equal(2, batchTokens.Shape[0]);
        }
    }

    #endregion
}
