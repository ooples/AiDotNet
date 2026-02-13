using AiDotNet.Diffusion.Conditioning;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Conditioning;

/// <summary>
/// Tests for text conditioning modules: CLIP, T5, Dual, and Triple.
/// </summary>
public class ConditioningModuleTests
{
    #region CLIP Text Conditioner Tests

    [Fact]
    public void CLIPConditioner_DefaultVariant_Creates768DimEmbedding()
    {
        var clip = new CLIPTextConditioner<double>();

        Assert.Equal(768, clip.EmbeddingDimension);
        Assert.Equal(77, clip.MaxSequenceLength);
        Assert.True(clip.ProducesPooledOutput);
        Assert.Equal(ConditioningType.Text, clip.ConditioningType);
    }

    [Theory]
    [InlineData("ViT-L/14", 768)]
    [InlineData("ViT-H/14", 1024)]
    [InlineData("ViT-bigG/14", 1280)]
    public void CLIPConditioner_Variants_HaveCorrectDimensions(string variant, int expectedDim)
    {
        var clip = new CLIPTextConditioner<double>(variant: variant);

        Assert.Equal(expectedDim, clip.EmbeddingDimension);
    }

    [Fact]
    public void CLIPConditioner_Tokenize_ReturnsCorrectShape()
    {
        var clip = new CLIPTextConditioner<double>();

        var tokens = clip.Tokenize("a cat sitting on a mat");

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(1, tokens.Shape[0]); // batch size 1
        Assert.Equal(77, tokens.Shape[1]); // max sequence length
    }

    [Fact]
    public void CLIPConditioner_TokenizeBatch_ReturnsCorrectShape()
    {
        var clip = new CLIPTextConditioner<double>();

        var tokens = clip.TokenizeBatch(new[] { "a cat", "a dog", "a bird" });

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(3, tokens.Shape[0]); // batch size 3
        Assert.Equal(77, tokens.Shape[1]);
    }

    [Fact]
    public void CLIPConditioner_EncodeText_ReturnsFiniteValues()
    {
        var clip = new CLIPTextConditioner<double>(seed: 42);
        var tokens = clip.Tokenize("a beautiful sunset");
        var embeddings = clip.EncodeText(tokens);

        Assert.Equal(3, embeddings.Shape.Length); // [batch, seqLen, embDim]
        Assert.Equal(1, embeddings.Shape[0]);
        Assert.Equal(77, embeddings.Shape[1]);
        Assert.Equal(768, embeddings.Shape[2]);

        // Check for finite values
        var span = embeddings.AsSpan();
        for (int i = 0; i < Math.Min(100, span.Length); i++)
        {
            Assert.False(double.IsNaN(span[i]), $"Embedding[{i}] is NaN");
            Assert.False(double.IsInfinity(span[i]), $"Embedding[{i}] is Infinity");
        }
    }

    [Fact]
    public void CLIPConditioner_GetPooledEmbedding_ReturnsCorrectShape()
    {
        var clip = new CLIPTextConditioner<double>(seed: 42);
        var tokens = clip.Tokenize("a cat");
        var embeddings = clip.EncodeText(tokens);
        var pooled = clip.GetPooledEmbedding(embeddings);

        Assert.Equal(2, pooled.Shape.Length); // [batch, embDim]
        Assert.Equal(1, pooled.Shape[0]);
        Assert.Equal(768, pooled.Shape[1]);
    }

    [Fact]
    public void CLIPConditioner_GetUnconditionalEmbedding_ReturnsCorrectShape()
    {
        var clip = new CLIPTextConditioner<double>(seed: 42);
        var uncond = clip.GetUnconditionalEmbedding(batchSize: 2);

        Assert.Equal(3, uncond.Shape.Length); // [batch, seqLen, embDim]
        Assert.Equal(2, uncond.Shape[0]);
    }

    #endregion

    #region T5 Text Conditioner Tests

    [Fact]
    public void T5Conditioner_DefaultVariant_Creates4096DimEmbedding()
    {
        var t5 = new T5TextConditioner<double>();

        Assert.Equal(4096, t5.EmbeddingDimension);
        Assert.Equal(256, t5.MaxSequenceLength);
        Assert.False(t5.ProducesPooledOutput);
        Assert.Equal(ConditioningType.Text, t5.ConditioningType);
    }

    [Theory]
    [InlineData("T5-XXL", 4096)]
    [InlineData("T5-XL", 2048)]
    [InlineData("T5-Large", 1024)]
    public void T5Conditioner_Variants_HaveCorrectDimensions(string variant, int expectedDim)
    {
        var t5 = new T5TextConditioner<double>(variant: variant);

        Assert.Equal(expectedDim, t5.EmbeddingDimension);
    }

    [Fact]
    public void T5Conditioner_Tokenize_ReturnsCorrectShape()
    {
        var t5 = new T5TextConditioner<double>();

        var tokens = t5.Tokenize("a cat sitting on a mat");

        Assert.Equal(2, tokens.Shape.Length);
        Assert.Equal(1, tokens.Shape[0]);
        Assert.Equal(256, tokens.Shape[1]);
    }

    [Fact]
    public void T5Conditioner_EncodeText_ReturnsFiniteValues()
    {
        var t5 = new T5TextConditioner<double>(seed: 42);
        var tokens = t5.Tokenize("a detailed landscape");
        var embeddings = t5.EncodeText(tokens);

        Assert.Equal(3, embeddings.Shape.Length); // [batch, seqLen, embDim]
        Assert.Equal(1, embeddings.Shape[0]);
        Assert.Equal(256, embeddings.Shape[1]);
        Assert.Equal(4096, embeddings.Shape[2]);
    }

    #endregion

    #region Dual Text Conditioner Tests

    [Fact]
    public void DualConditioner_DefaultConfig_HasCorrectProperties()
    {
        var dual = new DualTextConditioner<double>(seed: 42);

        Assert.Equal(4096, dual.EmbeddingDimension); // T5 dimension for cross-attention
        Assert.Equal(768, dual.CLIPEmbeddingDimension);
        Assert.Equal(4096, dual.T5EmbeddingDimension);
        Assert.Equal(256, dual.MaxSequenceLength); // T5's max
        Assert.True(dual.ProducesPooledOutput);
        Assert.Equal(ConditioningType.MultiModal, dual.ConditioningType);
    }

    [Fact]
    public void DualConditioner_EncodeDual_ReturnsBothEmbeddings()
    {
        var dual = new DualTextConditioner<double>(seed: 42);

        var (seqEmb, pooledEmb) = dual.EncodeDual("a starry night");

        // T5 sequence embeddings
        Assert.Equal(3, seqEmb.Shape.Length);
        Assert.Equal(1, seqEmb.Shape[0]);
        Assert.Equal(4096, seqEmb.Shape[2]);

        // CLIP pooled embedding
        Assert.Equal(2, pooledEmb.Shape.Length);
        Assert.Equal(1, pooledEmb.Shape[0]);
        Assert.Equal(768, pooledEmb.Shape[1]);
    }

    [Fact]
    public void DualConditioner_GetUnconditionalDual_ReturnsBothEmbeddings()
    {
        var dual = new DualTextConditioner<double>(seed: 42);

        var (seqEmb, pooledEmb) = dual.GetUnconditionalDual(batchSize: 1);

        Assert.NotNull(seqEmb);
        Assert.NotNull(pooledEmb);
    }

    [Fact]
    public void DualConditioner_Tokenize_DefaultsToT5()
    {
        var dual = new DualTextConditioner<double>(seed: 42);

        var tokens = dual.Tokenize("a prompt");

        Assert.Equal(256, tokens.Shape[1]); // T5 max sequence length
    }

    #endregion

    #region Triple Text Conditioner Tests

    [Fact]
    public void TripleConditioner_DefaultConfig_HasCorrectProperties()
    {
        var triple = new TripleTextConditioner<double>(seed: 42);

        Assert.Equal(4096, triple.EmbeddingDimension); // T5 dimension for cross-attention
        Assert.Equal(768, triple.CLIPLEmbeddingDimension); // CLIP ViT-L/14
        Assert.Equal(1280, triple.CLIPGEmbeddingDimension); // CLIP ViT-bigG/14
        Assert.Equal(4096, triple.T5EmbeddingDimension); // T5-XXL
        Assert.Equal(2048, triple.CombinedPooledDimension); // 768 + 1280
        Assert.Equal(256, triple.MaxSequenceLength); // T5's max
        Assert.True(triple.ProducesPooledOutput);
        Assert.Equal(ConditioningType.MultiModal, triple.ConditioningType);
    }

    [Fact]
    public void TripleConditioner_EncodeTriple_ReturnsBothEmbeddings()
    {
        var triple = new TripleTextConditioner<double>(seed: 42);

        var (seqEmb, combinedPooled) = triple.EncodeTriple("a serene mountain lake");

        // T5 sequence embeddings
        Assert.Equal(3, seqEmb.Shape.Length);
        Assert.Equal(1, seqEmb.Shape[0]);
        Assert.Equal(4096, seqEmb.Shape[2]);

        // Combined CLIP-L + CLIP-G pooled embedding
        Assert.Equal(2, combinedPooled.Shape.Length);
        Assert.Equal(1, combinedPooled.Shape[0]);
        Assert.Equal(2048, combinedPooled.Shape[1]); // 768 + 1280
    }

    [Fact]
    public void TripleConditioner_GetCombinedPooledEmbedding_Returns2048Dim()
    {
        var triple = new TripleTextConditioner<double>(seed: 42);

        var pooled = triple.GetCombinedPooledEmbedding("a prompt");

        Assert.Equal(2, pooled.Shape.Length);
        Assert.Equal(1, pooled.Shape[0]);
        Assert.Equal(2048, pooled.Shape[1]);
    }

    [Fact]
    public void TripleConditioner_GetUnconditionalTriple_ReturnsBothEmbeddings()
    {
        var triple = new TripleTextConditioner<double>(seed: 42);

        var (seqEmb, combinedPooled) = triple.GetUnconditionalTriple(batchSize: 1);

        Assert.NotNull(seqEmb);
        Assert.NotNull(combinedPooled);
        Assert.Equal(2048, combinedPooled.Shape[1]);
    }

    [Fact]
    public void TripleConditioner_EncodeTriple_ValuesAreFinite()
    {
        var triple = new TripleTextConditioner<double>(seed: 42);

        var (seqEmb, combinedPooled) = triple.EncodeTriple("test");

        // Check a sample of values
        var seqSpan = seqEmb.AsSpan();
        for (int i = 0; i < Math.Min(100, seqSpan.Length); i++)
        {
            Assert.False(double.IsNaN(seqSpan[i]), $"SeqEmb[{i}] is NaN");
        }

        var pooledSpan = combinedPooled.AsSpan();
        for (int i = 0; i < pooledSpan.Length; i++)
        {
            Assert.False(double.IsNaN(pooledSpan[i]), $"CombinedPooled[{i}] is NaN");
        }
    }

    [Fact]
    public void TripleConditioner_CustomVariants_Creates()
    {
        var triple = new TripleTextConditioner<double>(
            clipLVariant: "ViT-L/14",
            clipGVariant: "ViT-H/14",  // Use ViT-H instead of default ViT-bigG
            t5Variant: "T5-XL",
            seed: 42);

        Assert.Equal(768, triple.CLIPLEmbeddingDimension);
        Assert.Equal(1024, triple.CLIPGEmbeddingDimension); // ViT-H/14 = 1024
        Assert.Equal(2048, triple.T5EmbeddingDimension);     // T5-XL = 2048
        Assert.Equal(1792, triple.CombinedPooledDimension);  // 768 + 1024
    }

    #endregion

    #region IConditioningModule Interface Compliance Tests

    [Fact]
    public void AllConditioners_ImplementIConditioningModule()
    {
        var conditioners = new IConditioningModule<double>[]
        {
            new CLIPTextConditioner<double>(seed: 42),
            new T5TextConditioner<double>(seed: 42),
            new DualTextConditioner<double>(seed: 42),
            new TripleTextConditioner<double>(seed: 42),
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
        }
    }

    #endregion
}
