using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// ColBERT encodes every query as exactly N<sub>q</sub> tokens, padding with [MASK] (Khattab &amp; Zaharia 2020,
/// section 3.2) (#2396).
/// </summary>
public class ColBERTQueryAugmentationTests
{
    private const int QueryLength = 8;

    private static ColBERT<double> Create() =>
        new ColBERT<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 16, outputSize: 8),
            new ColBERTOptions
            {
                EmbeddingDimension = 16, OutputDimension = 8, MaxSequenceLength = 32, MaxQueryLength = QueryLength,
                NumLayers = 1, NumHeads = 2, FeedForwardDim = 32,
            });

    [Fact]
    public void Default_IsThePapersQueryLength()
    {
        Assert.Equal(32, new ColBERTOptions().MaxQueryLength);
    }

    [Fact]
    public void ShortQuery_IsPaddedToTheQueryLength_WithMaskTokens()
    {
        var model = Create();

        var query = model.EmbedQueryLateInteraction("a cat");
        var document = model.EmbedLateInteraction("a cat");

        Assert.Equal(QueryLength, query.Rows);
        Assert.True(document.Rows < QueryLength, $"The document was padded to {document.Rows} rows; only queries are augmented.");
    }

    [Fact]
    public void ShortQuery_PaddingIsTheMaskToken_NotAnyOtherId()
    {
        var tokenizer = AiDotNet.Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.RoBERTa);
        var model = new ColBERT<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 16, outputSize: 8),
            new ColBERTOptions
            {
                EmbeddingDimension = 16, OutputDimension = 8, MaxSequenceLength = 32, MaxQueryLength = QueryLength,
                NumLayers = 1, NumHeads = 2, FeedForwardDim = 32,
            },
            tokenizer: tokenizer);
        var ids = tokenizer.Encode("a cat").TokenIds.ToList();
        Assert.True(ids.Count < QueryLength, "The fixture query is too long to need padding.");
        int maskId = tokenizer.Vocabulary.GetTokenId(tokenizer.SpecialTokens.MaskToken);
        int padId = tokenizer.Vocabulary.GetTokenId(tokenizer.SpecialTokens.PadToken);
        Assert.NotEqual(maskId, padId);

        var query = model.EmbedQueryLateInteraction("a cat");
        var paddedWithMask = Encode(model, ids, maskId);
        var paddedWithPad = Encode(model, ids, padId);

        for (int s = 0; s < QueryLength; s++)
            for (int d = 0; d < query.Columns; d++)
                Assert.Equal(paddedWithMask[s, d], query[s, d], 10);

        // Guards the comparison above: a different padding id must change the padded rows, or it proves nothing.
        bool differs = false;
        for (int s = ids.Count; s < QueryLength && !differs; s++)
            for (int d = 0; d < query.Columns && !differs; d++)
                differs = Math.Abs(paddedWithPad[s, d] - query[s, d]) > 1e-6;
        Assert.True(differs, "Padding with the pad token gave the same embeddings as the mask token.");
    }

    private static double[,] Encode(ColBERT<double> model, System.Collections.Generic.List<int> ids, int padWith)
    {
        var tokens = ids.Concat(Enumerable.Repeat(padWith, QueryLength - ids.Count)).Select(id => (double)id).ToArray();
        var output = model.Predict(Tensor<double>.FromVector(new Vector<double>(tokens), new[] { 1, QueryLength }));
        int width = output.Shape[2];
        var rows = new double[QueryLength, width];
        for (int s = 0; s < QueryLength; s++)
        {
            double norm = 0;
            for (int d = 0; d < width; d++) norm += output[0, s, d] * output[0, s, d];
            norm = Math.Sqrt(norm);
            for (int d = 0; d < width; d++) rows[s, d] = norm > 0 ? output[0, s, d] / norm : 0;
        }

        return rows;
    }

    [Fact]
    public void ShortQuery_WithATokenizerThatHasNoMask_IsRefusedRatherThanPaddedWithSomethingElse()
    {
        var gptStyle = AiDotNet.Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
        var model = new ColBERT<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 16, outputSize: 8),
            new ColBERTOptions
            {
                EmbeddingDimension = 16, OutputDimension = 8, MaxSequenceLength = 32, MaxQueryLength = QueryLength,
                NumLayers = 1, NumHeads = 2, FeedForwardDim = 32,
            },
            tokenizer: gptStyle);

        Assert.Throws<InvalidOperationException>(() => model.EmbedQueryLateInteraction("a cat"));
    }

    [Fact]
    public void LongQuery_IsTruncatedToTheQueryLength()
    {
        var model = Create();
        string longQuery = string.Join(" ", Enumerable.Repeat("a photo of a cat", 10));

        Assert.True(model.EmbedLateInteraction(longQuery).Rows > QueryLength, "The fixture query is not long enough to truncate.");
        Assert.Equal(QueryLength, model.EmbedQueryLateInteraction(longQuery).Rows);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(33)]
    public void Validate_RejectsAQueryLengthOutsideTheSequence(int queryLength)
    {
        var options = new ColBERTOptions { MaxSequenceLength = 32, MaxQueryLength = queryLength };
        Assert.Throws<ArgumentException>(() => options.Validate());
    }
}
