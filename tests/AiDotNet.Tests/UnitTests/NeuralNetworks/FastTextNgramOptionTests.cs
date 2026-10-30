using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// FastText's subword range is an option (Bojanowski et al. 2017, 3 &lt;= n &lt;= 6), not a constant (#2396).
/// </summary>
public class FastTextNgramOptionTests
{
    // The default tokenizer's vocabulary size; the softmax head must be the same width.
    private const int VocabSize = 30000;

    private static FastText<double> Create(int minN, int maxN) =>
        new FastText<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputSize: 32, outputSize: VocabSize),
            new FastTextOptions
            {
                VocabSize = VocabSize, BucketSize = 1024, EmbeddingDimension = 16, MaxSequenceLength = 32,
                MinNgramLength = minN, MaxNgramLength = maxN,
            });

    [Fact]
    public void Defaults_AreThePapersRange()
    {
        var options = new FastTextOptions();
        Assert.Equal(3, options.MinNgramLength);
        Assert.Equal(6, options.MaxNgramLength);
    }

    [Fact]
    public void Embed_UsesTheConfiguredNgramRange()
    {
        var paper = Create(3, 6);
        var trigramsOnly = Create(3, 3);
        var wordsOnly = Create(3, 0);
        // Same weights everywhere, so only the subword range can separate the embeddings.
        trigramsOnly.SetParameters(paper.GetParameters());
        wordsOnly.SetParameters(paper.GetParameters());

        const string text = "whereabouts";
        var a = paper.Embed(text);
        var b = trigramsOnly.Embed(text);
        var c = wordsOnly.Embed(text);

        Assert.True(MaxAbsDifference(a, b) > 1e-9, "3..3 n-grams embedded exactly like 3..6, so the range was ignored.");
        Assert.True(MaxAbsDifference(a, c) > 1e-9, "MaxNgramLength = 0 still added subwords.");
    }

    [Theory]
    [InlineData(0, 6)]
    [InlineData(5, 4)]
    public void Validate_RejectsAnEmptyOrInvertedRange(int minN, int maxN)
    {
        var options = new FastTextOptions { MinNgramLength = minN, MaxNgramLength = maxN };
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    private static double MaxAbsDifference(AiDotNet.Tensors.LinearAlgebra.Vector<double> x, AiDotNet.Tensors.LinearAlgebra.Vector<double> y)
    {
        double max = 0;
        for (int i = 0; i < x.Length; i++)
        {
            max = Math.Max(max, Math.Abs(x[i] - y[i]));
        }

        return max;
    }
}
