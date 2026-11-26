using System;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Canary's decoder ends in a projection to the vocabulary (Puvvada et al. 2024): each output position holds one
/// logit per token, and the token is the argmax over that row.
/// </summary>
public class CanaryDecodingTests
{
    [Fact]
    public void DecodeTokenIds_TakesTheArgmaxOfEachPositionsVocabularyLogits()
    {
        // Two positions over a 4-token vocabulary. The tokens are 2 and 3; rounding the raw values (the old decoder)
        // would read eight ids [0, 2, 3, 1, 1, 0, 2, 4] instead.
        var logits = new Tensor<double>(new[] { 0.1, 2.0, 3.4, 1.2, 0.6, -0.2, 1.9, 4.1 }, new[] { 2, 4 });

        var ids = Canary<double>.DecodeTokenIds(AiDotNetEngine.Current, logits, 4, 10, 100);

        Assert.Equal(new[] { 2, 3 }, ids);
    }

    [Fact]
    public void DecodeTokenIds_ReadsAtMostMaxTokensPositions_AndNaNNeverWins()
    {
        var logits = new Tensor<double>(new[] { double.NaN, 1.0, 0.5, 0.0, 0.0, 9.0 }, new[] { 3, 2 });

        Assert.Equal(new[] { 1, 0 }, Canary<double>.DecodeTokenIds(AiDotNetEngine.Current, logits, 2, 2, 100));
    }

    [Fact]
    public void DecodeTokenIds_RejectsOutputsThatAreNotVocabularyRows()
    {
        var logits = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });

        Assert.Throws<ArgumentException>(() => Canary<double>.DecodeTokenIds(AiDotNetEngine.Current, logits, 2, 10, 100));
    }
}
