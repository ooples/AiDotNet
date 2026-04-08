using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for financial NLP models (FinBERT, BloombergGPT, etc.).
/// Inherits financial model invariants and adds NLP-specific: text sensitivity
/// and bounded sentiment/classification scores.
/// </summary>
public abstract class FinancialNLPTestBase : FinancialModelTestBase
{
    [Fact]
    public void DifferentText_DifferentSentiment()
    {
        var network = CreateNetwork();
        var positive = CreateConstantTensor(InputShape, 0.9);
        var negative = CreateConstantTensor(InputShape, 0.1);

        var out1 = network.Predict(positive);
        var out2 = network.Predict(negative);

        bool anyDifferent = false;
        int minLen = Math.Min(out1.Length, out2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(out1[i] - out2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Financial NLP produces identical output for different text — ignoring input.");
    }

    [Fact]
    public void SentimentScores_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Financial NLP output[{i}] is NaN.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Financial NLP output[{i}] = {output[i]:E4} is unbounded.");
        }
    }
}
