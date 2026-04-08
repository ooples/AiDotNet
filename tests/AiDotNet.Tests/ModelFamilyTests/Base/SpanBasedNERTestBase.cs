using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for span-based NER models (BiaffineNER, PyramidNER, etc.).
/// Inherits NER invariants and adds span-specific: bounded span scores and non-empty output.
/// </summary>
public abstract class SpanBasedNERTestBase : NERModelTestBase
{
    [Fact]
    public void SpanScores_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Span score[{i}] = {output[i]:E4} is unbounded.");
        }
    }

    [Fact]
    public void SpanOutput_ShouldBeNonEmpty()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Span-based NER produced empty output.");
    }
}
