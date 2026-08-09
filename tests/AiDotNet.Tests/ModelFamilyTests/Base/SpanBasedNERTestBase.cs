using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for span-based NER models (BiaffineNER, PyramidNER, etc.).
/// Inherits NER invariants and adds span-specific: bounded span scores and non-empty output.
/// </summary>
public abstract class SpanBasedNERTestBase<T> : NERModelTestBase<T>
{
    [Fact(Timeout = 120000)]
    public async Task SpanScores_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double value = ConvertToDouble(output[i]);
            Assert.True(Math.Abs(value) < 1e6,
                $"Span score[{i}] = {value:E4} is unbounded.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SpanOutput_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Span-based NER produced empty output.");
    }
}

/// <summary>Default double-precision alias for existing span-based NER tests.</summary>
public abstract class SpanBasedNERTestBase : SpanBasedNERTestBase<double> { }
