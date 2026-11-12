using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Reference tests for NODE's choice functions (Popov et al. 2019) in <see cref="ObliviousDecisionTreeLayer{T}"/>.
/// Expected values come from the reference <c>entmax</c> package algorithm (sort-based 1.5-entmax) and
/// NODE's closed-form <c>entmoid15</c>.
/// </summary>
public sealed class ObliviousDecisionTreeChoiceFunctionTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(1e16)]
    public void Entmax15_EqualLogitsRemainNormalizedAtALargeCommonOffset(double offset)
    {
        var logits = new Tensor<double>(new[] { offset, offset }, new[] { 1, 2 });
        var probabilities = Entmax15<double>.Forward(AiDotNetEngine.Current, logits).ToArray();
        Assert.All(probabilities, probability => Assert.Equal(0.5, probability, 12));
    }

    [Fact]
    public void Entmax15_HalfLargeEqualLogitsRemainFiniteAndNormalized()
    {
        var logits = new Tensor<Half>(new Half[] { (Half)10000, (Half)10000 }, new[] { 1, 2 });
        var probabilities = Entmax15<Half>.Forward(AiDotNetEngine.Current, logits).ToArray();
        Assert.Equal(new Half[] { (Half)0.5, (Half)0.5 }, probabilities);
    }

    [Fact]
    public void Entmax15_LargeCommonOffsetPreservesTheReferenceJacobian()
    {
        var logits = new Tensor<double>(new[] { 1e16, 1e16, 1e16, 1e16 }, new[] { 1, 4 });
        var coefficients = new Tensor<double>(new[] { 1.0, -1.0, 2.0, -2.0 }, new[] { 1, 4 });
        var engine = AiDotNetEngine.Current;
        using var tape = new GradientTape<double>();
        var probabilities = Entmax15<double>.Forward(engine, logits);
        var loss = engine.ReduceSum(engine.TensorMultiply(probabilities, coefficients), new[] { 0, 1 });
        var gradients = tape.ComputeGradients(loss, new[] { logits });
        Assert.True(gradients.TryGetValue(logits, out var gradient));
        var expected = new[] { 0.5, -0.5, 1.0, -1.0 }; // sqrt(p)=0.5 and sum(coefficients)=0.
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], gradient!.GetFlat(i), 12);
    }

    [Theory]
    [InlineData(1.0, 0.8307189138830738)]
    [InlineData(-1.0, 0.1692810861169262)]
    [InlineData(0.5, 0.6739926363384383)]
    [InlineData(1.9, 0.9976177624442278)]
    [InlineData(3.0, 1.0)]   // sparse: sigmoid would give 0.9526
    [InlineData(-3.0, 0.0)]
    public void DefaultSplit_IsEntmoid15(double feature, double expected)
    {
        // Depth 1, one feature, threshold 0, leaves [0, 1]: the output is the right-branch probability.
        var layer = new ObliviousDecisionTreeLayer<double>(inputDim: 1, depth: 1, outputDim: 1);
        layer.SetParameters(new Vector<double>(new[] { 0.0, 0.0, 0.0, 1.0 }));

        var output = layer.Forward(new Tensor<double>(new[] { feature }, new[] { 1, 1 }));

        Assert.Equal(expected, output.GetFlat(0), precision: 12);
    }

    [Fact]
    public void DefaultFeatureChoice_IsEntmax15()
    {
        // entmax15([1.0, 0.5, -0.2]) = [0.64901497, 0.30870764, 0.04227739]; the weighted feature minus the
        // threshold 2.5 is 0.365957463361255, and entmoid15 of that is 0.6282979351485183.
        var layer = new ObliviousDecisionTreeLayer<double>(inputDim: 3, depth: 1, outputDim: 1);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 0.5, -0.2, 2.5, 0.0, 1.0 }));

        var output = layer.Forward(new Tensor<double>(new[] { 0.5, 7.0, 9.0 }, new[] { 1, 3 }));

        Assert.Equal(0.6282979351485183, output.GetFlat(0), precision: 12);
    }

    [Fact]
    public void SoftmaxSigmoidOption_KeepsTheSmoothChoiceFunctions()
    {
        var layer = new ObliviousDecisionTreeLayer<double>(inputDim: 3, depth: 1, outputDim: 1, initScale: 0.01,
            choiceFunction: ObliviousTreeChoiceFunction.SoftmaxSigmoid);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 0.5, -0.2, 2.5, 0.0, 1.0 }));

        var output = layer.Forward(new Tensor<double>(new[] { 0.5, 7.0, 9.0 }, new[] { 1, 3 }));

        double e0 = Math.Exp(1.0), e1 = Math.Exp(0.5), e2 = Math.Exp(-0.2), sum = e0 + e1 + e2;
        double weighted = ((e0 * 0.5) + (e1 * 7.0) + (e2 * 9.0)) / sum;
        Assert.Equal(1.0 / (1.0 + Math.Exp(-(weighted - 2.5))), output.GetFlat(0), precision: 12);
    }

    [Fact]
    public void Entmax15_GradientMatchesTheReferenceJacobian()
    {
        // d/dx of c . entmax15(x) for x = [1, 0.5, -0.2, -3], c = [1, -2, 0.5, 4]: the reference backward
        // (s = sqrt(p), J = s_i (delta_ij - s_j / sum s)) gives these; the last entry is outside the support.
        var logits = new Tensor<double>(new[] { 1.0, 0.5, -0.2, -3.0 }, new[] { 1, 4 });
        var weights = new Tensor<double>(new[] { 1.0, -2.0, 0.5, 4.0 }, new[] { 1, 4 });
        var engine = AiDotNetEngine.Current;

        using var tape = new GradientTape<double>();
        var probabilities = Entmax15<double>.Forward(engine, logits);
        var loss = engine.ReduceSum(engine.TensorMultiply(probabilities, weights), new[] { 0, 1 });
        var gradients = tape.ComputeGradients(loss, new[] { logits });

        var expectedProbabilities = new[] { 0.6490149705243009, 0.30870764359109426, 0.04227738588460495, 0.0 };
        var expectedGradient = new[] { 0.9098908667460323, -1.0393123041783987, 0.12942143743236667, 0.0 };
        Assert.True(gradients.TryGetValue(logits, out var gradient));
        if (gradient is null) throw new Xunit.Sdk.XunitException("entmax15 disconnected the input gradient.");
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(expectedProbabilities[i], probabilities.GetFlat(i), precision: 12);
            Assert.Equal(expectedGradient[i], gradient.GetFlat(i), precision: 10);
        }
    }
}
