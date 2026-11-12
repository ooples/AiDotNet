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

    [Fact]
    public void Entmax15_HalfWideFiniteSpreadDoesNotOverflowTheThreshold()
    {
        // The sorted moments square the shifted logits: 500^2 = 250000 > Half.MaxValue (65504), which made
        // the threshold inf - inf = NaN. Reference entmax15([0, 1000, -1000]) = [0, 1, 0].
        var input = new Tensor<Half>(new Half[] { (Half)0, (Half)1000, (Half)(-1000) }, new[] { 1, 3 });
        var probabilities = Entmax15<Half>.Forward(AiDotNetEngine.Current, input).ToArray();
        Assert.Equal(new Half[] { (Half)0, (Half)1, (Half)0 }, probabilities);
    }

    [Fact]
    public void Entmax15_HalfWideSpreadKeepsTheExactTwoEntrySupport()
    {
        // entmax15([1000, 999.5, -1000]): z = x / 2 has support {500, 499.75}; tau solves
        // tau^2 + (tau + 0.25)^2 = 1 relative to the max, tau = -(0.5 + sqrt(7.75)) / 4, so
        // p = [0.6739926363, 0.3260073637, 0]. 999.5 is exact in Half (spacing 0.5 at 1000).
        var input = new Tensor<Half>(new Half[] { (Half)1000, (Half)999.5, (Half)(-1000) }, new[] { 1, 3 });
        var probabilities = Entmax15<Half>.Forward(AiDotNetEngine.Current, input).ToArray();
        double[] expected = { 0.6739926363384383, 0.3260073636615617, 0.0 };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(Math.Abs((double)probabilities[i] - expected[i]), 0, 2e-3);
        }
    }

    [Fact]
    public void Constructors_RejectUndefinedChoiceFunction()
    {
        var undefined = (ObliviousTreeChoiceFunction)2;
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ObliviousDecisionTreeLayer<double>(3, 2, 1, 0.01, undefined));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ObliviousDecisionTreeLayer<double>(2, 1, 0.01, undefined));
    }

    [Fact]
    public void NewLayer_DefaultsToEntmax15AndRoundTripsIt()
    {
        var layer = new ObliviousDecisionTreeLayer<double>(depth: 2, outputDim: 1);
        Assert.Equal(ObliviousTreeChoiceFunction.Entmax15, ChoiceFunction(layer));

        var metadata = layer.GetMetadata().ToDictionary(pair => pair.Key, pair => (object)pair.Value);
        Assert.True(metadata.ContainsKey("choiceFunction"));
        var restored = AiDotNet.Helpers.DeserializationHelper.CreateLayerFromType<double>(
            "ObliviousDecisionTreeLayer`1", new[] { 3 }, new[] { 1 }, metadata);

        Assert.Equal(ObliviousTreeChoiceFunction.Entmax15,
            ChoiceFunction(Assert.IsType<ObliviousDecisionTreeLayer<double>>(restored)));
    }

    [Fact]
    public void LegacyCheckpointWithoutChoiceFunction_RestoresSoftmaxSigmoidAndItsPredictions()
    {
        // A file written before the option existed: master's [LayerState] keys only, trained with softmax
        // feature choice and sigmoid splits.
        var original = new ObliviousDecisionTreeLayer<double>(3, 2, 2, 0.01, ObliviousTreeChoiceFunction.SoftmaxSigmoid);
        var input = new Tensor<double>(new[] { 0.5, -1.0, 2.0, 1.5, 0.25, -0.75 }, new[] { 2, 3 });
        var expected = original.Forward(input).ToArray();
        var weights = original.GetParameters();
        var legacy = new Dictionary<string, object> { ["depth"] = 2, ["outputDim"] = 2, ["initScale"] = 0.01 };

        var restored = Assert.IsType<ObliviousDecisionTreeLayer<double>>(
            AiDotNet.Helpers.DeserializationHelper.CreateLayerFromType<double>("ObliviousDecisionTreeLayer`1", new[] { 3 }, new[] { 2 }, legacy));
        Assert.Equal(ObliviousTreeChoiceFunction.SoftmaxSigmoid, ChoiceFunction(restored));
        restored.SetParameters(weights);

        var actual = restored.Forward(input).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
    }

    private static ObliviousTreeChoiceFunction ChoiceFunction(ObliviousDecisionTreeLayer<double> layer)
    {
        var field = typeof(ObliviousDecisionTreeLayer<double>).GetField("_choiceFunction",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            ?? throw new MissingFieldException("_choiceFunction");
        return field.GetValue(layer) is ObliviousTreeChoiceFunction value
            ? value
            : throw new Xunit.Sdk.XunitException("_choiceFunction is not an ObliviousTreeChoiceFunction.");
    }
}
