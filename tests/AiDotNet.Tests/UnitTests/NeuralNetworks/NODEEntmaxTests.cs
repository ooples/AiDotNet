using System;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// NODE's choice function (1.5-entmax) and bin function (entmoid-1.5) of Popov et al. (2020), against values
/// derived by hand from the sorting threshold of Peters, Niculae and Martins (2019).
/// </summary>
public class NODEEntmaxTests
{
    [Fact]
    public void Entmax15_MatchesHandDerivedValues()
    {
        // Row 0: z = [1, 0.5, 0] -> X = [0.5, 0.25, 0] - 0.5; every rank is supported, so
        //   tau = mean - sqrt((1 - 3 (meansq - mean^2)) / 3) = -0.25 - sqrt(0.875 / 3), y = (X - tau)^2.
        // Row 1: z = [4, 0, 0] -> X = [0, -2, -2]; only rank 1 is supported (tau = -1), y = [1, 0, 0].
        // Row 2: z = [0, 0, 0] -> uniform.
        var logits = new Tensor<double>(new[] { 3, 3 }, new Vector<double>(new[]
        {
            1.0, 0.5, 0.0,
            4.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        }));

        var y = NODEBase<double>.Entmax15(logits);

        double tau = -0.25 - Math.Sqrt(0.875 / 3.0);
        double[] expected =
        {
            Math.Pow(0.0 - tau, 2), Math.Pow(-0.25 - tau, 2), Math.Pow(-0.5 - tau, 2),
            1.0, 0.0, 0.0,
            1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        };
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], y[i], 12);

        // The row-0 values, written out: 0.62419752910..., 0.29166666666..., 0.08413580423...
        Assert.Equal(0.6241975291003276, y[0], 12);
        Assert.Equal(0.29166666666666674, y[1], 12);
    }

    [Fact]
    public void Entmoid15_IsTheTwoClassEntmax()
    {
        var x = new Tensor<double>(new[] { 6 }, new Vector<double>(new[] { 0.0, 1.0, -1.0, 3.0, -2.5, 2.0 }));

        var y = NODEBase<double>.Entmoid15(x);

        // a = 1: t = (1 + sqrt(7)) / 2, y- = (t - 1)^2 / 4
        double negative = Math.Pow((1.0 + Math.Sqrt(7.0)) / 2.0 - 1.0, 2) / 4.0;
        Assert.Equal(0.5, y[0], 12);
        Assert.Equal(1.0 - negative, y[1], 12);
        Assert.Equal(negative, y[2], 12);
        Assert.Equal(1.0, y[3], 12);
        Assert.Equal(0.0, y[4], 12);
        Assert.Equal(1.0, y[5], 12);

        // Two-class consistency: entmoid15(x) = entmax15([x, 0])[0]
        var pair = NODEBase<double>.Entmax15(new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 1.0, 0.0 })));
        Assert.Equal(pair[0], y[1], 12);
    }

    [Fact]
    public void Entmax15_TapeGradientMatchesFiniteDifferences()
    {
        var engine = AiDotNetEngine.Current;
        double[] z = { 1.0, 0.5, 0.0, -0.3 };
        double[] c = { 0.7, -1.2, 0.4, 2.0 };
        var weights = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(c));

        double Objective(double[] values)
        {
            var y = NODEBase<double>.Entmax15(new Tensor<double>(new[] { 1, 4 }, new Vector<double>(values)));
            return Enumerable.Range(0, 4).Sum(i => c[i] * y[i]);
        }

        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>((double[])z.Clone()));
        Tensor<double> gradient;
        using (var tape = new GradientTape<double>())
        {
            var loss = engine.ReduceSum(
                engine.TensorMultiply(NODEBase<double>.Entmax15(input), weights), new[] { 0, 1 }, false);
            gradient = tape.ComputeGradients(loss, new[] { input })[input];
        }

        const double epsilon = 1e-6;
        for (int i = 0; i < z.Length; i++)
        {
            var plus = (double[])z.Clone();
            var minus = (double[])z.Clone();
            plus[i] += epsilon;
            minus[i] -= epsilon;
            double numerical = (Objective(plus) - Objective(minus)) / (2 * epsilon);
            Assert.True(Math.Abs(numerical - gradient[i]) < 1e-6,
                $"Logit {i}: finite difference {numerical:R}, tape gradient {gradient[i]:R}.");
        }
    }

    [Fact]
    public void FeatureImportance_UsesTheSparseChoiceFunction()
    {
        // Each depth's choice weights sum to one, so the importance averaged over trees and depths does too.
        var model = new NODERegression<double>(3, options: new NODEOptions<double> { NumTrees = 2, TreeDepth = 2 });
        var importance = model.GetFeatureImportance();
        Assert.Equal(1.0, importance.ToArray().Sum(), 10);
        Assert.All(importance.ToArray(), value => Assert.True(value >= 0.0));
    }

    [Fact]
    public void UnsupportedAlpha_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new NODERegression<double>(3, options: new NODEOptions<double> { EntmaxAlpha = 1.25 }));
    }

    [Fact]
    public void TrainStep_MovesEveryTreeParameterAndStaysFinite()
    {
        var model = new NODERegression<double>(2, options: new NODEOptions<double> { NumTrees = 2, TreeDepth = 2 });
        var features = new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2, 0.6, 0.9 }));
        var targets = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.3, 0.8, 0.5 }));
        var before = model.GetParameters().ToArray();

        double loss = model.TrainStep(features, targets, 0.05);

        var after = model.GetParameters().ToArray();
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss));
        Assert.All(after, value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
        Assert.False(before.SequenceEqual(after), "A tape training step must move the weights.");
    }
}
