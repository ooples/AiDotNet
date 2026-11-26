using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regularization;

/// <summary>
/// Pins the whole-tensor (engine) data-regularization paths of L1, L2 and ElasticNet to the
/// element-by-element proximal definitions: soft thresholding sign(v) * max(|v| - t, 0) and the
/// floored shrink factor max(1 - amount, 0).
/// </summary>
public class RegularizationEngineParityTests
{
    private const double Strength = 0.3;
    private const double L1Ratio = 0.4;

    // Includes both signed zeros, values exactly at the threshold (which must become zero), values just
    // either side of it, and non-finite inputs.
    private static readonly double[] Probe =
    {
        -2.5, -0.3, -0.12, -0.11999, -0.0, 0.0, 0.12, 0.1200001, 0.3, 0.30000001, 4.0, 1e-300,
        double.NaN, double.PositiveInfinity, double.NegativeInfinity,
    };

    private static double SoftThreshold(double value, double threshold)
    {
        double sign = value > 0 ? 1.0 : value < 0 ? -1.0 : 0.0;
        double reduced = Math.Abs(value) - threshold;
        return sign * (reduced > 0 ? reduced : 0.0);
    }

    private static double Shrink(double amount)
    {
        double factor = 1.0 - amount;
        return factor < 0 ? 0.0 : factor;
    }

    private static double ExpectedL1(double value) => SoftThreshold(value, Strength);

    private static double ExpectedL2(double value) => value * Shrink(Strength);

    private static double ExpectedElasticNet(double value)
        => SoftThreshold(value, Strength * L1Ratio) * Shrink(Strength * (1 - L1Ratio));

    private static void AssertSame(double expected, double actual, string what)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"{what}: expected NaN but was {actual:R}");
            return;
        }

        Assert.True(expected.Equals(actual) || Math.Abs(expected - actual) <= 1e-15,
            $"{what}: expected {expected:R} but was {actual:R}");
    }

    private static void AssertVector(Func<double, double> expected, Vector<double> actual, string what)
    {
        Assert.Equal(Probe.Length, actual.Length);
        for (int i = 0; i < Probe.Length; i++)
        {
            AssertSame(expected(Probe[i]), actual[i], $"{what} vector [{i}] (input {Probe[i]:R})");
        }
    }

    private static void AssertMatrix(Func<double, double> expected, Matrix<double> actual, string what)
    {
        Assert.Equal(3, actual.Rows);
        Assert.Equal(5, actual.Columns);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                AssertSame(expected(Probe[i * 5 + j]), actual[i, j], $"{what} matrix [{i}, {j}] (input {Probe[i * 5 + j]:R})");
            }
        }
    }

    private static Matrix<double> ProbeMatrix()
    {
        var matrix = new Matrix<double>(3, 5);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                matrix[i, j] = Probe[i * 5 + j];
            }
        }

        return matrix;
    }

    [Fact(Timeout = 120000)]
    public async Task L1_DataRegularization_MatchesSoftThresholdPerElement()
    {
        var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L1, Strength = Strength });

        AssertVector(ExpectedL1, l1.Regularize(new Vector<double>(Probe)), "L1");
        AssertMatrix(ExpectedL1, l1.Regularize(ProbeMatrix()), "L1");
    }

    [Fact(Timeout = 120000)]
    public async Task L2_DataRegularization_MatchesShrinkPerElement()
    {
        var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.L2, Strength = Strength });

        AssertVector(ExpectedL2, l2.Regularize(new Vector<double>(Probe)), "L2");
        AssertMatrix(ExpectedL2, l2.Regularize(ProbeMatrix()), "L2");
    }

    [Fact(Timeout = 120000)]
    public async Task ElasticNet_DataRegularization_MatchesThresholdThenShrinkPerElement()
    {
        var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(
            new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = Strength, L1Ratio = L1Ratio });

        AssertVector(ExpectedElasticNet, elasticNet.Regularize(new Vector<double>(Probe)), "ElasticNet");
        AssertMatrix(ExpectedElasticNet, elasticNet.Regularize(ProbeMatrix()), "ElasticNet");
    }

    [Fact(Timeout = 120000)]
    public async Task DataRegularization_OfEmptyInputs_ReturnsEmptyResults()
    {
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = Strength, L1Ratio = L1Ratio };
        var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
        var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
        var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);

        Assert.Equal(0, l1.Regularize(new Vector<double>(0)).Length);
        Assert.Equal(0, l2.Regularize(new Vector<double>(0)).Length);
        Assert.Equal(0, elasticNet.Regularize(new Vector<double>(0)).Length);
    }
}
