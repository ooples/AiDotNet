using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Regularization;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regularization;

/// <summary>
/// Pins the data-regularization paths of L1, L2 and ElasticNet, bit for bit and in both float and double, to the
/// element-by-element proximal definitions: soft thresholding sign(v) * max(|v| - t, 0) and the floored shrink
/// factor max(1 - amount, 0). L2 runs on the engine; bit comparison is what catches a path that loses precision
/// (a float kernel under a double T) or returns the wrong signed zero.
/// </summary>
public class RegularizationEngineParityTests
{
    private const double Strength = 0.3;
    private const double L1Ratio = 0.4;

    // Includes both signed zeros, values exactly at the threshold (which must become zero), values just either side
    // of it, values whose shrunk result is near the edge of the type's range, and non-finite inputs.
    private static readonly double[] Probe =
    {
        -2.5, -0.3, -0.12, -0.11999, -0.0, 0.0, 0.12, 0.1200001, 0.3, 0.30000001, 4.0, 1e-300,
        double.NaN, double.PositiveInfinity, double.NegativeInfinity, 1e30, 1.5e300, -1e-38,
    };

    private const int Rows = 3;
    private const int Columns = 6;

    // The references below compute in T, operation by operation, as the element-wise definitions do.
    private static double SoftThreshold(double value, double threshold)
    {
        double sign = value > 0 ? 1.0 : value < 0 ? -1.0 : 0.0;
        double reduced = Math.Abs(value) - threshold;
        return sign * (reduced > 0 ? reduced : 0.0);
    }

    private static float SoftThreshold(float value, float threshold)
    {
        float sign = value > 0 ? 1f : value < 0 ? -1f : 0f;
        float reduced = Math.Abs(value) - threshold;
        return sign * (reduced > 0 ? reduced : 0f);
    }

    private static double Shrink(double amount)
    {
        double factor = 1.0 - amount;
        return factor < 0 ? 0.0 : factor;
    }

    private static float Shrink(float amount)
    {
        float factor = 1f - amount;
        return factor < 0 ? 0f : factor;
    }

    private static readonly Func<double, double> L1Double = v => SoftThreshold(v, Strength);
    private static readonly Func<double, double> L2Double = v => v * Shrink(Strength);
    private static readonly Func<double, double> ElasticDouble =
        v => SoftThreshold(v, Strength * L1Ratio) * Shrink(Strength * (1 - L1Ratio));

    private static readonly Func<float, float> L1Float = v => SoftThreshold(v, (float)Strength);
    private static readonly Func<float, float> L2Float = v => v * Shrink((float)Strength);
    private static readonly Func<float, float> ElasticFloat =
        v => SoftThreshold(v, (float)Strength * (float)L1Ratio) * Shrink((float)Strength * (1f - (float)L1Ratio));

    private static long Bits(double value) => BitConverter.DoubleToInt64Bits(value);

    private static int Bits(float value) => BitConverter.ToInt32(BitConverter.GetBytes(value), 0);

    private static void AssertSame(double expected, double actual, string what)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(double.IsNaN(actual), $"{what}: expected NaN but was {actual:R}");
            return;
        }

        Assert.True(Bits(expected) == Bits(actual), $"{what}: expected {expected:R} but was {actual:R}");
    }

    private static void AssertSame(float expected, float actual, string what)
    {
        if (float.IsNaN(expected))
        {
            Assert.True(float.IsNaN(actual), $"{what}: expected NaN but was {actual:R}");
            return;
        }

        Assert.True(Bits(expected) == Bits(actual), $"{what}: expected {expected:R} but was {actual:R}");
    }

    private static void AssertResults<T>(Func<T, T> expected, Action<T, T, string> same, T[] probe,
        Vector<T> vector, Matrix<T> matrix, string what)
    {
        Assert.Equal(probe.Length, vector.Length);
        for (int i = 0; i < probe.Length; i++)
            same(expected(probe[i]), vector[i], $"{what} vector [{i}] (input {probe[i]})");

        Assert.Equal(Rows, matrix.Rows);
        Assert.Equal(Columns, matrix.Columns);
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < Columns; j++)
                same(expected(probe[i * Columns + j]), matrix[i, j], $"{what} matrix [{i}, {j}] (input {probe[i * Columns + j]})");
    }

    private static Matrix<T> ProbeMatrix<T>(T[] probe)
    {
        var matrix = new Matrix<T>(Rows, Columns);
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < Columns; j++)
                matrix[i, j] = probe[i * Columns + j];
        return matrix;
    }

    private static float[] FloatProbe()
    {
        var probe = new float[Probe.Length];
        for (int i = 0; i < Probe.Length; i++) probe[i] = (float)Probe[i];
        return probe;
    }

    private static RegularizationOptions Options(RegularizationType type)
        => new() { Type = type, Strength = Strength, L1Ratio = L1Ratio };

    [Fact(Timeout = 120000)]
    public async Task L1_DataRegularization_MatchesSoftThresholdPerElement()
    {
        var l1d = new L1Regularization<double, Matrix<double>, Vector<double>>(Options(RegularizationType.L1));
        AssertResults(L1Double, AssertSame, Probe, l1d.Regularize(new Vector<double>(Probe)), l1d.Regularize(ProbeMatrix(Probe)), "L1 double");

        var probe = FloatProbe();
        var l1f = new L1Regularization<float, Matrix<float>, Vector<float>>(Options(RegularizationType.L1));
        AssertResults(L1Float, AssertSame, probe, l1f.Regularize(new Vector<float>(probe)), l1f.Regularize(ProbeMatrix(probe)), "L1 float");
    }

    [Fact(Timeout = 120000)]
    public async Task L2_DataRegularization_MatchesShrinkPerElement()
    {
        var l2d = new L2Regularization<double, Matrix<double>, Vector<double>>(Options(RegularizationType.L2));
        AssertResults(L2Double, AssertSame, Probe, l2d.Regularize(new Vector<double>(Probe)), l2d.Regularize(ProbeMatrix(Probe)), "L2 double");

        var probe = FloatProbe();
        var l2f = new L2Regularization<float, Matrix<float>, Vector<float>>(Options(RegularizationType.L2));
        AssertResults(L2Float, AssertSame, probe, l2f.Regularize(new Vector<float>(probe)), l2f.Regularize(ProbeMatrix(probe)), "L2 float");
    }

    [Fact(Timeout = 120000)]
    public async Task ElasticNet_DataRegularization_MatchesThresholdThenShrinkPerElement()
    {
        var end = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(Options(RegularizationType.ElasticNet));
        AssertResults(ElasticDouble, AssertSame, Probe, end.Regularize(new Vector<double>(Probe)), end.Regularize(ProbeMatrix(Probe)), "ElasticNet double");

        var probe = FloatProbe();
        var enf = new ElasticNetRegularization<float, Matrix<float>, Vector<float>>(Options(RegularizationType.ElasticNet));
        AssertResults(ElasticFloat, AssertSame, probe, enf.Regularize(new Vector<float>(probe)), enf.Regularize(ProbeMatrix(probe)), "ElasticNet float");
    }

    [Fact(Timeout = 120000)]
    public async Task DataRegularization_OfEmptyInputs_ReturnsEmptyResults()
    {
        var options = Options(RegularizationType.ElasticNet);
        var l1 = new L1Regularization<double, Matrix<double>, Vector<double>>(options);
        var l2 = new L2Regularization<double, Matrix<double>, Vector<double>>(options);
        var elasticNet = new ElasticNetRegularization<double, Matrix<double>, Vector<double>>(options);

        Assert.Equal(0, l1.Regularize(new Vector<double>(0)).Length);
        Assert.Equal(0, l2.Regularize(new Vector<double>(0)).Length);
        Assert.Equal(0, elasticNet.Regularize(new Vector<double>(0)).Length);
        foreach (var (rows, columns) in new[] { (0, 5), (5, 0) })
        {
            var empty = new Matrix<double>(rows, columns);
            foreach (var result in new[] { l1.Regularize(empty), l2.Regularize(empty), elasticNet.Regularize(empty) })
                Assert.Equal((rows, columns), (result.Rows, result.Columns));
        }
    }
}
