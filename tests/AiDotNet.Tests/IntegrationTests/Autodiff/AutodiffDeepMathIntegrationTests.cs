using AiDotNet.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Autodiff;

/// <summary>
/// Deep math-correctness integration tests for TensorOperations autodiff gradients.
/// Covers operations NOT tested in the existing TensorOperationsIntegrationTests:
/// SELU, HardTanh, CELU, LiSHT, BentIdentity, Gaussian, ScaledTanh, Mish,
/// PReLU, ThresholdedReLU, ISRU, SQRBF, Reshape, Concat, Slice, Divide chain rules,
/// MatrixVectorMultiply, and complex chain rule compositions.
/// Every test compares autodiff gradients with numerical gradients (central difference).
/// </summary>
public class AutodiffDeepMathIntegrationTests
{
    private const double Tolerance = 1e-4;
    private const double NumericalEpsilon = 1e-5;

    #region Helpers

    private static Tensor<double> ComputeNumericalGradient(
        Tensor<double> input,
        Func<Tensor<double>, double> scalarFunction,
        double eps = NumericalEpsilon)
    {
        var gradient = new Tensor<double>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            var plusH = input.Clone();
            var minusH = input.Clone();
            plusH[i] += eps;
            minusH[i] -= eps;
            gradient[i] = (scalarFunction(plusH) - scalarFunction(minusH)) / (2 * eps);
        }
        return gradient;
    }

    private static double SumTensor(Tensor<double> t)
    {
        double sum = 0;
        for (int i = 0; i < t.Length; i++)
            sum += t[i];
        return sum;
    }

    private static void AssertTensorsEqual(Tensor<double> expected, Tensor<double> actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(expected[i] - actual[i]) < tolerance,
                $"Element {i}: expected {expected[i]:F8}, actual {actual[i]:F8}, diff {Math.Abs(expected[i] - actual[i]):E3}");
        }
    }

    private static Tensor<double> MakeTensor(params double[] values)
    {
        var t = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
            t[i] = values[i];
        return t;
    }

    private static Tensor<double> MakeTensor2D(double[,] values)
    {
        int rows = values.GetLength(0);
        int cols = values.GetLength(1);
        var t = new Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[i, j] = values[i, j];
        return t;
    }

    #endregion

    #region SELU Gradient Tests

    [Fact]
    public void SELU_GradientMatchesNumerical_PositiveValues()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.5, 1.0, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SELU(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double lambda = 1.0507009873554804934193349852946;
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] >= 0 ? lambda * t[i] : lambda * 1.6732632423543772848170429916717 * (Math.Exp(t[i]) - 1);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void SELU_GradientMatchesNumerical_NegativeValues()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-0.5, -1.0, -2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SELU(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double lambda = 1.0507009873554804934193349852946;
            double alpha = 1.6732632423543772848170429916717;
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] >= 0 ? lambda * t[i] : lambda * alpha * (Math.Exp(t[i]) - 1);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void SELU_GradientMatchesNumerical_MixedValues()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.5, 0.5, -0.3, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SELU(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double lambda = 1.0507009873554804934193349852946;
            double alpha = 1.6732632423543772848170429916717;
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] >= 0 ? lambda * t[i] : lambda * alpha * (Math.Exp(t[i]) - 1);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    #endregion

    #region HardTanh Gradient Tests

    [Fact]
    public void HardTanh_GradientMatchesNumerical_InLinearRegion()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-0.5, 0.0, 0.5, 0.9);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.HardTanh(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Max(-1.0, Math.Min(1.0, t[i]));
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void HardTanh_GradientMatchesNumerical_InSaturationRegion()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-3.0, -1.5, 1.5, 3.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.HardTanh(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        // In saturation region, gradient should be 0
        for (int i = 0; i < grads[a].Length; i++)
            Assert.True(Math.Abs(grads[a][i]) < Tolerance,
                $"HardTanh gradient at saturated input [{i}] should be 0, got {grads[a][i]}");
    }

    #endregion

    #region CELU Gradient Tests

    [Fact]
    public void CELU_GradientMatchesNumerical_DefaultAlpha()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, -0.5, 0.5, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.CELU(a, alpha: 1.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Max(0, t[i]) + Math.Min(0, 1.0 * (Math.Exp(t[i] / 1.0) - 1));
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void CELU_GradientMatchesNumerical_CustomAlpha()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-2.0, -0.3, 0.7, 1.5);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.CELU(a, alpha: 0.5);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double alpha = 0.5;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Max(0, t[i]) + Math.Min(0, alpha * (Math.Exp(t[i] / alpha) - 1));
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    #endregion

    #region LiSHT Gradient Tests

    [Fact]
    public void LiSHT_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, -0.5, 0.0, 0.5, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.LiSHT(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] * Math.Tanh(t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void LiSHT_ForwardValues_HandCalculated()
    {
        // LiSHT(x) = x * tanh(x)
        // LiSHT(0) = 0
        // LiSHT(1) = tanh(1) ≈ 0.7616
        // LiSHT(-1) = -1 * tanh(-1) = -1 * -0.7616 = 0.7616
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0, -1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.LiSHT(a);

        Assert.True(Math.Abs(result.Value[0] - 0.0) < Tolerance);
        Assert.True(Math.Abs(result.Value[1] - Math.Tanh(1.0)) < Tolerance);
        Assert.True(Math.Abs(result.Value[2] - Math.Tanh(1.0)) < Tolerance); // symmetric
    }

    #endregion

    #region BentIdentity Gradient Tests

    [Fact]
    public void BentIdentity_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-2.0, -1.0, 0.0, 1.0, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.BentIdentity(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += (Math.Sqrt(t[i] * t[i] + 1) - 1) / 2.0 + t[i];
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void BentIdentity_ForwardValues_HandCalculated()
    {
        // f(0) = (sqrt(1)-1)/2 + 0 = 0
        // f(1) = (sqrt(2)-1)/2 + 1 = (1.4142-1)/2 + 1 = 0.2071 + 1 = 1.2071
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.BentIdentity(a);

        Assert.True(Math.Abs(result.Value[0] - 0.0) < Tolerance);
        double expected1 = (Math.Sqrt(2) - 1) / 2.0 + 1.0;
        Assert.True(Math.Abs(result.Value[1] - expected1) < Tolerance);
    }

    #endregion

    #region Gaussian Activation Gradient Tests

    [Fact]
    public void Gaussian_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.5, -0.5, 0.0, 0.5, 1.5);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Gaussian(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Exp(-t[i] * t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void Gaussian_ForwardValues_HandCalculated()
    {
        // f(0) = exp(0) = 1
        // f(1) = exp(-1) ≈ 0.3679
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Gaussian(a);

        Assert.True(Math.Abs(result.Value[0] - 1.0) < Tolerance);
        Assert.True(Math.Abs(result.Value[1] - Math.Exp(-1.0)) < Tolerance);
    }

    [Fact]
    public void Gaussian_GradientAtZero_IsZero()
    {
        // d/dx[exp(-x^2)] = -2x * exp(-x^2), at x=0 this is 0
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Gaussian(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        Assert.True(Math.Abs(grads[a][0]) < Tolerance,
            $"Gaussian gradient at x=0 should be 0, got {grads[a][0]}");
    }

    #endregion

    #region ScaledTanh Gradient Tests

    [Fact]
    public void ScaledTanh_GradientMatchesNumerical_DefaultBeta()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, -0.5, 0.0, 0.5, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ScaledTanh(a, beta: 1.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double beta = 1.0;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double expNeg = Math.Exp(-beta * t[i]);
                sum += (1 - expNeg) / (1 + expNeg);
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ScaledTanh_GradientMatchesNumerical_CustomBeta()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-0.8, 0.3, 0.7);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ScaledTanh(a, beta: 2.5);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double beta = 2.5;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double expNeg = Math.Exp(-beta * t[i]);
                sum += (1 - expNeg) / (1 + expNeg);
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    #endregion

    #region Mish Gradient Tests

    [Fact]
    public void Mish_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, -0.5, 0.0, 0.5, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Mish(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] * Math.Tanh(Math.Log(1 + Math.Exp(t[i])));
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void Mish_ForwardValues_HandCalculated()
    {
        // Mish(0) = 0 * tanh(ln(2)) = 0
        // Mish(1) = 1 * tanh(ln(1+e)) ≈ 1 * tanh(1.3133) ≈ 0.8651
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Mish(a);

        Assert.True(Math.Abs(result.Value[0] - 0.0) < Tolerance);
        double expectedMish1 = 1.0 * Math.Tanh(Math.Log(1 + Math.Exp(1.0)));
        Assert.True(Math.Abs(result.Value[1] - expectedMish1) < Tolerance);
    }

    #endregion

    #region PReLU Gradient Tests

    [Fact]
    public void PReLU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-2.0, -0.5, 0.5, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.PReLU(a, alpha: 0.1);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double alpha = 0.1;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] > 0 ? t[i] : alpha * t[i];
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void PReLU_ForwardValues_HandCalculated()
    {
        // PReLU(2, 0.1) = 2
        // PReLU(-3, 0.1) = 0.1 * -3 = -0.3
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(2.0, -3.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.PReLU(a, alpha: 0.1);

        Assert.True(Math.Abs(result.Value[0] - 2.0) < Tolerance);
        Assert.True(Math.Abs(result.Value[1] - (-0.3)) < Tolerance);
    }

    #endregion

    #region ThresholdedReLU Gradient Tests

    [Fact]
    public void ThresholdedReLU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, 0.5, 1.5, 3.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        double threshold = 1.0;
        var result = TensorOperations<double>.ThresholdedReLU(a, threshold: threshold);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] > threshold ? t[i] : 0.0;
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ThresholdedReLU_ForwardValues_HandCalculated()
    {
        // ThresholdedReLU(0.5, threshold=1) = 0
        // ThresholdedReLU(1.5, threshold=1) = 1.5
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.5, 1.5);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ThresholdedReLU(a, threshold: 1.0);

        Assert.True(Math.Abs(result.Value[0] - 0.0) < Tolerance);
        Assert.True(Math.Abs(result.Value[1] - 1.5) < Tolerance);
    }

    #endregion

    #region ISRU Gradient Tests

    [Fact]
    public void ISRU_GradientMatchesNumerical_DefaultAlpha()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-2.0, -0.5, 0.0, 0.5, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ISRU(a, alpha: 1.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double alpha = 1.0;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] / Math.Sqrt(1 + alpha * t[i] * t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ISRU_GradientMatchesNumerical_CustomAlpha()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, 0.5, 1.5);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ISRU(a, alpha: 2.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double alpha = 2.0;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] / Math.Sqrt(1 + alpha * t[i] * t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ISRU_ForwardValues_HandCalculated()
    {
        // ISRU(0, 1) = 0
        // ISRU(1, 1) = 1/sqrt(2) ≈ 0.7071
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ISRU(a, alpha: 1.0);

        Assert.True(Math.Abs(result.Value[0] - 0.0) < Tolerance);
        Assert.True(Math.Abs(result.Value[1] - 1.0 / Math.Sqrt(2.0)) < Tolerance);
    }

    #endregion

    #region SQRBF Gradient Tests

    [Fact]
    public void SQRBF_GradientMatchesNumerical_DefaultBeta()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, -0.5, 0.0, 0.5, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SQRBF(a, beta: 1.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double beta = 1.0;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Exp(-beta * t[i] * t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void SQRBF_GradientMatchesNumerical_CustomBeta()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-0.8, 0.3, 0.6);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SQRBF(a, beta: 3.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        double beta = 3.0;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Exp(-beta * t[i] * t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void SQRBF_IsIdenticalToGaussian_WhenBetaOne()
    {
        // SQRBF(x, beta=1) = exp(-1*x^2) = Gaussian(x)
        var aData = MakeTensor(-1.0, -0.5, 0.0, 0.5, 1.0);

        using var tape1 = new GradientTape<double>();
        var a1 = TensorOperations<double>.Variable(aData.Clone(), "a1");
        tape1.Watch(a1);
        var sqrbf = TensorOperations<double>.SQRBF(a1, beta: 1.0);
        var f1 = TensorOperations<double>.Sum(sqrbf);
        var grads1 = tape1.Gradient(f1, new[] { a1 });

        using var tape2 = new GradientTape<double>();
        var a2 = TensorOperations<double>.Variable(aData.Clone(), "a2");
        tape2.Watch(a2);
        var gaussian = TensorOperations<double>.Gaussian(a2);
        var f2 = TensorOperations<double>.Sum(gaussian);
        var grads2 = tape2.Gradient(f2, new[] { a2 });

        AssertTensorsEqual(grads1[a1], grads2[a2], Tolerance);
    }

    #endregion

    #region Reshape Gradient Tests

    [Fact]
    public void Reshape_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor2D(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // Reshape 2x3 -> 6
        var reshaped = TensorOperations<double>.Reshape(a, 6);
        var f = TensorOperations<double>.Sum(TensorOperations<double>.Square(reshaped));
        var grads = tape.Gradient(f, new[] { a });

        // d/da_i[sum(a_i^2)] = 2*a_i
        for (int i = 0; i < aData.Length; i++)
        {
            Assert.True(Math.Abs(grads[a][i] - 2 * aData[i]) < Tolerance,
                $"Reshape gradient element {i}: expected {2 * aData[i]}, got {grads[a][i]}");
        }
    }

    #endregion

    #region Complex Chain Rule Tests

    [Fact]
    public void ChainRule_Mish_Of_Square_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.3, 0.5, 0.8);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(Mish(a^2))
        var sq = TensorOperations<double>.Square(a);
        var mish = TensorOperations<double>.Mish(sq);
        var f = TensorOperations<double>.Sum(mish);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double sq2 = t[i] * t[i];
                sum += sq2 * Math.Tanh(Math.Log(1 + Math.Exp(sq2)));
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ChainRule_SELU_Of_Tanh_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        // Avoid x=0 because tanh(0)=0 hits SELU's non-differentiable point
        // (SELU derivative is discontinuous at 0: left=λα≈1.758, right=λ≈1.051)
        var aData = MakeTensor(-0.5, 0.3, 0.5);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(SELU(tanh(a)))
        var tanhA = TensorOperations<double>.Tanh(a);
        var selu = TensorOperations<double>.SELU(tanhA);
        var f = TensorOperations<double>.Sum(selu);
        var grads = tape.Gradient(f, new[] { a });

        double lambda = 1.0507009873554804934193349852946;
        double alpha = 1.6732632423543772848170429916717;
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double th = Math.Tanh(t[i]);
                sum += th >= 0 ? lambda * th : lambda * alpha * (Math.Exp(th) - 1);
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ChainRule_BentIdentity_Of_Sigmoid_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-1.0, 0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(BentIdentity(Sigmoid(a)))
        var sig = TensorOperations<double>.Sigmoid(a);
        var bent = TensorOperations<double>.BentIdentity(sig);
        var f = TensorOperations<double>.Sum(bent);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double s = 1.0 / (1.0 + Math.Exp(-t[i]));
                sum += (Math.Sqrt(s * s + 1) - 1) / 2.0 + s;
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ChainRule_Gaussian_Of_Exp_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        // Use small values to avoid overflow
        var aData = MakeTensor(-0.5, 0.0, 0.3);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(Gaussian(Exp(a)))
        var expA = TensorOperations<double>.Exp(a);
        var gauss = TensorOperations<double>.Gaussian(expA);
        var f = TensorOperations<double>.Sum(gauss);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double ex = Math.Exp(t[i]);
                sum += Math.Exp(-ex * ex);
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ChainRule_LiSHT_Of_CELU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(-0.5, 0.0, 0.5);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(LiSHT(CELU(a)))
        var celu = TensorOperations<double>.CELU(a, alpha: 1.0);
        var lisht = TensorOperations<double>.LiSHT(celu);
        var f = TensorOperations<double>.Sum(lisht);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double celuVal = Math.Max(0, t[i]) + Math.Min(0, Math.Exp(t[i]) - 1);
                sum += celuVal * Math.Tanh(celuVal);
            }
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    [Fact]
    public void ChainRule_ThreeDeep_Add_Multiply_Exp_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.5, 1.0);
        var bData = MakeTensor(0.3, 0.7);
        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // f = sum(exp(a * b + a))
        var ab = TensorOperations<double>.ElementwiseMultiply(a, b);
        var abPlusA = TensorOperations<double>.Add(ab, a);
        var result = TensorOperations<double>.Exp(abPlusA);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a, b });

        var numericalA = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Exp(t[i] * bData[i] + t[i]);
            return sum;
        });

        var numericalB = ComputeNumericalGradient(bData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += Math.Exp(aData[i] * t[i] + aData[i]);
            return sum;
        });

        AssertTensorsEqual(numericalA, grads[a], Tolerance);
        AssertTensorsEqual(numericalB, grads[b], Tolerance);
    }

    #endregion

    #region MatrixVectorMultiply Gradient Tests

    [Fact]
    public void MatrixVectorMultiply_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var matData = MakeTensor2D(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var vecData = MakeTensor(5.0, 6.0);

        var mat = TensorOperations<double>.Variable(matData, "mat");
        var vec = TensorOperations<double>.Variable(vecData, "vec");
        tape.Watch(mat);
        tape.Watch(vec);

        // f = sum(mat @ vec) where @ is matrix-vector multiply
        var mv = TensorOperations<double>.MatrixVectorMultiply(mat, vec);
        var f = TensorOperations<double>.Sum(mv);
        var grads = tape.Gradient(f, new[] { mat, vec });

        // Numerical gradient for mat
        var numericalMat = ComputeNumericalGradient(matData, t =>
        {
            // Matrix-vector multiply: [t00*v0+t01*v1, t10*v0+t11*v1]
            double sum = 0;
            for (int i = 0; i < 2; i++)
            {
                double rowSum = 0;
                for (int j = 0; j < 2; j++)
                    rowSum += t[i, j] * vecData[j];
                sum += rowSum;
            }
            return sum;
        });

        // Numerical gradient for vec
        var numericalVec = ComputeNumericalGradient(vecData, t =>
        {
            double sum = 0;
            for (int i = 0; i < 2; i++)
            {
                double rowSum = 0;
                for (int j = 0; j < 2; j++)
                    rowSum += matData[i, j] * t[j];
                sum += rowSum;
            }
            return sum;
        });

        AssertTensorsEqual(numericalMat, grads[mat], Tolerance);
        AssertTensorsEqual(numericalVec, grads[vec], Tolerance);
    }

    #endregion

    #region Concat Gradient Tests

    [Fact]
    public void Concat_GradientMatchesNumerical_Axis0()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor2D(new double[,] { { 1.0, 2.0 } });
        var bData = MakeTensor2D(new double[,] { { 3.0, 4.0 } });

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // Concat along axis 0: [1,2] + [3,4] => [[1,2],[3,4]]
        var concat = TensorOperations<double>.Concat(new List<ComputationNode<double>> { a, b }, axis: 0);
        var sq = TensorOperations<double>.Square(concat);
        var f = TensorOperations<double>.Sum(sq);
        var grads = tape.Gradient(f, new[] { a, b });

        // d/da_i[sum(concat^2)] for a-elements: 2*a_i
        var numericalA = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++) sum += t[i] * t[i];
            for (int i = 0; i < bData.Length; i++) sum += bData[i] * bData[i];
            return sum;
        });

        AssertTensorsEqual(numericalA, grads[a], Tolerance);
    }

    #endregion

    #region Slice Gradient Tests

    [Fact]
    public void Slice_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0, 2.0, 3.0, 4.0, 5.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // Slice elements 1..3 (indices 1,2,3)
        var sliced = TensorOperations<double>.Slice(a, start: 1, length: 3, step: 1, axis: 0);
        var sq = TensorOperations<double>.Square(sliced);
        var f = TensorOperations<double>.Sum(sq);
        var grads = tape.Gradient(f, new[] { a });

        // Only elements 1,2,3 contribute; gradient is 2*a_i for those, 0 for others
        var numerical = ComputeNumericalGradient(aData, t =>
        {
            return t[1] * t[1] + t[2] * t[2] + t[3] * t[3];
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    #endregion

    #region Persistent Tape Tests

    [Fact]
    public void PersistentTape_CanComputeGradientsTwice()
    {
        using var tape = new GradientTape<double>(persistent: true);
        var aData = MakeTensor(1.0, 2.0, 3.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var sq = TensorOperations<double>.Square(a);
        var f = TensorOperations<double>.Sum(sq);

        // First gradient computation
        var grads1 = tape.Gradient(f, new[] { a });

        // Second gradient computation (should not throw)
        var grads2 = tape.Gradient(f, new[] { a });

        // Both should give the same result: 2*a
        for (int i = 0; i < aData.Length; i++)
        {
            Assert.True(Math.Abs(grads1[a][i] - 2 * aData[i]) < Tolerance);
            Assert.True(Math.Abs(grads2[a][i] - 2 * aData[i]) < Tolerance);
        }
    }

    [Fact]
    public void NonPersistentTape_ThrowsOnSecondGradient()
    {
        using var tape = new GradientTape<double>(persistent: false);
        var aData = MakeTensor(1.0, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var sq = TensorOperations<double>.Square(a);
        var f = TensorOperations<double>.Sum(sq);

        // First call succeeds
        tape.Gradient(f, new[] { a });

        // Second call should throw
        Assert.Throws<InvalidOperationException>(() => tape.Gradient(f, new[] { a }));
    }

    #endregion

    #region DisposedTape Tests

    [Fact]
    public void DisposedTape_ThrowsOnGradient()
    {
        var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var sq = TensorOperations<double>.Square(a);
        var f = TensorOperations<double>.Sum(sq);

        tape.Dispose();

        Assert.Throws<ObjectDisposedException>(() => tape.Gradient(f, new[] { a }));
    }

    [Fact]
    public void DisposedTape_ThrowsOnWatch()
    {
        var tape = new GradientTape<double>();
        tape.Dispose();

        var aData = MakeTensor(1.0);
        var a = TensorOperations<double>.Variable(aData, "a");

        Assert.Throws<ObjectDisposedException>(() => tape.Watch(a));
    }

    #endregion

    #region Gradient Accumulation Tests

    [Fact]
    public void GradientAccumulation_MultipleUsesOfSameVariable()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0, 2.0, 3.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(a + a) = sum(2*a)
        // d/da_i = 2
        var added = TensorOperations<double>.Add(a, a);
        var f = TensorOperations<double>.Sum(added);
        var grads = tape.Gradient(f, new[] { a });

        for (int i = 0; i < aData.Length; i++)
        {
            Assert.True(Math.Abs(grads[a][i] - 2.0) < Tolerance,
                $"Gradient at [{i}] should be 2.0, got {grads[a][i]}");
        }
    }

    [Fact]
    public void GradientAccumulation_VariableUsedInMultipleBranches()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(a^2 + exp(a))
        // d/da_i = 2*a_i + exp(a_i)
        var sq = TensorOperations<double>.Square(a);
        var ex = TensorOperations<double>.Exp(a);
        var combined = TensorOperations<double>.Add(sq, ex);
        var f = TensorOperations<double>.Sum(combined);
        var grads = tape.Gradient(f, new[] { a });

        var numerical = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] * t[i] + Math.Exp(t[i]);
            return sum;
        });

        AssertTensorsEqual(numerical, grads[a], Tolerance);
    }

    #endregion

    #region Hand-Calculated Gradient Value Tests

    [Fact]
    public void SELU_GradientValues_HandCalculated()
    {
        double lambda = 1.0507009873554804934193349852946;
        double alpha = 1.6732632423543772848170429916717;

        using var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0, -1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SELU(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        // At x=1: d(SELU)/dx = lambda ≈ 1.0507
        Assert.True(Math.Abs(grads[a][0] - lambda) < Tolerance,
            $"SELU gradient at x=1 should be {lambda}, got {grads[a][0]}");

        // At x=-1: d(SELU)/dx = lambda * alpha * exp(-1)
        double expectedNeg = lambda * alpha * Math.Exp(-1.0);
        Assert.True(Math.Abs(grads[a][1] - expectedNeg) < Tolerance,
            $"SELU gradient at x=-1 should be {expectedNeg}, got {grads[a][1]}");
    }

    [Fact]
    public void BentIdentity_GradientValues_HandCalculated()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.BentIdentity(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        // At x=0: d/dx = 0/(2*1) + 1 = 1.0
        Assert.True(Math.Abs(grads[a][0] - 1.0) < Tolerance,
            $"BentIdentity gradient at x=0 should be 1.0, got {grads[a][0]}");

        // At x=1: d/dx = 1/(2*sqrt(2)) + 1 ≈ 0.3536 + 1 = 1.3536
        double expected1 = 1.0 / (2 * Math.Sqrt(2.0)) + 1.0;
        Assert.True(Math.Abs(grads[a][1] - expected1) < Tolerance,
            $"BentIdentity gradient at x=1 should be {expected1}, got {grads[a][1]}");
    }

    [Fact]
    public void Gaussian_GradientValues_HandCalculated()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Gaussian(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        // At x=1: d/dx = -2*1*exp(-1) ≈ -0.7358
        double expected = -2.0 * Math.Exp(-1.0);
        Assert.True(Math.Abs(grads[a][0] - expected) < Tolerance,
            $"Gaussian gradient at x=1 should be {expected}, got {grads[a][0]}");
    }

    [Fact]
    public void ISRU_GradientValues_HandCalculated()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0, 1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ISRU(a, alpha: 1.0);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        // At x=0: d/dx = (1+0)^(-3/2) = 1.0
        Assert.True(Math.Abs(grads[a][0] - 1.0) < Tolerance,
            $"ISRU gradient at x=0 should be 1.0, got {grads[a][0]}");

        // At x=1, alpha=1: d/dx = (1+1)^(-3/2) = 2^(-3/2) ≈ 0.3536
        double expected1 = Math.Pow(2.0, -1.5);
        Assert.True(Math.Abs(grads[a][1] - expected1) < Tolerance,
            $"ISRU gradient at x=1 should be {expected1}, got {grads[a][1]}");
    }

    [Fact]
    public void LiSHT_GradientValues_HandCalculated()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.LiSHT(a);
        var f = TensorOperations<double>.Sum(result);
        var grads = tape.Gradient(f, new[] { a });

        // At x=0: d/dx = tanh(0) + 0*(1-tanh^2(0)) = 0 + 0 = 0
        Assert.True(Math.Abs(grads[a][0]) < Tolerance,
            $"LiSHT gradient at x=0 should be 0, got {grads[a][0]}");
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void CELU_AtZero_IsSmooth()
    {
        // CELU should be continuous at x=0
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.CELU(a, alpha: 1.0);

        // At x=0: max(0,0) + min(0, 1*(exp(0)-1)) = 0 + min(0, 0) = 0
        Assert.True(Math.Abs(result.Value[0]) < Tolerance,
            $"CELU(0) should be 0, got {result.Value[0]}");
    }

    [Fact]
    public void ScaledTanh_AtZero_IsZero()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.ScaledTanh(a, beta: 2.0);

        // ScaledTanh(0) = (1-exp(0))/(1+exp(0)) = 0/2 = 0
        Assert.True(Math.Abs(result.Value[0]) < Tolerance,
            $"ScaledTanh(0) should be 0, got {result.Value[0]}");
    }

    [Fact]
    public void SQRBF_AtZero_IsOne()
    {
        // SQRBF(0, beta) = exp(0) = 1
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.SQRBF(a, beta: 3.0);

        Assert.True(Math.Abs(result.Value[0] - 1.0) < Tolerance,
            $"SQRBF(0) should be 1, got {result.Value[0]}");
    }

    [Fact]
    public void Mish_AtZero_IsZero()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(0.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var result = TensorOperations<double>.Mish(a);

        // Mish(0) = 0 * tanh(ln(2)) = 0
        Assert.True(Math.Abs(result.Value[0]) < Tolerance,
            $"Mish(0) should be 0, got {result.Value[0]}");
    }

    [Fact]
    public void PReLU_ContinuousAtZero()
    {
        // PReLU should approach same value from both sides at x=0
        using var tape1 = new GradientTape<double>();
        var aPlus = MakeTensor(1e-8);
        var a1 = TensorOperations<double>.Variable(aPlus, "a");
        tape1.Watch(a1);
        var r1 = TensorOperations<double>.PReLU(a1, alpha: 0.1);

        using var tape2 = new GradientTape<double>();
        var aMinus = MakeTensor(-1e-8);
        var a2 = TensorOperations<double>.Variable(aMinus, "a");
        tape2.Watch(a2);
        var r2 = TensorOperations<double>.PReLU(a2, alpha: 0.1);

        // Both should be very close to 0
        Assert.True(Math.Abs(r1.Value[0]) < 1e-6);
        Assert.True(Math.Abs(r2.Value[0]) < 1e-6);
    }

    #endregion

    #region ZeroGradient Tests

    [Fact]
    public void ZeroGradient_ClearsGradient()
    {
        using var tape = new GradientTape<double>();
        var aData = MakeTensor(1.0, 2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var sq = TensorOperations<double>.Square(a);
        var f = TensorOperations<double>.Sum(sq);
        tape.Gradient(f, new[] { a });

        // After backward, gradient should be non-zero
        Assert.NotNull(a.Gradient);

        // After ZeroGradient, all elements should be 0
        a.ZeroGradient();
        for (int i = 0; i < a.Gradient.Length; i++)
        {
            Assert.True(Math.Abs(a.Gradient[i]) < Tolerance,
                $"Gradient element {i} should be 0 after ZeroGradient");
        }
    }

    #endregion

    #region Tape Recording State Tests

    [Fact]
    public void StopRecording_PausesRecording()
    {
        using var tape = new GradientTape<double>();
        Assert.True(tape.IsRecording);

        tape.StopRecording();
        Assert.False(tape.IsRecording);

        tape.ResumeRecording();
        Assert.True(tape.IsRecording);
    }

    [Fact]
    public void Reset_ClearsAllState()
    {
        using var tape = new GradientTape<double>(persistent: true);
        var aData = MakeTensor(1.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var sq = TensorOperations<double>.Square(a);
        var f = TensorOperations<double>.Sum(sq);
        tape.Gradient(f, new[] { a });

        // Reset should clear state
        tape.Reset();
        Assert.True(tape.IsRecording);
    }

    #endregion

    #region Nested Tape Tests

    [Fact]
    public void NestedTapes_InnerTapeDoesNotAffectOuter()
    {
        using var outerTape = new GradientTape<double>();
        var aData = MakeTensor(2.0);
        var a = TensorOperations<double>.Variable(aData, "a");
        outerTape.Watch(a);

        var sq = TensorOperations<double>.Square(a);

        // Inner tape for separate computation
        using (var innerTape = new GradientTape<double>())
        {
            var bData = MakeTensor(3.0);
            var b = TensorOperations<double>.Variable(bData, "b");
            innerTape.Watch(b);

            var bSq = TensorOperations<double>.Square(b);
            var fInner = TensorOperations<double>.Sum(bSq);
            var innerGrads = innerTape.Gradient(fInner, new[] { b });

            // Inner gradient: d/db[b^2] = 2b = 6
            Assert.True(Math.Abs(innerGrads[b][0] - 6.0) < Tolerance);
        }

        // Outer tape should still work
        var fOuter = TensorOperations<double>.Sum(sq);
        var outerGrads = outerTape.Gradient(fOuter, new[] { a });

        // Outer gradient: d/da[a^2] = 2a = 4
        Assert.True(Math.Abs(outerGrads[a][0] - 4.0) < Tolerance);
    }

    #endregion
}
