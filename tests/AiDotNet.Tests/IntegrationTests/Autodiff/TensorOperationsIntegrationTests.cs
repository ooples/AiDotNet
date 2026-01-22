using AiDotNet.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Autodiff;

/// <summary>
/// Comprehensive integration tests for TensorOperations automatic differentiation.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify the mathematical correctness of gradient computations for
/// TensorOperations by comparing autodiff gradients with numerical gradients.
/// </para>
/// <para><b>For Beginners:</b> These tests ensure that when we compute gradients
/// automatically (autodiff), they match what we'd get from the mathematical definition
/// of derivatives (numerical gradients). If a test fails, it means our gradient
/// implementation has a bug.
/// </para>
/// </remarks>
public class TensorOperationsIntegrationTests
{
    private const double Tolerance = 1e-4;
    private const double NumericalEpsilon = 1e-5;
    private const float FloatTolerance = 1e-3f;

    #region Helper Methods

    /// <summary>
    /// Computes numerical gradient using central difference.
    /// </summary>
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

    /// <summary>
    /// Computes the sum of all elements in a tensor.
    /// </summary>
    private static double SumTensor(Tensor<double> t)
    {
        double sum = 0;
        for (int i = 0; i < t.Length; i++)
            sum += t[i];
        return sum;
    }

    /// <summary>
    /// Compares two tensors element-wise with tolerance.
    /// </summary>
    private static void AssertTensorsEqual(Tensor<double> expected, Tensor<double> actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(
                Math.Abs(expected[i] - actual[i]) < tolerance,
                $"Element {i}: expected {expected[i]}, actual {actual[i]}, diff {Math.Abs(expected[i] - actual[i])}");
        }
    }

    #endregion

    #region Basic Arithmetic Operations

    [Fact]
    public void Add_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        // Create input tensors
        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 1.0; aData[1] = 2.0; aData[2] = 3.0;

        var bData = new Tensor<double>(new[] { 3 });
        bData[0] = 4.0; bData[1] = 5.0; bData[2] = 6.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // f = sum(a + b)
        var c = TensorOperations<double>.Add(a, b);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        // df/da[i] = 1, df/db[i] = 1
        Assert.NotNull(gradA);
        Assert.NotNull(gradB);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0, gradA[i], Tolerance);
            Assert.Equal(1.0, gradB[i], Tolerance);
        }
    }

    [Fact]
    public void Subtract_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 5.0; aData[1] = 7.0; aData[2] = 9.0;

        var bData = new Tensor<double>(new[] { 3 });
        bData[0] = 2.0; bData[1] = 3.0; bData[2] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // f = sum(a - b)
        var c = TensorOperations<double>.Subtract(a, b);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        // df/da[i] = 1, df/db[i] = -1
        Assert.NotNull(gradA);
        Assert.NotNull(gradB);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0, gradA[i], Tolerance);
            Assert.Equal(-1.0, gradB[i], Tolerance);
        }
    }

    [Fact]
    public void ElementwiseMultiply_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 2.0; aData[1] = 3.0; aData[2] = 4.0;

        var bData = new Tensor<double>(new[] { 3 });
        bData[0] = 5.0; bData[1] = 6.0; bData[2] = 7.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // f = sum(a * b)
        var c = TensorOperations<double>.ElementwiseMultiply(a, b);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        // df/da[i] = b[i], df/db[i] = a[i]
        Assert.NotNull(gradA);
        Assert.NotNull(gradB);
        Assert.Equal(5.0, gradA[0], Tolerance);
        Assert.Equal(6.0, gradA[1], Tolerance);
        Assert.Equal(7.0, gradA[2], Tolerance);
        Assert.Equal(2.0, gradB[0], Tolerance);
        Assert.Equal(3.0, gradB[1], Tolerance);
        Assert.Equal(4.0, gradB[2], Tolerance);
    }

    [Fact]
    public void Divide_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 10.0; aData[1] = 12.0; aData[2] = 14.0;

        var bData = new Tensor<double>(new[] { 3 });
        bData[0] = 2.0; bData[1] = 3.0; bData[2] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // f = sum(a / b)
        var c = TensorOperations<double>.Divide(a, b);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        // df/da[i] = 1/b[i], df/db[i] = -a[i]/b[i]^2
        Assert.NotNull(gradA);
        Assert.NotNull(gradB);
        Assert.Equal(1.0 / 2.0, gradA[0], Tolerance);
        Assert.Equal(1.0 / 3.0, gradA[1], Tolerance);
        Assert.Equal(1.0 / 4.0, gradA[2], Tolerance);
        Assert.Equal(-10.0 / 4.0, gradB[0], Tolerance);
        Assert.Equal(-12.0 / 9.0, gradB[1], Tolerance);
        Assert.Equal(-14.0 / 16.0, gradB[2], Tolerance);
    }

    [Fact]
    public void Power_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 2.0; aData[1] = 3.0; aData[2] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(a^3)
        var c = TensorOperations<double>.Power(a, 3.0);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 3 * a[i]^2
        Assert.NotNull(gradA);
        Assert.Equal(3.0 * 4.0, gradA[0], Tolerance);   // 12
        Assert.Equal(3.0 * 9.0, gradA[1], Tolerance);   // 27
        Assert.Equal(3.0 * 16.0, gradA[2], Tolerance);  // 48
    }

    [Fact]
    public void Square_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 2.0; aData[1] = 3.0; aData[2] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(a^2)
        var c = TensorOperations<double>.Square(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 2 * a[i]
        Assert.NotNull(gradA);
        Assert.Equal(4.0, gradA[0], Tolerance);
        Assert.Equal(6.0, gradA[1], Tolerance);
        Assert.Equal(8.0, gradA[2], Tolerance);
    }

    #endregion

    #region Math Operations

    [Fact]
    public void Exp_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.5; aData[1] = 1.0; aData[2] = 1.5;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(exp(a))
        var c = TensorOperations<double>.Exp(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = exp(a[i])
        Assert.NotNull(gradA);
        Assert.Equal(Math.Exp(0.5), gradA[0], Tolerance);
        Assert.Equal(Math.Exp(1.0), gradA[1], Tolerance);
        Assert.Equal(Math.Exp(1.5), gradA[2], Tolerance);
    }

    [Fact]
    public void Log_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 1.0; aData[1] = 2.0; aData[2] = 3.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(log(a))
        var c = TensorOperations<double>.Log(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1/a[i]
        Assert.NotNull(gradA);
        Assert.Equal(1.0, gradA[0], Tolerance);
        Assert.Equal(0.5, gradA[1], Tolerance);
        Assert.Equal(1.0 / 3.0, gradA[2], Tolerance);
    }

    [Fact]
    public void Sqrt_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 1.0; aData[1] = 4.0; aData[2] = 9.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(sqrt(a))
        var c = TensorOperations<double>.Sqrt(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 0.5 / sqrt(a[i])
        Assert.NotNull(gradA);
        Assert.Equal(0.5, gradA[0], Tolerance);       // 0.5 / 1
        Assert.Equal(0.25, gradA[1], Tolerance);      // 0.5 / 2
        Assert.Equal(1.0 / 6.0, gradA[2], Tolerance); // 0.5 / 3
    }

    [Fact]
    public void Negate_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 1.0; aData[1] = 2.0; aData[2] = 3.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(-a)
        var c = TensorOperations<double>.Negate(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = -1
        Assert.NotNull(gradA);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(-1.0, gradA[i], Tolerance);
        }
    }

    [Fact]
    public void Abs_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 2.0; aData[1] = -3.0; aData[2] = 4.0; aData[3] = -5.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(abs(a))
        var c = TensorOperations<double>.Abs(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = sign(a[i])
        Assert.NotNull(gradA);
        Assert.Equal(1.0, gradA[0], Tolerance);   // positive
        Assert.Equal(-1.0, gradA[1], Tolerance);  // negative
        Assert.Equal(1.0, gradA[2], Tolerance);   // positive
        Assert.Equal(-1.0, gradA[3], Tolerance);  // negative
    }

    #endregion

    #region Activation Functions

    [Fact]
    public void ReLU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 2.0; aData[1] = -1.0; aData[2] = 0.5; aData[3] = -0.5;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(relu(a))
        var c = TensorOperations<double>.ReLU(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1 if a[i] > 0 else 0
        Assert.NotNull(gradA);
        Assert.Equal(1.0, gradA[0], Tolerance);  // positive
        Assert.Equal(0.0, gradA[1], Tolerance);  // negative
        Assert.Equal(1.0, gradA[2], Tolerance);  // positive
        Assert.Equal(0.0, gradA[3], Tolerance);  // negative
    }

    [Fact]
    public void Sigmoid_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.0; aData[1] = 1.0; aData[2] = -1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(sigmoid(a))
        var c = TensorOperations<double>.Sigmoid(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = sigmoid(a[i]) * (1 - sigmoid(a[i]))
        Assert.NotNull(gradA);
        double s0 = 1.0 / (1.0 + Math.Exp(-0.0));  // 0.5
        double s1 = 1.0 / (1.0 + Math.Exp(-1.0));
        double s2 = 1.0 / (1.0 + Math.Exp(1.0));
        Assert.Equal(s0 * (1 - s0), gradA[0], Tolerance);  // 0.25
        Assert.Equal(s1 * (1 - s1), gradA[1], Tolerance);
        Assert.Equal(s2 * (1 - s2), gradA[2], Tolerance);
    }

    [Fact]
    public void Tanh_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.0; aData[1] = 0.5; aData[2] = -0.5;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(tanh(a))
        var c = TensorOperations<double>.Tanh(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1 - tanh(a[i])^2
        Assert.NotNull(gradA);
        double t0 = Math.Tanh(0.0);
        double t1 = Math.Tanh(0.5);
        double t2 = Math.Tanh(-0.5);
        Assert.Equal(1 - t0 * t0, gradA[0], Tolerance);  // 1
        Assert.Equal(1 - t1 * t1, gradA[1], Tolerance);
        Assert.Equal(1 - t2 * t2, gradA[2], Tolerance);
    }

    [Fact]
    public void LeakyReLU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 2.0; aData[1] = -1.0; aData[2] = 0.5; aData[3] = -2.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        double alpha = 0.1;
        // f = sum(leaky_relu(a, alpha))
        var c = TensorOperations<double>.LeakyReLU(a, alpha);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1 if a[i] > 0 else alpha
        Assert.NotNull(gradA);
        Assert.Equal(1.0, gradA[0], Tolerance);
        Assert.Equal(alpha, gradA[1], Tolerance);
        Assert.Equal(1.0, gradA[2], Tolerance);
        Assert.Equal(alpha, gradA[3], Tolerance);
    }

    [Fact]
    public void ELU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 2.0; aData[1] = -0.5; aData[2] = 0.5; aData[3] = -1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        double alpha = 1.0;
        // f = sum(elu(a, alpha))
        var c = TensorOperations<double>.ELU(a, alpha);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1 if a[i] > 0 else alpha * exp(a[i])
        Assert.NotNull(gradA);
        Assert.Equal(1.0, gradA[0], Tolerance);
        Assert.Equal(alpha * Math.Exp(-0.5), gradA[1], Tolerance);
        Assert.Equal(1.0, gradA[2], Tolerance);
        Assert.Equal(alpha * Math.Exp(-1.0), gradA[3], Tolerance);
    }

    [Fact]
    public void SoftPlus_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.0; aData[1] = 1.0; aData[2] = -1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(softplus(a)) = sum(log(1 + exp(a)))
        var c = TensorOperations<double>.SoftPlus(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = exp(a[i]) / (1 + exp(a[i])) = sigmoid(a[i])
        Assert.NotNull(gradA);
        Assert.Equal(1.0 / (1.0 + Math.Exp(-0.0)), gradA[0], Tolerance);  // 0.5
        Assert.Equal(1.0 / (1.0 + Math.Exp(-1.0)), gradA[1], Tolerance);
        Assert.Equal(1.0 / (1.0 + Math.Exp(1.0)), gradA[2], Tolerance);
    }

    [Fact]
    public void Swish_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.0; aData[1] = 1.0; aData[2] = -1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(swish(a)) = sum(a * sigmoid(a))
        var c = TensorOperations<double>.Swish(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = sigmoid(a[i]) + a[i] * sigmoid(a[i]) * (1 - sigmoid(a[i]))
        //          = sigmoid(a[i]) * (1 + a[i] * (1 - sigmoid(a[i])))
        Assert.NotNull(gradA);

        // Verify with numerical gradient
        var numericalGrad = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
            {
                double sig = 1.0 / (1.0 + Math.Exp(-t[i]));
                sum += t[i] * sig;
            }
            return sum;
        });

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(numericalGrad[i], gradA[i], 1e-3);
        }
    }

    [Fact]
    public void GELU_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.0; aData[1] = 1.0; aData[2] = -1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<double>.GELU(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // Verify with numerical gradient
        var numericalGrad = ComputeNumericalGradient(aData, t =>
        {
            double sum = 0;
            double sqrtTwoPi = Math.Sqrt(2.0 / Math.PI);
            for (int i = 0; i < t.Length; i++)
            {
                double x = t[i];
                double arg = sqrtTwoPi * (x + 0.044715 * x * x * x);
                sum += 0.5 * x * (1 + Math.Tanh(arg));
            }
            return sum;
        });

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(numericalGrad[i], gradA[i], 1e-3);
        }
    }

    [Fact]
    public void HardSigmoid_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 5 });
        aData[0] = -3.0;  // Below lower bound
        aData[1] = -1.0;  // In linear region
        aData[2] = 0.0;   // Center
        aData[3] = 1.0;   // In linear region
        aData[4] = 3.0;   // Above upper bound

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<double>.HardSigmoid(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // HardSigmoid (aligned with JIT IR): clip((x + 3) / 6, 0, 1)
        // Derivative: 1/6 if -3 < x < 3, else 0
        double expectedGrad = 1.0 / 6.0;
        Assert.Equal(0.0, gradA[0], Tolerance);           // x = -3, at boundary (saturated)
        Assert.Equal(expectedGrad, gradA[1], Tolerance);  // x = -1, linear region
        Assert.Equal(expectedGrad, gradA[2], Tolerance);  // x = 0, linear region
        Assert.Equal(expectedGrad, gradA[3], Tolerance);  // x = 1, linear region
        Assert.Equal(0.0, gradA[4], Tolerance);           // x = 3, at boundary (saturated)
    }

    [Fact]
    public void SoftSign_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.0; aData[1] = 2.0; aData[2] = -2.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // softsign(x) = x / (1 + |x|)
        var c = TensorOperations<double>.SoftSign(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // d/dx = 1 / (1 + |x|)^2
        Assert.Equal(1.0 / (1.0 * 1.0), gradA[0], Tolerance);  // x=0: 1
        Assert.Equal(1.0 / (3.0 * 3.0), gradA[1], Tolerance);  // x=2: 1/9
        Assert.Equal(1.0 / (3.0 * 3.0), gradA[2], Tolerance);  // x=-2: 1/9
    }

    #endregion

    #region Reduction Operations

    [Fact]
    public void Sum_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 6 });
        for (int i = 0; i < 6; i++) aData[i] = i + 1;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(a)
        var f = TensorOperations<double>.Sum(a);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1 for all i
        Assert.NotNull(gradA);
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(1.0, gradA[i], Tolerance);
        }
    }

    [Fact]
    public void Mean_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 1.0; aData[1] = 2.0; aData[2] = 3.0; aData[3] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = mean(a)
        var f = TensorOperations<double>.Mean(a);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = 1/n for all i
        Assert.NotNull(gradA);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(0.25, gradA[i], Tolerance);  // 1/4
        }
    }

    #endregion

    #region Matrix Operations

    [Fact]
    public void MatrixMultiply_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        // A: 2x3 matrix
        var aData = new Tensor<double>(new[] { 2, 3 });
        aData[0] = 1; aData[1] = 2; aData[2] = 3;
        aData[3] = 4; aData[4] = 5; aData[5] = 6;

        // B: 3x2 matrix
        var bData = new Tensor<double>(new[] { 3, 2 });
        bData[0] = 7; bData[1] = 8;
        bData[2] = 9; bData[3] = 10;
        bData[4] = 11; bData[5] = 12;

        var a = TensorOperations<double>.Variable(aData, "A");
        var b = TensorOperations<double>.Variable(bData, "B");
        tape.Watch(a);
        tape.Watch(b);

        // C = A @ B (2x2 result)
        var c = TensorOperations<double>.MatrixMultiply(a, b);
        // f = sum(C)
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        Assert.NotNull(gradA);
        Assert.NotNull(gradB);

        // Verify shapes
        Assert.Equal(6, gradA.Length);  // 2x3
        Assert.Equal(6, gradB.Length);  // 3x2

        // Verify with numerical gradient
        var numGradA = ComputeNumericalGradient(aData, tA =>
        {
            double sum = 0;
            // C[i,j] = sum_k A[i,k] * B[k,j]
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    double cij = 0;
                    for (int k = 0; k < 3; k++)
                    {
                        cij += tA[i * 3 + k] * bData[k * 2 + j];
                    }
                    sum += cij;
                }
            }
            return sum;
        });

        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(numGradA[i], gradA[i], 1e-3);
        }
    }

    [Fact]
    public void Transpose_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++) aData[i] = i + 1;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(transpose(a))
        var t = TensorOperations<double>.Transpose(a);
        var f = TensorOperations<double>.Sum(t);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // Transpose of upstream gradient of all ones is still all ones
        Assert.NotNull(gradA);
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(1.0, gradA[i], Tolerance);
        }
    }

    #endregion

    #region Softmax

    [Fact]
    public void Softmax_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 1.0; aData[1] = 2.0; aData[2] = 3.0; aData[3] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(softmax(a) * [1, 2, 3, 4])  (weighted sum)
        var s = TensorOperations<double>.Softmax(a);
        var weights = TensorOperations<double>.Constant(new Tensor<double>(new[] { 4 }));
        weights.Value[0] = 1.0; weights.Value[1] = 2.0;
        weights.Value[2] = 3.0; weights.Value[3] = 4.0;
        var weighted = TensorOperations<double>.ElementwiseMultiply(s, weights);
        var f = TensorOperations<double>.Sum(weighted);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // Verify with numerical gradient
        var numGrad = ComputeNumericalGradient(aData, t =>
        {
            // Compute softmax
            double maxVal = double.MinValue;
            for (int i = 0; i < t.Length; i++)
                if (t[i] > maxVal) maxVal = t[i];

            double sumExp = 0;
            var expVals = new double[t.Length];
            for (int i = 0; i < t.Length; i++)
            {
                expVals[i] = Math.Exp(t[i] - maxVal);
                sumExp += expVals[i];
            }

            double sum = 0;
            double[] weightArr = { 1.0, 2.0, 3.0, 4.0 };
            for (int i = 0; i < t.Length; i++)
            {
                sum += (expVals[i] / sumExp) * weightArr[i];
            }
            return sum;
        });

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(numGrad[i], gradA[i], 1e-3);
        }
    }

    [Fact]
    public void LogSoftmax_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 1.0; aData[1] = 2.0; aData[2] = 3.0; aData[3] = 4.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var s = TensorOperations<double>.LogSoftmax(a);
        var f = TensorOperations<double>.Sum(s);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // For sum(log_softmax), the gradient should be 1 - n*softmax
        // where n is the dimension
        var numGrad = ComputeNumericalGradient(aData, t =>
        {
            double maxVal = double.MinValue;
            for (int i = 0; i < t.Length; i++)
                if (t[i] > maxVal) maxVal = t[i];

            double sumExp = 0;
            for (int i = 0; i < t.Length; i++)
                sumExp += Math.Exp(t[i] - maxVal);

            double logSumExp = maxVal + Math.Log(sumExp);
            double sum = 0;
            for (int i = 0; i < t.Length; i++)
                sum += t[i] - logSumExp;
            return sum;
        });

        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(numGrad[i], gradA[i], 1e-3);
        }
    }

    #endregion

    #region Chain Rule Tests

    [Fact]
    public void ChainRule_ExpOfSquare_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.5; aData[1] = 1.0; aData[2] = 0.3;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(exp(a^2))
        var sq = TensorOperations<double>.Square(a);
        var exp = TensorOperations<double>.Exp(sq);
        var f = TensorOperations<double>.Sum(exp);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];

        // df/da[i] = exp(a[i]^2) * 2 * a[i]
        Assert.NotNull(gradA);
        for (int i = 0; i < 3; i++)
        {
            double expected = Math.Exp(aData[i] * aData[i]) * 2 * aData[i];
            Assert.Equal(expected, gradA[i], Tolerance);
        }
    }

    [Fact]
    public void ChainRule_LogOfSigmoid_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 0.5; aData[1] = 1.0; aData[2] = -0.5;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        // f = sum(log(sigmoid(a)))
        var sig = TensorOperations<double>.Sigmoid(a);
        var logSig = TensorOperations<double>.Log(sig);
        var f = TensorOperations<double>.Sum(logSig);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // df/da[i] = 1 - sigmoid(a[i])
        for (int i = 0; i < 3; i++)
        {
            double sigVal = 1.0 / (1.0 + Math.Exp(-aData[i]));
            double expected = 1.0 - sigVal;
            Assert.Equal(expected, gradA[i], Tolerance);
        }
    }

    [Fact]
    public void ChainRule_MultipleOperations_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 2 });
        aData[0] = 1.5; aData[1] = 2.0;

        var bData = new Tensor<double>(new[] { 2 });
        bData[0] = 0.5; bData[1] = 0.3;

        var a = TensorOperations<double>.Variable(aData, "a");
        var b = TensorOperations<double>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        // f = sum(tanh(a * b + a^2))
        var ab = TensorOperations<double>.ElementwiseMultiply(a, b);
        var aSq = TensorOperations<double>.Square(a);
        var sumTerm = TensorOperations<double>.Add(ab, aSq);
        var th = TensorOperations<double>.Tanh(sumTerm);
        var f = TensorOperations<double>.Sum(th);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        Assert.NotNull(gradA);
        Assert.NotNull(gradB);

        // Verify with numerical gradient
        var numGradA = ComputeNumericalGradient(aData, tA =>
        {
            double sum = 0;
            for (int i = 0; i < tA.Length; i++)
            {
                double val = tA[i] * bData[i] + tA[i] * tA[i];
                sum += Math.Tanh(val);
            }
            return sum;
        });

        var numGradB = ComputeNumericalGradient(bData, tB =>
        {
            double sum = 0;
            for (int i = 0; i < tB.Length; i++)
            {
                double val = aData[i] * tB[i] + aData[i] * aData[i];
                sum += Math.Tanh(val);
            }
            return sum;
        });

        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(numGradA[i], gradA[i], 1e-3);
            Assert.Equal(numGradB[i], gradB[i], 1e-3);
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void Exp_LargeInput_HandlesOverflow()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 2 });
        aData[0] = 700.0;  // Very large, near overflow
        aData[1] = 1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<double>.Exp(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // Should not be NaN or Inf for reasonable inputs
        Assert.True(!double.IsNaN(gradA[1]) && !double.IsInfinity(gradA[1]));
    }

    [Fact]
    public void Log_SmallPositiveInput_HandlesCorrectly()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 1e-10;  // Very small but positive
        aData[1] = 1e-5;
        aData[2] = 1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<double>.Log(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // Gradient should be 1/x, which can be very large for small x
        Assert.True(!double.IsNaN(gradA[0]));
        Assert.True(!double.IsNaN(gradA[1]));
        Assert.Equal(1.0, gradA[2], Tolerance);
    }

    [Fact]
    public void Sqrt_SmallPositiveInput_HandlesCorrectly()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 3 });
        aData[0] = 1e-10;  // Very small but positive
        aData[1] = 0.01;
        aData[2] = 1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<double>.Sqrt(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // Gradient should be 0.5/sqrt(x), which can be very large for small x
        Assert.True(!double.IsNaN(gradA[0]));
        Assert.True(!double.IsNaN(gradA[1]));
        Assert.Equal(0.5, gradA[2], Tolerance);
    }

    [Fact]
    public void Sigmoid_ExtremeInputs_NumericallyStable()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = -50.0;  // Very negative
        aData[1] = 50.0;   // Very positive
        aData[2] = 0.0;
        aData[3] = 1.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<double>.Sigmoid(a);
        var f = TensorOperations<double>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // Gradients should not be NaN
        for (int i = 0; i < 4; i++)
        {
            Assert.False(double.IsNaN(gradA[i]), $"Gradient at index {i} is NaN");
        }

        // For extreme inputs, sigmoid saturates, so gradient should be near 0
        Assert.True(gradA[0] < 1e-10);  // Nearly 0 for -50
        Assert.True(gradA[1] < 1e-10);  // Nearly 0 for +50
        Assert.Equal(0.25, gradA[2], Tolerance);  // 0.5 * 0.5 for x=0
    }

    [Fact]
    public void Softmax_NumericallyStable()
    {
        using var tape = new GradientTape<double>();

        var aData = new Tensor<double>(new[] { 4 });
        aData[0] = 1000.0;  // Very large values
        aData[1] = 1001.0;
        aData[2] = 1002.0;
        aData[3] = 1003.0;

        var a = TensorOperations<double>.Variable(aData, "a");
        tape.Watch(a);

        var s = TensorOperations<double>.Softmax(a);
        var f = TensorOperations<double>.Sum(s);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        // Softmax should handle large inputs via max subtraction
        // Sum of softmax is always 1, so gradients should sum to 0 (approximately)
        for (int i = 0; i < 4; i++)
        {
            Assert.False(double.IsNaN(gradA[i]), $"Gradient at index {i} is NaN");
            Assert.False(double.IsInfinity(gradA[i]), $"Gradient at index {i} is Inf");
        }
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Add_Float_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<float>();

        var aData = new Tensor<float>(new[] { 3 });
        aData[0] = 1.0f; aData[1] = 2.0f; aData[2] = 3.0f;

        var bData = new Tensor<float>(new[] { 3 });
        bData[0] = 4.0f; bData[1] = 5.0f; bData[2] = 6.0f;

        var a = TensorOperations<float>.Variable(aData, "a");
        var b = TensorOperations<float>.Variable(bData, "b");
        tape.Watch(a);
        tape.Watch(b);

        var c = TensorOperations<float>.Add(a, b);
        var f = TensorOperations<float>.Sum(c);

        var grads = tape.Gradient(f, new[] { a, b });
        var gradA = grads[a];
        var gradB = grads[b];

        Assert.NotNull(gradA);
        Assert.NotNull(gradB);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(1.0f, gradA[i], FloatTolerance);
            Assert.Equal(1.0f, gradB[i], FloatTolerance);
        }
    }

    [Fact]
    public void Sigmoid_Float_GradientMatchesNumerical()
    {
        using var tape = new GradientTape<float>();

        var aData = new Tensor<float>(new[] { 3 });
        aData[0] = 0.0f; aData[1] = 1.0f; aData[2] = -1.0f;

        var a = TensorOperations<float>.Variable(aData, "a");
        tape.Watch(a);

        var c = TensorOperations<float>.Sigmoid(a);
        var f = TensorOperations<float>.Sum(c);

        var grads = tape.Gradient(f, new[] { a });
        var gradA = grads[a];
        Assert.NotNull(gradA);

        for (int i = 0; i < 3; i++)
        {
            float sigVal = 1.0f / (1.0f + (float)Math.Exp(-aData[i]));
            float expected = sigVal * (1.0f - sigVal);
            Assert.Equal(expected, gradA[i], FloatTolerance);
        }
    }

    #endregion
}
