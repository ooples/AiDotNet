using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Deep mathematical tests for activation functions not covered in Parts 1 and 2.
/// Each test verifies hand-calculated values, derivative correctness, and mathematical properties.
/// </summary>
public class ActivationFunctionDeepMathIntegrationTests3
{
    private const double Tol = 1e-8;
    private const double GradTol = 1e-4;

    /// <summary>
    /// Verifies analytical derivative matches numerical gradient: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    /// </summary>
    private static void AssertNumericalGradient(IActivationFunction<double> fn, double x, double h = 1e-7)
    {
        double analytical = fn.Derivative(x);
        double numerical = (fn.Activate(x + h) - fn.Activate(x - h)) / (2 * h);
        Assert.True(Math.Abs(analytical - numerical) < GradTol,
            $"Gradient mismatch at x={x}: analytical={analytical}, numerical={numerical}");
    }

    // ====================================================================
    // Identity: f(x) = x, f'(x) = 1
    // ====================================================================

    [Fact]
    public void Identity_Activate_ReturnsInput()
    {
        var fn = new IdentityActivation<double>();
        double[] xs = [-5, -1, 0, 0.5, 3, 100];
        foreach (double x in xs)
            Assert.Equal(x, fn.Activate(x), Tol);
    }

    [Fact]
    public void Identity_Derivative_AlwaysOne()
    {
        var fn = new IdentityActivation<double>();
        double[] xs = [-5, -1, 0, 0.5, 3, 100];
        foreach (double x in xs)
            Assert.Equal(1.0, fn.Derivative(x), Tol);
    }

    // ====================================================================
    // PReLU: f(x) = x if x>0, alpha*x otherwise; f'(x) = 1 if x>0, alpha otherwise
    // ====================================================================

    [Fact]
    public void PReLU_Positive_ReturnsInput()
    {
        var fn = new PReLUActivation<double>(0.01);
        Assert.Equal(2.0, fn.Activate(2.0), Tol);
        Assert.Equal(0.5, fn.Activate(0.5), Tol);
    }

    [Fact]
    public void PReLU_Negative_ReturnsAlphaTimesInput()
    {
        var fn = new PReLUActivation<double>(0.1);
        // f(-2) = 0.1 * -2 = -0.2
        Assert.Equal(-0.2, fn.Activate(-2.0), Tol);
        Assert.Equal(-0.05, fn.Activate(-0.5), Tol);
    }

    [Fact]
    public void PReLU_AtZero_ReturnsZero()
    {
        var fn = new PReLUActivation<double>(0.1);
        Assert.Equal(0.0, fn.Activate(0.0), Tol);
    }

    [Fact]
    public void PReLU_Derivative_PositiveRegion_IsOne()
    {
        var fn = new PReLUActivation<double>(0.1);
        Assert.Equal(1.0, fn.Derivative(1.0), Tol);
        Assert.Equal(1.0, fn.Derivative(0.01), Tol);
    }

    [Fact]
    public void PReLU_Derivative_NegativeRegion_IsAlpha()
    {
        var fn = new PReLUActivation<double>(0.1);
        Assert.Equal(0.1, fn.Derivative(-1.0), Tol);
        Assert.Equal(0.1, fn.Derivative(-0.01), Tol);
    }

    [Fact]
    public void PReLU_NumericalGradient()
    {
        var fn = new PReLUActivation<double>(0.25);
        AssertNumericalGradient(fn, 1.0);
        AssertNumericalGradient(fn, -1.0);
        AssertNumericalGradient(fn, 3.0);
        AssertNumericalGradient(fn, -3.0);
    }

    // ====================================================================
    // RReLU: f(x) = x if x>0, alpha*x otherwise (alpha random in [lower, upper])
    // In eval mode, alpha = (lower+upper)/2
    // ====================================================================

    [Fact]
    public void RReLU_Positive_ReturnsInput()
    {
        var fn = new RReLUActivation<double>(0.1, 0.3);
        fn.SetTrainingMode(false); // Use fixed alpha = (0.1+0.3)/2 = 0.2
        Assert.Equal(5.0, fn.Activate(5.0), Tol);
    }

    [Fact]
    public void RReLU_Negative_ReturnsMidAlphaTimesInput()
    {
        var fn = new RReLUActivation<double>(0.1, 0.3);
        fn.SetTrainingMode(false); // Use fixed alpha = (0.1+0.3)/2 = 0.2
        // eval alpha = (0.1+0.3)/2 = 0.2
        // f(-2) = 0.2 * -2 = -0.4
        Assert.Equal(-0.4, fn.Activate(-2.0), Tol);
    }

    [Fact]
    public void RReLU_NumericalGradient()
    {
        var fn = new RReLUActivation<double>(0.1, 0.3);
        fn.SetTrainingMode(false); // Fixed alpha for deterministic gradient check
        AssertNumericalGradient(fn, 2.0);
        AssertNumericalGradient(fn, -2.0);
    }

    // ====================================================================
    // SiLU/Swish: f(x) = x * sigmoid(x), f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    // ====================================================================

    [Fact]
    public void SiLU_AtZero_ReturnsZero()
    {
        var fn = new SiLUActivation<double>();
        Assert.Equal(0.0, fn.Activate(0.0), Tol);
    }

    [Fact]
    public void SiLU_HandCalculated()
    {
        var fn = new SiLUActivation<double>();
        // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + e^-1)
        double expected = 1.0 / (1.0 + Math.Exp(-1.0));
        Assert.Equal(expected, fn.Activate(1.0), Tol);
    }

    [Fact]
    public void SiLU_Derivative_AtZero()
    {
        var fn = new SiLUActivation<double>();
        // f'(0) = sigmoid(0) + 0 * sigmoid(0) * (1 - sigmoid(0)) = 0.5
        Assert.Equal(0.5, fn.Derivative(0.0), Tol);
    }

    [Fact]
    public void SiLU_NumericalGradient()
    {
        var fn = new SiLUActivation<double>();
        AssertNumericalGradient(fn, -2.0);
        AssertNumericalGradient(fn, 0.0);
        AssertNumericalGradient(fn, 1.0);
        AssertNumericalGradient(fn, 3.0);
    }

    // ====================================================================
    // Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
    // ====================================================================

    [Fact]
    public void Softmax_OutputsSumToOne()
    {
        var fn = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, Tol);
    }

    [Fact]
    public void Softmax_AllEqual_ReturnsUniform()
    {
        var fn = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(1.0 / 3.0, output[i], Tol);
    }

    [Fact]
    public void Softmax_HandCalculated()
    {
        var fn = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);

        double e1 = Math.Exp(1), e2 = Math.Exp(2), e3 = Math.Exp(3);
        double total = e1 + e2 + e3;
        Assert.Equal(e1 / total, output[0], Tol);
        Assert.Equal(e2 / total, output[1], Tol);
        Assert.Equal(e3 / total, output[2], Tol);
    }

    [Fact]
    public void Softmax_ShiftInvariance()
    {
        // softmax(x + c) = softmax(x) for any constant c
        var fn = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var shifted = new Vector<double>(new[] { 101.0, 102.0, 103.0 });
        var output1 = fn.Activate(input);
        var output2 = fn.Activate(shifted);
        for (int i = 0; i < 3; i++)
            Assert.Equal(output1[i], output2[i], 1e-6);
    }

    [Fact]
    public void Softmax_AllNonNegative()
    {
        var fn = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { -5.0, -10.0, -100.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] >= 0, $"Softmax output should be non-negative, got {output[i]}");
    }

    // ====================================================================
    // LogSoftmax: f(x_i) = log(softmax(x_i)) = x_i - log(sum(exp(x_j)))
    // ====================================================================

    [Fact]
    public void LogSoftmax_AllEqual_ReturnsLogUniform()
    {
        var fn = new LogSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var output = fn.Activate(input);
        double expected = -Math.Log(3.0);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(expected, output[i], Tol);
    }

    [Fact]
    public void LogSoftmax_OutputsAlwaysNonPositive()
    {
        var fn = new LogSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] <= 0 + 1e-10, $"LogSoftmax should be <= 0, got {output[i]}");
    }

    [Fact]
    public void LogSoftmax_ExpMatchesSoftmax()
    {
        var logSm = new LogSoftmaxActivation<double>();
        var sm = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var logOutput = logSm.Activate(input);
        var smOutput = sm.Activate(input);
        for (int i = 0; i < 3; i++)
            Assert.Equal(smOutput[i], Math.Exp(logOutput[i]), Tol);
    }

    // ====================================================================
    // Sign: f(x) = -1 if x<0, 0 if x=0, 1 if x>0
    // ====================================================================

    [Fact]
    public void Sign_Values()
    {
        var fn = new SignActivation<double>();
        Assert.Equal(-1.0, fn.Activate(-5.0), Tol);
        Assert.Equal(0.0, fn.Activate(0.0), Tol);
        Assert.Equal(1.0, fn.Activate(5.0), Tol);
    }

    [Fact]
    public void Sign_Derivative_AlwaysZero()
    {
        // Sign function derivative is 0 everywhere (undefined at 0, conventionally 0)
        var fn = new SignActivation<double>();
        Assert.Equal(0.0, fn.Derivative(-1.0), Tol);
        Assert.Equal(0.0, fn.Derivative(0.0), Tol);
        Assert.Equal(0.0, fn.Derivative(1.0), Tol);
    }

    // ====================================================================
    // ScaledTanh: f(x) = (1 - e^(-βx)) / (1 + e^(-βx)) = tanh(βx/2)
    // ====================================================================

    [Fact]
    public void ScaledTanh_AtZero_ReturnsZero()
    {
        var fn = new ScaledTanhActivation<double>(2.0 / 3.0);
        Assert.Equal(0.0, fn.Activate(0.0), Tol);
    }

    [Fact]
    public void ScaledTanh_HandCalculated()
    {
        double beta = 2.0 / 3.0;
        var fn = new ScaledTanhActivation<double>(beta);
        // f(x) = (1 - e^(-βx)) / (1 + e^(-βx)) = tanh(βx/2)
        double bx = beta * 1.0;
        double expected = (1.0 - Math.Exp(-bx)) / (1.0 + Math.Exp(-bx));
        Assert.Equal(expected, fn.Activate(1.0), Tol);
    }

    [Fact]
    public void ScaledTanh_Beta2_EqualsTanh()
    {
        // When β=2: f(x) = tanh(x)
        var fn = new ScaledTanhActivation<double>(2.0);
        double[] xs = [-2, -1, 0, 1, 2];
        foreach (double x in xs)
            Assert.Equal(Math.Tanh(x), fn.Activate(x), 1e-6);
    }

    [Fact]
    public void ScaledTanh_OddFunction()
    {
        // f(-x) = -f(x) (odd symmetry)
        var fn = new ScaledTanhActivation<double>(2.0 / 3.0);
        double[] xs = [0.5, 1.0, 2.0, 5.0];
        foreach (double x in xs)
            Assert.Equal(-fn.Activate(x), fn.Activate(-x), Tol);
    }

    [Fact]
    public void ScaledTanh_NumericalGradient()
    {
        var fn = new ScaledTanhActivation<double>(2.0 / 3.0);
        AssertNumericalGradient(fn, -2.0);
        AssertNumericalGradient(fn, 0.0);
        AssertNumericalGradient(fn, 1.0);
        AssertNumericalGradient(fn, 3.0);
    }

    // ====================================================================
    // Softmin: f(x_i) = exp(-x_i) / sum(exp(-x_j))
    // ====================================================================

    [Fact]
    public void Softmin_OutputsSumToOne()
    {
        var fn = new SoftminActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, Tol);
    }

    [Fact]
    public void Softmin_SmallestInput_GetsLargestWeight()
    {
        var fn = new SoftminActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 5.0, 10.0 });
        var output = fn.Activate(input);
        // Smallest input (1.0) should get the largest softmin weight
        Assert.True(output[0] > output[1]);
        Assert.True(output[1] > output[2]);
    }

    // ====================================================================
    // SQRBF: f(x) = exp(-x^2), Gaussian radial basis function
    // ====================================================================

    [Fact]
    public void SQRBF_AtZero_ReturnsOne()
    {
        var fn = new SQRBFActivation<double>();
        Assert.Equal(1.0, fn.Activate(0.0), Tol);
    }

    [Fact]
    public void SQRBF_HandCalculated()
    {
        var fn = new SQRBFActivation<double>();
        // f(1) = exp(-1) ≈ 0.36788
        Assert.Equal(Math.Exp(-1.0), fn.Activate(1.0), Tol);
        // f(2) = exp(-4) ≈ 0.01832
        Assert.Equal(Math.Exp(-4.0), fn.Activate(2.0), Tol);
    }

    [Fact]
    public void SQRBF_EvenFunction()
    {
        // f(-x) = f(x) (even symmetry)
        var fn = new SQRBFActivation<double>();
        double[] xs = [0.5, 1.0, 2.0, 3.0];
        foreach (double x in xs)
            Assert.Equal(fn.Activate(x), fn.Activate(-x), Tol);
    }

    [Fact]
    public void SQRBF_OutputAlwaysPositive()
    {
        var fn = new SQRBFActivation<double>();
        double[] xs = [-10, -1, 0, 1, 10];
        foreach (double x in xs)
            Assert.True(fn.Activate(x) > 0);
    }

    [Fact]
    public void SQRBF_NumericalGradient()
    {
        var fn = new SQRBFActivation<double>();
        AssertNumericalGradient(fn, -2.0);
        AssertNumericalGradient(fn, -0.5);
        AssertNumericalGradient(fn, 0.0);
        AssertNumericalGradient(fn, 1.0);
        AssertNumericalGradient(fn, 2.0);
    }

    [Fact]
    public void SQRBF_Derivative_HandCalculated()
    {
        // f'(x) = -2x * exp(-x^2)
        var fn = new SQRBFActivation<double>();
        double x = 1.0;
        double expected = -2.0 * x * Math.Exp(-x * x);
        Assert.Equal(expected, fn.Derivative(x), Tol);
    }

    // ====================================================================
    // LogSoftmin: f(x_i) = log(softmin(x_i)) = -x_i - log(sum(exp(-x_j)))
    // ====================================================================

    [Fact]
    public void LogSoftmin_AllEqual_ReturnsLogUniform()
    {
        var fn = new LogSoftminActivation<double>();
        var input = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var output = fn.Activate(input);
        double expected = -Math.Log(3.0);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(expected, output[i], Tol);
    }

    [Fact]
    public void LogSoftmin_OutputsAlwaysNonPositive()
    {
        var fn = new LogSoftminActivation<double>();
        var input = new Vector<double>(new[] { -1.0, -2.0, -3.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] <= 0 + 1e-10, $"LogSoftmin should be <= 0, got {output[i]}");
    }
}
