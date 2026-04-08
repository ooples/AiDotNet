using AiDotNet.ActivationFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ActivationFunctions;

/// <summary>
/// Deep mathematical tests for remaining activation functions (Part 4).
/// Tests hand-calculated values, mathematical properties, and gradient correctness.
/// </summary>
public class ActivationFunctionDeepMathIntegrationTests4
{
    private const double Tol = 1e-6;

    // ====================================================================
    // BinarySpiking: f(x) = 1 if x >= threshold, 0 otherwise
    // ====================================================================

    [Fact]
    public void BinarySpiking_AboveThreshold_ReturnsOne()
    {
        var fn = new BinarySpikingActivation<double>(); // default threshold = 1.0
        Assert.Equal(1.0, fn.Activate(1.5), Tol);
        Assert.Equal(1.0, fn.Activate(1.0), Tol); // at threshold
    }

    [Fact]
    public void BinarySpiking_BelowThreshold_ReturnsZero()
    {
        var fn = new BinarySpikingActivation<double>();
        Assert.Equal(0.0, fn.Activate(0.5), Tol);
        Assert.Equal(0.0, fn.Activate(-1.0), Tol);
    }

    [Fact]
    public void BinarySpiking_CustomThreshold()
    {
        var fn = new BinarySpikingActivation<double>(0.5, 1.0, 0.2);
        Assert.Equal(1.0, fn.Activate(0.5), Tol); // at threshold
        Assert.Equal(0.0, fn.Activate(0.3), Tol); // below
    }

    // ====================================================================
    // Squash: f(x) = x / (1 + |x|), bounded in (-1, 1)
    // f'(x) = 1 / (1 + |x|)^2
    // ====================================================================

    [Fact]
    public void Squash_AtZero_ReturnsZero()
    {
        var fn = new SquashActivation<double>();
        Assert.Equal(0.0, fn.Activate(0.0), Tol);
    }

    [Fact]
    public void Squash_HandCalculated()
    {
        // Scalar squash: f(x) = sign(x) * x² / (1 + x²) = x * |x| / (1 + x²)
        var fn = new SquashActivation<double>();
        // f(2) = 2*2 / (1+4) = 4/5 = 0.8
        Assert.Equal(0.8, fn.Activate(2.0), Tol);
        // f(-3) = -3*3 / (1+9) = -9/10 = -0.9
        Assert.Equal(-0.9, fn.Activate(-3.0), Tol);
        // f(1) = 1*1 / (1+1) = 1/2 = 0.5
        Assert.Equal(0.5, fn.Activate(1.0), Tol);
    }

    [Fact]
    public void Squash_OddFunction()
    {
        var fn = new SquashActivation<double>();
        double[] xs = [0.5, 1.0, 2.0, 10.0];
        foreach (double x in xs)
            Assert.Equal(-fn.Activate(x), fn.Activate(-x), Tol);
    }

    [Fact]
    public void Squash_BoundedBetweenMinusOneAndOne()
    {
        var fn = new SquashActivation<double>();
        double[] xs = [-100, -10, -1, 0, 1, 10, 100];
        foreach (double x in xs)
        {
            double y = fn.Activate(x);
            Assert.True(y > -1.0 && y < 1.0, $"Squash({x}) = {y} should be in (-1, 1)");
        }
    }

    [Fact]
    public void Squash_NumericalGradient()
    {
        var fn = new SquashActivation<double>();
        double h = 1e-7;
        double[] xs = [-2, -0.5, 0, 0.5, 2];
        foreach (double x in xs)
        {
            double analytical = fn.Derivative(x);
            double numerical = (fn.Activate(x + h) - fn.Activate(x - h)) / (2 * h);
            Assert.True(Math.Abs(analytical - numerical) < 1e-4,
                $"Gradient mismatch at x={x}: analytical={analytical}, numerical={numerical}");
        }
    }

    // ====================================================================
    // Sparsemax: projects onto probability simplex (sparse softmax alternative)
    // Properties: outputs sum to 1, some outputs are exactly 0 (sparse)
    // ====================================================================

    [Fact]
    public void Sparsemax_OutputsSumToOne()
    {
        var fn = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, 1e-5);
    }

    [Fact]
    public void Sparsemax_AllEqual_ReturnsUniform()
    {
        var fn = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { 2.0, 2.0, 2.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(1.0 / 3.0, output[i], 1e-5);
    }

    [Fact]
    public void Sparsemax_AllNonNegative()
    {
        var fn = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { -5.0, 1.0, 3.0, -2.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] >= -1e-10, $"Sparsemax output should be >= 0, got {output[i]}");
    }

    [Fact]
    public void Sparsemax_ProducesSparseOutput()
    {
        // With very different inputs, sparsemax should zero out the smallest ones
        var fn = new SparsemaxActivation<double>();
        var input = new Vector<double>(new[] { -10.0, 0.0, 10.0 });
        var output = fn.Activate(input);

        // The largest input should get most weight; small inputs should be ~0
        Assert.True(output[2] > 0.5, $"Largest input should get most weight, got {output[2]}");
    }

    // ====================================================================
    // SphericalSoftmax: normalizes to unit sphere, then softmax
    // ====================================================================

    [Fact]
    public void SphericalSoftmax_OutputsSumToOne()
    {
        var fn = new SphericalSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, 1e-5);
    }

    [Fact]
    public void SphericalSoftmax_AllNonNegative()
    {
        var fn = new SphericalSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { -5.0, 0.0, 5.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] >= 0, $"SphericalSoftmax output should be >= 0, got {output[i]}");
    }

    [Fact]
    public void SphericalSoftmax_ScaleInvariance()
    {
        // SphericalSoftmax normalizes to unit sphere first, so scaling shouldn't change output
        var fn = new SphericalSoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var scaled = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
        var output1 = fn.Activate(input);
        var output2 = fn.Activate(scaled);
        for (int i = 0; i < 3; i++)
            Assert.Equal(output1[i], output2[i], 1e-5);
    }

    // ====================================================================
    // TaylorSoftmax: softmax approximation using Taylor expansion of exp
    // ====================================================================

    [Fact]
    public void TaylorSoftmax_OutputsSumToOne()
    {
        var fn = new TaylorSoftmaxActivation<double>(order: 2);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, 1e-5);
    }

    [Fact]
    public void TaylorSoftmax_AllNonNegative()
    {
        var fn = new TaylorSoftmaxActivation<double>(order: 2);
        var input = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] >= 0, $"TaylorSoftmax output should be >= 0, got {output[i]}");
    }

    [Fact]
    public void TaylorSoftmax_AllEqual_ReturnsUniform()
    {
        var fn = new TaylorSoftmaxActivation<double>(order: 2);
        var input = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(1.0 / 3.0, output[i], 1e-5);
    }

    [Fact]
    public void TaylorSoftmax_ApproachesSoftmaxForSmallInputs()
    {
        // For small inputs, Taylor approximation should be close to exact softmax
        var taylor = new TaylorSoftmaxActivation<double>(order: 4);
        var softmax = new SoftmaxActivation<double>();
        var input = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
        var taylorOut = taylor.Activate(input);
        var softmaxOut = softmax.Activate(input);
        for (int i = 0; i < 3; i++)
            Assert.Equal(softmaxOut[i], taylorOut[i], 0.05); // within 5%
    }

    // ====================================================================
    // Maxout: f(x) = max over groups of inputs
    // ====================================================================

    [Fact]
    public void Maxout_SelectsMaxPerGroup()
    {
        var fn = new MaxoutActivation<double>(2); // 2 pieces per group
        // Input [1, 3, 2, 4] with 2 pieces → groups: (1,3), (2,4) → max: [3, 4]
        var input = new Vector<double>(new[] { 1.0, 3.0, 2.0, 4.0 });
        var output = fn.Activate(input);

        Assert.Equal(2, output.Length); // 4 inputs / 2 pieces = 2 outputs
        Assert.Equal(3.0, output[0], Tol); // max(1, 3) = 3
        Assert.Equal(4.0, output[1], Tol); // max(2, 4) = 4
    }

    [Fact]
    public void Maxout_AllSame_ReturnsSame()
    {
        var fn = new MaxoutActivation<double>(3);
        var input = new Vector<double>(new[] { 5.0, 5.0, 5.0 });
        var output = fn.Activate(input);
        Assert.Single(output);
        Assert.Equal(5.0, output[0], Tol);
    }

    // ====================================================================
    // GumbelSoftmax: temperature-dependent softmax with Gumbel noise
    // ====================================================================

    [Fact]
    public void GumbelSoftmax_OutputsSumToOne()
    {
        var fn = new GumbelSoftmaxActivation<double>(temperature: 1.0, seed: 42);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, 1e-5);
    }

    [Fact]
    public void GumbelSoftmax_AllNonNegative()
    {
        var fn = new GumbelSoftmaxActivation<double>(temperature: 1.0, seed: 42);
        var input = new Vector<double>(new[] { -5.0, 0.0, 5.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] >= -1e-10, $"GumbelSoftmax should be >= 0, got {output[i]}");
    }

    [Fact]
    public void GumbelSoftmax_LowTemperature_ApproachesOneHot()
    {
        // With very low temperature, output should approach one-hot
        var fn = new GumbelSoftmaxActivation<double>(temperature: 0.01, seed: 42);
        var input = new Vector<double>(new[] { 1.0, 5.0, 2.0 });
        var output = fn.Activate(input);

        // The max element should dominate
        double maxVal = double.MinValue;
        for (int i = 0; i < output.Length; i++)
            if (output[i] > maxVal) maxVal = output[i];
        Assert.True(maxVal > 0.9, $"Low-temp Gumbel softmax max should be > 0.9, got {maxVal}");
    }

    // ====================================================================
    // HierarchicalSoftmax: binary tree decomposition of softmax
    // ====================================================================

    [Fact]
    public void HierarchicalSoftmax_OutputsSumToOne()
    {
        var fn = new HierarchicalSoftmaxActivation<double>(4);
        var input = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var output = fn.Activate(input);
        double sum = 0;
        for (int i = 0; i < output.Length; i++) sum += output[i];
        Assert.Equal(1.0, sum, 1e-4);
    }

    [Fact]
    public void HierarchicalSoftmax_AllNonNegative()
    {
        var fn = new HierarchicalSoftmaxActivation<double>(3);
        var input = new Vector<double>(new[] { -1.0, 0.0, 1.0 });
        var output = fn.Activate(input);
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] >= -1e-10, $"HierSoftmax should be >= 0, got {output[i]}");
    }
}
