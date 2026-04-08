using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Kernels;

/// <summary>
/// Integration tests for kernel function classes.
/// Tests mathematical properties: K(x,x)=max, K(x,y)=K(y,x), positive values, boundary behavior.
/// </summary>
public class KernelFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    private static Vector<double> V(params double[] values) => new(values);

    #region Gaussian Kernel Tests

    [Fact]
    public void GaussianKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var x = V(1.0, 2.0, 3.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void GaussianKernel_DifferentVectors_ReturnsLessThanOne()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var x = V(1.0, 2.0, 3.0);
        var y = V(4.0, 5.0, 6.0);
        var result = kernel.Calculate(x, y);
        Assert.True(result > 0);
        Assert.True(result < 1.0);
    }

    [Fact]
    public void GaussianKernel_IsSymmetric()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        var r1 = kernel.Calculate(x, y);
        var r2 = kernel.Calculate(y, x);
        Assert.Equal(r1, r2, Tolerance);
    }

    [Fact]
    public void GaussianKernel_LargerSigma_HigherSimilarityForDistantPoints()
    {
        var x = V(1.0, 2.0);
        var y = V(5.0, 6.0);
        var narrowKernel = new GaussianKernel<double>(0.5);
        var wideKernel = new GaussianKernel<double>(5.0);
        var narrow = narrowKernel.Calculate(x, y);
        var wide = wideKernel.Calculate(x, y);
        Assert.True(wide > narrow);
    }

    [Fact]
    public void GaussianKernel_DifferentLengthVectors_ThrowsArgumentException()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(1.0, 2.0, 3.0);
        Assert.Throws<ArgumentException>(() => kernel.Calculate(x, y));
    }

    #endregion

    #region Linear Kernel Tests

    [Fact]
    public void LinearKernel_Calculate_ReturnsDotProduct()
    {
        var kernel = new LinearKernel<double>();
        var x = V(1.0, 2.0, 3.0);
        var y = V(4.0, 5.0, 6.0);
        var result = kernel.Calculate(x, y);
        // dot product: 1*4 + 2*5 + 3*6 = 32
        Assert.Equal(32.0, result, Tolerance);
    }

    [Fact]
    public void LinearKernel_IsSymmetric()
    {
        var kernel = new LinearKernel<double>();
        var x = V(1.0, 3.0);
        var y = V(2.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Polynomial Kernel Tests

    [Fact]
    public void PolynomialKernel_Degree2_ComputesCorrectly()
    {
        var kernel = new PolynomialKernel<double>(degree: 2, coef0: 1.0);
        var x = V(1.0, 0.0);
        var y = V(1.0, 0.0);
        // (1*1 + 0*0 + 1)^2 = (2)^2 = 4
        var result = kernel.Calculate(x, y);
        Assert.Equal(4.0, result, Tolerance);
    }

    [Fact]
    public void PolynomialKernel_Degree1_ReducesToLinear()
    {
        var polyKernel = new PolynomialKernel<double>(degree: 1, coef0: 0.0);
        var linearKernel = new LinearKernel<double>();
        var x = V(1.0, 2.0, 3.0);
        var y = V(4.0, 5.0, 6.0);
        Assert.Equal(linearKernel.Calculate(x, y), polyKernel.Calculate(x, y), Tolerance);
    }

    [Fact]
    public void PolynomialKernel_IsSymmetric()
    {
        var kernel = new PolynomialKernel<double>(degree: 3, coef0: 1.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Sigmoid Kernel Tests

    [Fact]
    public void SigmoidKernel_Calculate_ReturnsBounded()
    {
        var kernel = new SigmoidKernel<double>();
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        var result = kernel.Calculate(x, y);
        // tanh-based, output bounded between -1 and 1
        Assert.True(result >= -1.0);
        Assert.True(result <= 1.0);
    }

    #endregion

    #region Laplacian Kernel Tests

    [Fact]
    public void LaplacianKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new LaplacianKernel<double>(1.0);
        var x = V(1.0, 2.0, 3.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void LaplacianKernel_DifferentVectors_ReturnsLessThanOne()
    {
        var kernel = new LaplacianKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        var result = kernel.Calculate(x, y);
        Assert.True(result > 0);
        Assert.True(result < 1.0);
    }

    [Fact]
    public void LaplacianKernel_IsSymmetric()
    {
        var kernel = new LaplacianKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Exponential Kernel Tests

    [Fact]
    public void ExponentialKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new ExponentialKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ExponentialKernel_IsSymmetric()
    {
        var kernel = new ExponentialKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Matern Kernel Tests

    [Fact]
    public void MaternKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new MaternKernel<double>(1.0, 1.5);
        var x = V(1.0, 2.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MaternKernel_IsSymmetric()
    {
        var kernel = new MaternKernel<double>(1.0, 2.5);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Cauchy Kernel Tests

    [Fact]
    public void CauchyKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new CauchyKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CauchyKernel_AlwaysPositive()
    {
        var kernel = new CauchyKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(10.0, 20.0);
        var result = kernel.Calculate(x, y);
        Assert.True(result > 0);
    }

    #endregion

    #region RationalQuadratic Kernel Tests

    [Fact]
    public void RationalQuadraticKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new RationalQuadraticKernel<double>(1.0, 1.0);
        var x = V(1.0, 2.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void RationalQuadraticKernel_IsSymmetric()
    {
        var kernel = new RationalQuadraticKernel<double>(1.0, 2.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Cosine Kernel Tests

    [Fact]
    public void CosineKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new CosineKernel<double>();
        var x = V(1.0, 2.0, 3.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void CosineKernel_OrthogonalVectors_ReturnsZero()
    {
        var kernel = new CosineKernel<double>();
        var x = V(1.0, 0.0);
        var y = V(0.0, 1.0);
        var result = kernel.Calculate(x, y);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CosineKernel_IsSymmetric()
    {
        var kernel = new CosineKernel<double>();
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region WhiteNoise Kernel Tests

    [Fact]
    public void WhiteNoiseKernel_IdenticalVectors_ReturnsVariance()
    {
        var kernel = new WhiteNoiseKernel<double>(2.0);
        var x = V(1.0, 2.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void WhiteNoiseKernel_DifferentVectors_ReturnsZero()
    {
        var kernel = new WhiteNoiseKernel<double>(2.0);
        var x = V(1.0, 2.0);
        var y = V(1.0, 2.001);
        var result = kernel.Calculate(x, y);
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Constant Kernel Tests

    [Fact]
    public void ConstantKernel_AlwaysReturnsConstant()
    {
        var kernel = new ConstantKernel<double>(3.0);
        var x = V(1.0, 2.0);
        var y = V(100.0, 200.0);
        var result = kernel.Calculate(x, y);
        Assert.Equal(3.0, result, Tolerance);
    }

    #endregion

    #region DotProduct Kernel Tests

    [Fact]
    public void DotProductKernel_Calculate_ReturnsDotProduct()
    {
        var kernel = new DotProductKernel<double>();
        var x = V(1.0, 2.0, 3.0);
        var y = V(4.0, 5.0, 6.0);
        var result = kernel.Calculate(x, y);
        Assert.Equal(32.0, result, Tolerance);
    }

    #endregion

    #region Multiquadric Kernel Tests

    [Fact]
    public void MultiquadricKernel_IdenticalVectors_ReturnsConstant()
    {
        var kernel = new MultiquadricKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var result = kernel.Calculate(x, x);
        // sqrt(||x-x||^2 + c^2) = sqrt(0 + 1) = 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MultiquadricKernel_IsSymmetric()
    {
        var kernel = new MultiquadricKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region InverseMultiquadric Kernel Tests

    [Fact]
    public void InverseMultiquadricKernel_IdenticalVectors_ReturnsMax()
    {
        var kernel = new InverseMultiquadricKernel<double>(1.0);
        var x = V(1.0, 2.0);
        var self = kernel.Calculate(x, x);
        var y = V(5.0, 6.0);
        var other = kernel.Calculate(x, y);
        Assert.True(self >= other);
    }

    #endregion

    #region Helinger Kernel Tests

    [Fact]
    public void HellingerKernel_IdenticalVectors_ReturnsMax()
    {
        var kernel = new HellingerKernel<double>();
        var x = V(0.25, 0.25, 0.5);
        var result = kernel.Calculate(x, x);
        Assert.True(result > 0);
    }

    [Fact]
    public void HellingerKernel_IsSymmetric()
    {
        var kernel = new HellingerKernel<double>();
        var x = V(0.1, 0.4, 0.5);
        var y = V(0.3, 0.3, 0.4);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Tanimoto Kernel Tests

    [Fact]
    public void TanimotoKernel_IdenticalVectors_ReturnsOne()
    {
        var kernel = new TanimotoKernel<double>();
        var x = V(1.0, 2.0, 3.0);
        var result = kernel.Calculate(x, x);
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void TanimotoKernel_IsSymmetric()
    {
        var kernel = new TanimotoKernel<double>();
        var x = V(1.0, 2.0);
        var y = V(3.0, 4.0);
        Assert.Equal(kernel.Calculate(x, y), kernel.Calculate(y, x), Tolerance);
    }

    #endregion

    #region Comprehensive Symmetry and Positivity Tests

    [Fact]
    public void AllKernels_Symmetry_KxyEqualsKyx()
    {
        var kernels = new IKernelFunction<double>[]
        {
            new GaussianKernel<double>(1.0),
            new LinearKernel<double>(),
            new PolynomialKernel<double>(2, 1.0),
            new LaplacianKernel<double>(1.0),
            new ExponentialKernel<double>(1.0),
            new CauchyKernel<double>(1.0),
            new RationalQuadraticKernel<double>(1.0, 1.0),
            new CosineKernel<double>(),
            new ConstantKernel<double>(1.0),
            new DotProductKernel<double>(),
            new MultiquadricKernel<double>(1.0),
            new InverseMultiquadricKernel<double>(1.0),
            new HellingerKernel<double>(),
            new TanimotoKernel<double>(),
        };

        var x = V(1.0, 2.0, 3.0);
        var y = V(4.0, 5.0, 6.0);

        foreach (var kernel in kernels)
        {
            var kxy = kernel.Calculate(x, y);
            var kyx = kernel.Calculate(y, x);
            Assert.Equal(kxy, kyx, Tolerance);
        }
    }

    [Fact]
    public void AllKernels_SelfSimilarity_NoNaN()
    {
        var kernels = new IKernelFunction<double>[]
        {
            new GaussianKernel<double>(1.0),
            new LinearKernel<double>(),
            new PolynomialKernel<double>(2, 1.0),
            new LaplacianKernel<double>(1.0),
            new ExponentialKernel<double>(1.0),
            new CauchyKernel<double>(1.0),
            new RationalQuadraticKernel<double>(1.0, 1.0),
            new CosineKernel<double>(),
            new ConstantKernel<double>(1.0),
            new DotProductKernel<double>(),
            new MultiquadricKernel<double>(1.0),
            new InverseMultiquadricKernel<double>(1.0),
            new HellingerKernel<double>(),
            new TanimotoKernel<double>(),
        };

        var x = V(1.0, 2.0, 3.0);

        foreach (var kernel in kernels)
        {
            var result = kernel.Calculate(x, x);
            Assert.False(double.IsNaN(result), $"{kernel.GetType().Name} produced NaN");
            Assert.False(double.IsInfinity(result), $"{kernel.GetType().Name} produced Infinity");
        }
    }

    #endregion
}
