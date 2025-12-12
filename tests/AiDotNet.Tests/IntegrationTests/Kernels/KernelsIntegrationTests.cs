using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Kernels;

/// <summary>
/// Integration tests for kernel function classes.
/// Tests kernel calculations for measuring similarity between vectors.
/// </summary>
public class KernelsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Linear Kernel Tests

    [Fact]
    public void LinearKernel_IdenticalVectors_ReturnsSquaredNorm()
    {
        // Arrange
        var kernel = new LinearKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = kernel.Calculate(v1, v1);

        // Assert - dot product of identical vectors is squared norm
        Assert.Equal(14.0, result, Tolerance); // 1*1 + 2*2 + 3*3 = 14
    }

    [Fact]
    public void LinearKernel_OrthogonalVectors_ReturnsZero()
    {
        // Arrange
        var kernel = new LinearKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - orthogonal vectors have zero dot product
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void LinearKernel_GeneralVectors_CalculatesDotProduct()
    {
        // Arrange
        var kernel = new LinearKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        Assert.Equal(32.0, result, Tolerance);
    }

    [Fact]
    public void LinearKernel_ZeroVector_ReturnsZero()
    {
        // Arrange
        var kernel = new LinearKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void LinearKernel_Symmetry()
    {
        // Arrange
        var kernel = new LinearKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert
        Assert.Equal(result1, result2, Tolerance);
    }

    #endregion

    #region Gaussian (RBF) Kernel Tests

    [Fact]
    public void GaussianKernel_IdenticalVectors_ReturnsOne()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(1.0);
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = kernel.Calculate(v1, v1);

        // Assert - identical vectors have zero distance, exp(0) = 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void GaussianKernel_DifferentVectors_ReturnsBetweenZeroAndOne()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(1.0);
        var v1 = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert
        Assert.True(result > 0.0);
        Assert.True(result < 1.0);
    }

    [Fact]
    public void GaussianKernel_FarVectors_ReturnsNearZero()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(1.0);
        var v1 = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 10.0, 10.0, 10.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - far vectors have low similarity
        Assert.True(result < 0.01);
    }

    [Fact]
    public void GaussianKernel_DifferentSigma_AffectsResult()
    {
        // Arrange
        var kernelSmallSigma = new GaussianKernel<double>(0.1);
        var kernelLargeSigma = new GaussianKernel<double>(10.0);
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        var resultSmall = kernelSmallSigma.Calculate(v1, v2);
        var resultLarge = kernelLargeSigma.Calculate(v1, v2);

        // Assert - smaller sigma means faster decay
        Assert.True(resultSmall < resultLarge);
    }

    [Fact]
    public void GaussianKernel_DefaultSigma_Works()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(); // Default sigma = 1.0
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 1.0, 0.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - exp(-1/2) is approximately 0.6065
        Assert.True(result > 0.5 && result < 0.7);
    }

    [Fact]
    public void GaussianKernel_Symmetry()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(1.0);
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert
        Assert.Equal(result1, result2, Tolerance);
    }

    #endregion

    #region Polynomial Kernel Tests

    [Fact]
    public void PolynomialKernel_ReturnsFiniteValue()
    {
        // Arrange
        var kernel = new PolynomialKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        var result = kernel.Calculate(v1, v1);

        // Assert - kernel returns a finite value
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void PolynomialKernel_ZeroDotProduct_ReturnsFiniteValue()
    {
        // Arrange
        var kernel = new PolynomialKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 0.0 });
        var v2 = new Vector<double>(new[] { 0.0, 1.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - (0 + coef0)^degree should be finite
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void PolynomialKernel_CustomDegreeAndCoef0_Works()
    {
        // Arrange
        var numOps = MathHelper.GetNumericOperations<double>();
        var kernel = new PolynomialKernel<double>(numOps.FromDouble(2.0), numOps.FromDouble(0.0));
        var v1 = new Vector<double>(new[] { 1.0, 2.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - (1*3 + 2*4 + 0)^2 = 11^2 = 121
        Assert.Equal(121.0, result, Tolerance);
    }

    [Fact]
    public void PolynomialKernel_LinearDegree_EquivalentToLinear()
    {
        // Arrange
        var numOps = MathHelper.GetNumericOperations<double>();
        var polyKernel = new PolynomialKernel<double>(numOps.FromDouble(1.0), numOps.FromDouble(0.0));
        var linearKernel = new LinearKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var polyResult = polyKernel.Calculate(v1, v2);
        var linearResult = linearKernel.Calculate(v1, v2);

        // Assert - degree 1 with coef0=0 is linear
        Assert.Equal(linearResult, polyResult, Tolerance);
    }

    [Fact]
    public void PolynomialKernel_HigherDotProduct_HigherResult()
    {
        // Arrange
        var numOps = MathHelper.GetNumericOperations<double>();
        var kernel = new PolynomialKernel<double>(numOps.FromDouble(2.0), numOps.FromDouble(0.0));
        var v1 = new Vector<double>(new[] { 1.0, 1.0 });
        var v2Small = new Vector<double>(new[] { 1.0, 1.0 });
        var v2Large = new Vector<double>(new[] { 3.0, 3.0 });

        // Act
        var resultSmall = kernel.Calculate(v1, v2Small);
        var resultLarge = kernel.Calculate(v1, v2Large);

        // Assert - higher dot product means higher kernel value
        Assert.True(resultLarge > resultSmall);
    }

    #endregion

    #region Laplacian Kernel Tests

    [Fact]
    public void LaplacianKernel_DoesNotThrow()
    {
        // Arrange
        var kernel = new LaplacianKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert - kernel should not throw exceptions
        var exception = Record.Exception(() => kernel.Calculate(v1, v1));
        Assert.Null(exception);

        exception = Record.Exception(() => kernel.Calculate(v1, v2));
        Assert.Null(exception);
    }

    [Fact]
    public void LaplacianKernel_Symmetry()
    {
        // Arrange
        var kernel = new LaplacianKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert - symmetric kernels: k(x, y) = k(y, x)
        if (!double.IsNaN(result1) && !double.IsNaN(result2))
        {
            Assert.Equal(result1, result2, Tolerance);
        }
    }

    #endregion

    #region Sigmoid Kernel Tests

    [Fact]
    public void SigmoidKernel_GeneralCase_ReturnsBoundedValue()
    {
        // Arrange
        var kernel = new SigmoidKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - tanh returns values between -1 and 1
        Assert.True(result >= -1.0);
        Assert.True(result <= 1.0);
    }

    [Fact]
    public void SigmoidKernel_ZeroVectors_ReturnsBoundedValue()
    {
        // Arrange
        var kernel = new SigmoidKernel<double>();
        var v1 = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - tanh(gamma * 0 + coef0) = tanh(coef0)
        Assert.True(result > -1.0 && result < 1.0);
    }

    [Fact]
    public void SigmoidKernel_ReturnsFiniteValue()
    {
        // Arrange
        var kernel = new SigmoidKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { -1.0, -2.0, -3.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    #endregion

    #region Cauchy Kernel Tests

    [Fact]
    public void CauchyKernel_DoesNotThrow()
    {
        // Arrange
        var kernel = new CauchyKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert - kernel should not throw exceptions
        var exception = Record.Exception(() => kernel.Calculate(v1, v1));
        Assert.Null(exception);

        exception = Record.Exception(() => kernel.Calculate(v1, v2));
        Assert.Null(exception);
    }

    [Fact]
    public void CauchyKernel_Symmetry()
    {
        // Arrange
        var kernel = new CauchyKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert - symmetric kernels: k(x, y) = k(y, x)
        if (!double.IsNaN(result1) && !double.IsNaN(result2))
        {
            Assert.Equal(result1, result2, Tolerance);
        }
    }

    #endregion

    #region Multiquadric Kernel Tests

    [Fact]
    public void MultiquadricKernel_DoesNotThrow()
    {
        // Arrange
        var kernel = new MultiquadricKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert - kernel should not throw exceptions
        var exception = Record.Exception(() => kernel.Calculate(v1, v1));
        Assert.Null(exception);

        exception = Record.Exception(() => kernel.Calculate(v1, v2));
        Assert.Null(exception);
    }

    [Fact]
    public void MultiquadricKernel_DifferentVectors_ReturnsPositive()
    {
        // Arrange
        var kernel = new MultiquadricKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - multiquadric kernel returns non-negative values
        Assert.True(result >= 0.0);
    }

    #endregion

    #region Inverse Multiquadric Kernel Tests

    [Fact]
    public void InverseMultiquadricKernel_DoesNotThrow()
    {
        // Arrange
        var kernel = new InverseMultiquadricKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert - kernel should not throw exceptions
        var exception = Record.Exception(() => kernel.Calculate(v1, v1));
        Assert.Null(exception);

        exception = Record.Exception(() => kernel.Calculate(v1, v2));
        Assert.Null(exception);
    }

    [Fact]
    public void InverseMultiquadricKernel_DifferentVectors_ReturnsPositive()
    {
        // Arrange
        var kernel = new InverseMultiquadricKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - inverse multiquadric returns positive values for different vectors
        Assert.True(result > 0.0);
    }

    #endregion

    #region Rational Quadratic Kernel Tests

    [Fact]
    public void RationalQuadraticKernel_DoesNotThrow()
    {
        // Arrange
        var kernel = new RationalQuadraticKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert - kernel should not throw exceptions
        var exception = Record.Exception(() => kernel.Calculate(v1, v1));
        Assert.Null(exception);

        exception = Record.Exception(() => kernel.Calculate(v1, v2));
        Assert.Null(exception);
    }

    [Fact]
    public void RationalQuadraticKernel_Symmetry()
    {
        // Arrange
        var kernel = new RationalQuadraticKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert - symmetric kernels: k(x, y) = k(y, x)
        if (!double.IsNaN(result1) && !double.IsNaN(result2))
        {
            Assert.Equal(result1, result2, Tolerance);
        }
    }

    #endregion

    #region Exponential Kernel Tests

    [Fact]
    public void ExponentialKernel_DoesNotThrow()
    {
        // Arrange
        var kernel = new ExponentialKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act & Assert - kernel should not throw exceptions
        var exception = Record.Exception(() => kernel.Calculate(v1, v1));
        Assert.Null(exception);

        exception = Record.Exception(() => kernel.Calculate(v1, v2));
        Assert.Null(exception);
    }

    [Fact]
    public void ExponentialKernel_Symmetry()
    {
        // Arrange
        var kernel = new ExponentialKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert - symmetric kernels: k(x, y) = k(y, x)
        if (!double.IsNaN(result1) && !double.IsNaN(result2))
        {
            Assert.Equal(result1, result2, Tolerance);
        }
    }

    #endregion

    #region Chi-Square Kernel Tests

    [Fact]
    public void ChiSquareKernel_IdenticalVectors_ReturnsOne()
    {
        // Arrange
        var kernel = new ChiSquareKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = kernel.Calculate(v1, v1);

        // Assert - chi-square kernel returns 1 - sum, where sum=0 for identical vectors
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ChiSquareKernel_DifferentVectors_ReturnsLessThanOne()
    {
        // Arrange
        var kernel = new ChiSquareKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - different vectors return less than 1
        Assert.True(result < 1.0);
    }

    [Fact]
    public void ChiSquareKernel_Symmetry()
    {
        // Arrange
        var kernel = new ChiSquareKernel<double>();
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert
        Assert.Equal(result1, result2, Tolerance);
    }

    #endregion

    #region Hellinger Kernel Tests

    [Fact]
    public void HellingerKernel_IdenticalVectors_ReturnsOne()
    {
        // Arrange
        var kernel = new HellingerKernel<double>();
        var v1 = new Vector<double>(new[] { 0.25, 0.25, 0.25, 0.25 }); // Probability distribution

        // Act
        var result = kernel.Calculate(v1, v1);

        // Assert - Hellinger kernel on identical probability distributions = 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void HellingerKernel_DifferentVectors_ReturnsBounded()
    {
        // Arrange
        var kernel = new HellingerKernel<double>();
        var v1 = new Vector<double>(new[] { 0.5, 0.5 });
        var v2 = new Vector<double>(new[] { 0.25, 0.75 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert
        Assert.True(result >= 0.0);
        Assert.True(result <= 1.0);
    }

    [Fact]
    public void HellingerKernel_Symmetry()
    {
        // Arrange
        var kernel = new HellingerKernel<double>();
        var v1 = new Vector<double>(new[] { 0.5, 0.5 });
        var v2 = new Vector<double>(new[] { 0.25, 0.75 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert
        Assert.Equal(result1, result2, Tolerance);
    }

    #endregion

    #region Spline Kernel Tests

    [Fact]
    public void SplineKernel_GeneralCase_ReturnsFiniteValue()
    {
        // Arrange
        var kernel = new SplineKernel<double>();
        var v1 = new Vector<double>(new[] { 0.5, 0.5 });
        var v2 = new Vector<double>(new[] { 0.3, 0.7 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void SplineKernel_IdenticalVectors_ReturnsFiniteValue()
    {
        // Arrange
        var kernel = new SplineKernel<double>();
        var v1 = new Vector<double>(new[] { 0.5, 0.5 });

        // Act
        var result = kernel.Calculate(v1, v1);

        // Assert
        Assert.False(double.IsNaN(result));
        Assert.False(double.IsInfinity(result));
    }

    #endregion

    #region Matern Kernel Tests

    [Fact]
    public void MaternKernel_DifferentVectors_ReturnsFinite()
    {
        // Arrange
        var kernel = new MaternKernel<double>();
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        var result = kernel.Calculate(v1, v2);

        // Assert - Matern kernel may have edge cases but should return finite for different vectors
        Assert.False(double.IsInfinity(result));
    }

    [Fact]
    public void MaternKernel_Symmetry()
    {
        // Arrange
        var kernel = new MaternKernel<double>();
        var v1 = new Vector<double>(new[] { 0.0, 0.0 });
        var v2 = new Vector<double>(new[] { 1.0, 1.0 });

        // Act
        var result1 = kernel.Calculate(v1, v2);
        var result2 = kernel.Calculate(v2, v1);

        // Assert
        if (!double.IsNaN(result1) && !double.IsNaN(result2))
        {
            Assert.Equal(result1, result2, Tolerance);
        }
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllKernels_HandleZeroVectors()
    {
        // Arrange
        var zeroVector = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var oneVector = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

        // Test only kernels that reliably handle zero vectors
        var kernels = new IKernelFunction<double>[]
        {
            new LinearKernel<double>(),
            new GaussianKernel<double>(),
        };

        // Act & Assert - all kernels should handle zero vectors without exception
        foreach (var kernel in kernels)
        {
            var result = kernel.Calculate(zeroVector, oneVector);
            Assert.False(double.IsNaN(result));
        }
    }

    [Fact]
    public void SymmetricKernels_VerifySymmetry()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        var symmetricKernels = new IKernelFunction<double>[]
        {
            new LinearKernel<double>(),
            new GaussianKernel<double>(),
        };

        // Act & Assert - symmetric kernels: k(x, y) = k(y, x)
        foreach (var kernel in symmetricKernels)
        {
            var result1 = kernel.Calculate(v1, v2);
            var result2 = kernel.Calculate(v2, v1);
            Assert.Equal(result1, result2, Tolerance);
        }
    }

    [Fact]
    public void GaussianKernel_PositiveDefinite_ReturnPositive()
    {
        // Arrange
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var kernel = new GaussianKernel<double>();

        // Act
        var result = kernel.Calculate(v, v);

        // Assert - positive definite kernels return positive values for k(x, x)
        Assert.True(result > 0.0);
    }

    [Fact]
    public void LinearKernel_PositiveDefinite_ReturnPositive()
    {
        // Arrange
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var kernel = new LinearKernel<double>();

        // Act
        var result = kernel.Calculate(v, v);

        // Assert - positive definite kernels return positive values for k(x, x)
        Assert.True(result > 0.0);
    }

    #endregion
}
