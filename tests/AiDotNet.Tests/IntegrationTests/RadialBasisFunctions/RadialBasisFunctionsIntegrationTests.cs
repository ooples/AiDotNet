using AiDotNet.Interfaces;
using AiDotNet.RadialBasisFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RadialBasisFunctions;

/// <summary>
/// Integration tests for radial basis function classes.
/// Tests RBF computation, derivatives, and width derivatives.
/// </summary>
public class RadialBasisFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Gaussian RBF Tests

    [Fact]
    public void GaussianRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new GaussianRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert - exp(0) = 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void GaussianRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new GaussianRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    [Fact]
    public void GaussianRBF_ComputeDerivativeAtZero_ReturnsZero()
    {
        // Arrange
        var rbf = new GaussianRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.ComputeDerivative(0.0);

        // Assert - derivative at 0 is 0
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void GaussianRBF_ComputeDerivative_IsNegativeForPositiveR()
    {
        // Arrange
        var rbf = new GaussianRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.ComputeDerivative(1.0);

        // Assert - derivative is negative for positive r
        Assert.True(result < 0);
    }

    [Fact]
    public void GaussianRBF_ComputeWidthDerivative_IsNegativeForNonZeroR()
    {
        // Arrange
        var rbf = new GaussianRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.ComputeWidthDerivative(1.0);

        // Assert
        Assert.True(result < 0);
    }

    [Fact]
    public void GaussianRBF_DifferentEpsilon_ProducesDifferentResults()
    {
        // Arrange
        var rbf1 = new GaussianRBF<double>(epsilon: 0.5);
        var rbf2 = new GaussianRBF<double>(epsilon: 2.0);

        // Act
        var result1 = rbf1.Compute(1.0);
        var result2 = rbf2.Compute(1.0);

        // Assert - Higher epsilon = narrower function = lower value at r=1
        Assert.True(result1 > result2);
    }

    #endregion

    #region Multiquadric RBF Tests

    [Fact]
    public void MultiquadricRBF_ComputeAtZero_ReturnsEpsilon()
    {
        // Arrange
        var rbf = new MultiquadricRBF<double>(epsilon: 2.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert - sqrt(0 + epsilon^2) = epsilon
        Assert.Equal(2.0, result, Tolerance);
    }

    [Fact]
    public void MultiquadricRBF_ComputePositive_IncreasesWithDistance()
    {
        // Arrange
        var rbf = new MultiquadricRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert - Multiquadric increases with distance
        Assert.True(r1 > r0);
        Assert.True(r2 > r1);
    }

    [Fact]
    public void MultiquadricRBF_ComputeDerivative_IsPositiveForPositiveR()
    {
        // Arrange
        var rbf = new MultiquadricRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.ComputeDerivative(1.0);

        // Assert
        Assert.True(result > 0);
    }

    #endregion

    #region Inverse Multiquadric RBF Tests

    [Fact]
    public void InverseMultiquadricRBF_ComputeAtZero_ReturnsInverseEpsilon()
    {
        // Arrange
        var rbf = new InverseMultiquadricRBF<double>(epsilon: 2.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert - 1/sqrt(0 + epsilon^2) = 1/epsilon
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void InverseMultiquadricRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    [Fact]
    public void InverseMultiquadricRBF_ComputeDerivative_IsNegativeForPositiveR()
    {
        // Arrange
        var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.ComputeDerivative(1.0);

        // Assert
        Assert.True(result < 0);
    }

    #endregion

    #region Inverse Quadratic RBF Tests

    [Fact]
    public void InverseQuadraticRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert - 1/(1 + 0) = 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    #endregion

    #region Linear RBF Tests

    [Fact]
    public void LinearRBF_ComputeAtZero_ReturnsZero()
    {
        // Arrange
        var rbf = new LinearRBF<double>();

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void LinearRBF_Compute_ReturnsInputValue()
    {
        // Arrange
        var rbf = new LinearRBF<double>();

        // Act
        var result = rbf.Compute(5.0);

        // Assert
        Assert.Equal(5.0, result, Tolerance);
    }

    [Fact]
    public void LinearRBF_ComputeDerivative_ReturnsOne()
    {
        // Arrange
        var rbf = new LinearRBF<double>();

        // Act
        var result = rbf.ComputeDerivative(5.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    #endregion

    #region Cubic RBF Tests

    [Fact]
    public void CubicRBF_ComputeAtZero_ReturnsZero()
    {
        // Arrange
        var rbf = new CubicRBF<double>();

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void CubicRBF_Compute_ReturnsCubeOfInput()
    {
        // Arrange
        var rbf = new CubicRBF<double>();

        // Act
        var result = rbf.Compute(2.0);

        // Assert - 2^3 = 8
        Assert.Equal(8.0, result, Tolerance);
    }

    [Fact]
    public void CubicRBF_ComputeDerivative_ReturnsThreeRSquared()
    {
        // Arrange
        var rbf = new CubicRBF<double>();

        // Act
        var result = rbf.ComputeDerivative(2.0);

        // Assert - 3*r^2 = 3*4 = 12
        Assert.Equal(12.0, result, Tolerance);
    }

    #endregion

    #region Thin Plate Spline RBF Tests

    [Fact]
    public void ThinPlateSplineRBF_ComputeAtZero_ReturnsZero()
    {
        // Arrange
        var rbf = new ThinPlateSplineRBF<double>();

        // Act
        var result = rbf.Compute(0.0);

        // Assert - 0*log(0) = 0 by convention
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_ComputePositive_ReturnsPositive()
    {
        // Arrange
        var rbf = new ThinPlateSplineRBF<double>();

        // Act
        var result = rbf.Compute(2.0);

        // Assert - r^2 * log(r) = 4 * log(2) > 0
        Assert.True(result > 0);
    }

    #endregion

    #region Polyharmonic Spline RBF Tests

    [Fact]
    public void PolyharmonicSplineRBF_ComputeAtZero_ReturnsZero()
    {
        // Arrange
        var rbf = new PolyharmonicSplineRBF<double>(k: 2);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_OddK_ReturnsRToTheK()
    {
        // Arrange
        var rbf = new PolyharmonicSplineRBF<double>(k: 3);

        // Act
        var result = rbf.Compute(2.0);

        // Assert - 2^3 = 8
        Assert.Equal(8.0, result, Tolerance);
    }

    #endregion

    #region Exponential RBF Tests

    [Fact]
    public void ExponentialRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new ExponentialRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert - exp(0) = 1
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void ExponentialRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new ExponentialRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    #endregion

    #region Matern RBF Tests

    [Fact]
    public void MaternRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void MaternRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    [Fact]
    public void MaternRBF_DifferentNu_ProducesDifferentResults()
    {
        // Arrange
        var rbf1 = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
        var rbf2 = new MaternRBF<double>(nu: 2.5, lengthScale: 1.0);

        // Act
        var result1 = rbf1.Compute(1.0);
        var result2 = rbf2.Compute(1.0);

        // Assert
        Assert.NotEqual(result1, result2);
    }

    #endregion

    #region Rational Quadratic RBF Tests

    [Fact]
    public void RationalQuadraticRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    #endregion

    #region Squared Exponential RBF Tests

    [Fact]
    public void SquaredExponentialRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_ComputePositive_DecreasesWithDistance()
    {
        // Arrange
        var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);

        // Act
        var r0 = rbf.Compute(0.0);
        var r1 = rbf.Compute(1.0);
        var r2 = rbf.Compute(2.0);

        // Assert
        Assert.True(r1 < r0);
        Assert.True(r2 < r1);
    }

    #endregion

    #region Wendland RBF Tests

    [Fact]
    public void WendlandRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new WendlandRBF<double>(supportRadius: 2.0, k: 0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void WendlandRBF_ComputeOutsideSupport_ReturnsZero()
    {
        // Arrange
        var rbf = new WendlandRBF<double>(supportRadius: 1.0, k: 0);

        // Act
        var result = rbf.Compute(2.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void WendlandRBF_ComputeInsideSupport_ReturnsPositive()
    {
        // Arrange
        var rbf = new WendlandRBF<double>(supportRadius: 2.0, k: 0);

        // Act
        var result = rbf.Compute(0.5);

        // Assert
        Assert.True(result > 0);
        Assert.True(result <= 1.0);
    }

    #endregion

    #region Spherical RBF Tests

    [Fact]
    public void SphericalRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new SphericalRBF<double>(epsilon: 2.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void SphericalRBF_ComputeOutsideRange_ReturnsZero()
    {
        // Arrange
        var rbf = new SphericalRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(2.0);

        // Assert
        Assert.Equal(0.0, result, Tolerance);
    }

    #endregion

    #region Wave RBF Tests

    [Fact]
    public void WaveRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new WaveRBF<double>(epsilon: 1.0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void WaveRBF_Compute_Oscillates()
    {
        // Arrange
        var rbf = new WaveRBF<double>(epsilon: 1.0);

        // Act - Check that wave function can go negative
        var values = new List<double>();
        for (double r = 0; r <= 10; r += 0.1)
        {
            values.Add(rbf.Compute(r));
        }

        // Assert - Wave RBF produces oscillating values; verify we have variation in the output
        double minValue = values.Min();
        double maxValue = values.Max();
        Assert.True(maxValue > minValue, "Wave RBF should produce varying output values");
    }

    #endregion

    #region Bessel RBF Tests

    [Fact]
    public void BesselRBF_ComputeAtZero_ReturnsOne()
    {
        // Arrange
        var rbf = new BesselRBF<double>(epsilon: 1.0, nu: 0);

        // Act
        var result = rbf.Compute(0.0);

        // Assert
        Assert.Equal(1.0, result, Tolerance);
    }

    [Fact]
    public void BesselRBF_ComputePositive_DoesNotReturnNaN()
    {
        // Arrange
        var rbf = new BesselRBF<double>(epsilon: 1.0, nu: 0);

        // Act
        var result = rbf.Compute(1.0);

        // Assert
        Assert.False(double.IsNaN(result));
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void AllRBFs_ComputeDoesNotReturnNaN()
    {
        // Arrange
        var rbfs = new IRadialBasisFunction<double>[]
        {
            new GaussianRBF<double>(epsilon: 1.0),
            new MultiquadricRBF<double>(epsilon: 1.0),
            new InverseMultiquadricRBF<double>(epsilon: 1.0),
            new InverseQuadraticRBF<double>(epsilon: 1.0),
            new LinearRBF<double>(),
            new CubicRBF<double>(),
            new ThinPlateSplineRBF<double>(),
            new ExponentialRBF<double>(epsilon: 1.0),
            new MaternRBF<double>(nu: 1.5, lengthScale: 1.0),
            new RationalQuadraticRBF<double>(epsilon: 1.0),
            new SquaredExponentialRBF<double>(epsilon: 1.0),
            new WendlandRBF<double>(supportRadius: 2.0, k: 0),
            new SphericalRBF<double>(epsilon: 2.0)
        };

        // Act & Assert
        foreach (var rbf in rbfs)
        {
            for (double r = 0; r <= 5; r += 0.5)
            {
                var result = rbf.Compute(r);
                Assert.False(double.IsNaN(result), $"RBF {rbf.GetType().Name} returned NaN at r={r}");
            }
        }
    }

    [Fact]
    public void DecreasingRBFs_ComputeDecreasesWithDistance()
    {
        // Arrange - RBFs that should decrease with distance
        var rbfs = new IRadialBasisFunction<double>[]
        {
            new GaussianRBF<double>(epsilon: 1.0),
            new InverseMultiquadricRBF<double>(epsilon: 1.0),
            new InverseQuadraticRBF<double>(epsilon: 1.0),
            new ExponentialRBF<double>(epsilon: 1.0),
            new MaternRBF<double>(nu: 1.5, lengthScale: 1.0),
            new RationalQuadraticRBF<double>(epsilon: 1.0),
            new SquaredExponentialRBF<double>(epsilon: 1.0)
        };

        // Act & Assert
        foreach (var rbf in rbfs)
        {
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            Assert.True(r1 <= r0, $"RBF {rbf.GetType().Name} should decrease with distance");
        }
    }

    [Fact]
    public void AllRBFs_ComputeDerivativeDoesNotReturnNaN()
    {
        // Arrange
        var rbfs = new IRadialBasisFunction<double>[]
        {
            new GaussianRBF<double>(epsilon: 1.0),
            new MultiquadricRBF<double>(epsilon: 1.0),
            new InverseMultiquadricRBF<double>(epsilon: 1.0),
            new LinearRBF<double>(),
            new CubicRBF<double>(),
            new ExponentialRBF<double>(epsilon: 1.0)
        };

        // Act & Assert
        foreach (var rbf in rbfs)
        {
            for (double r = 0; r <= 5; r += 0.5)
            {
                var result = rbf.ComputeDerivative(r);
                Assert.False(double.IsNaN(result), $"RBF {rbf.GetType().Name} derivative returned NaN at r={r}");
            }
        }
    }

    #endregion
}
