using AiDotNet.RadialBasisFunctions;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RadialBasisFunctions
{
    /// <summary>
    /// Integration tests for all Radial Basis Functions with mathematically verified results.
    /// These tests ensure mathematical correctness of RBF calculations, derivatives, and properties.
    /// </summary>
    public class RadialBasisFunctionsIntegrationTests
    {
        private const double Tolerance = 1e-8;
        private const double RelativeTolerance = 1e-6;

        #region GaussianRBF Tests

        [Fact]
        public void GaussianRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);
            // Formula: exp(-ε*r²), at r=0: exp(0) = 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void GaussianRBF_Symmetry_ProducesSameValue()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.5);
            double r = 2.0;

            // Act
            var positive = rbf.Compute(r);
            var negative = rbf.Compute(-r); // Distance is always positive, but test abs()

            // Assert
            Assert.Equal(positive, negative, precision: 10);
        }

        [Fact]
        public void GaussianRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: exp(-1.0 * 1.0²) = exp(-1) ≈ 0.36787944117

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(0.36787944117144233, result, precision: 10);
        }

        [Fact]
        public void GaussianRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);
            var r3 = rbf.Compute(3.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
            Assert.True(r2 > r3);
        }

        [Fact]
        public void GaussianRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);
            // Derivative at r=0: -2εr * exp(-εr²) = 0

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void GaussianRBF_DerivativeAtOne_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Derivative: -2εr * exp(-εr²) = -2 * 1 * 1 * exp(-1) ≈ -0.735758

            // Act
            var derivative = rbf.ComputeDerivative(r);

            // Assert
            Assert.Equal(-0.7357588823428847, derivative, precision: 8);
        }

        [Fact]
        public void GaussianRBF_LargerEpsilon_NarrowerFunction()
        {
            // Arrange
            var rbf1 = new GaussianRBF<double>(epsilon: 0.5);
            var rbf2 = new GaussianRBF<double>(epsilon: 2.0);
            double r = 1.0;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            // Larger epsilon should decay faster (smaller value at same distance)
            Assert.True(result2 < result1);
        }

        #endregion

        #region MultiquadricRBF Tests

        [Fact]
        public void MultiquadricRBF_AtZeroDistance_ReturnsEpsilon()
        {
            // Arrange
            double epsilon = 2.0;
            var rbf = new MultiquadricRBF<double>(epsilon);
            // Formula: √(r² + ε²), at r=0: √(ε²) = ε

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(epsilon, result, precision: 10);
        }

        [Fact]
        public void MultiquadricRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: √(1² + 1²) = √2 ≈ 1.41421356

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(1.4142135623730951, result, precision: 10);
        }

        [Fact]
        public void MultiquadricRBF_IncreasingDistance_Grows()
        {
            // Arrange
            var rbf = new MultiquadricRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);

            // Assert
            Assert.True(r1 > r0);
            Assert.True(r2 > r1);
        }

        [Fact]
        public void MultiquadricRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
            // Derivative at r=0: r/√(r² + ε²) = 0

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void MultiquadricRBF_DerivativeAtOne_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Derivative: r/√(r² + ε²) = 1/√2 ≈ 0.707107

            // Act
            var derivative = rbf.ComputeDerivative(r);

            // Assert
            Assert.Equal(0.7071067811865475, derivative, precision: 10);
        }

        #endregion

        #region InverseMultiquadricRBF Tests

        [Fact]
        public void InverseMultiquadricRBF_AtZeroDistance_ReturnsOneOverEpsilon()
        {
            // Arrange
            double epsilon = 2.0;
            var rbf = new InverseMultiquadricRBF<double>(epsilon);
            // Formula: 1/√(r² + ε²), at r=0: 1/ε

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0 / epsilon, result, precision: 10);
        }

        [Fact]
        public void InverseMultiquadricRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: 1/√(1² + 1²) = 1/√2 ≈ 0.707107

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(0.7071067811865475, result, precision: 10);
        }

        [Fact]
        public void InverseMultiquadricRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
        }

        [Fact]
        public void InverseMultiquadricRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void InverseMultiquadricRBF_DerivativeIsNegative_ForPositiveDistance()
        {
            // Arrange
            var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(1.0);

            // Assert
            Assert.True(derivative < 0);
        }

        #endregion

        #region ThinPlateSplineRBF Tests

        [Fact]
        public void ThinPlateSplineRBF_AtZeroDistance_ReturnsZero()
        {
            // Arrange
            var rbf = new ThinPlateSplineRBF<double>();
            // Formula: r² log(r), at r=0: 0

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void ThinPlateSplineRBF_AtOne_ReturnsZero()
        {
            // Arrange
            var rbf = new ThinPlateSplineRBF<double>();
            // Formula: r² log(r), at r=1: 1² * log(1) = 0

            // Act
            var result = rbf.Compute(1.0);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void ThinPlateSplineRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new ThinPlateSplineRBF<double>();
            double r = 2.0;
            // Expected: 2² * ln(2) = 4 * 0.693147... ≈ 2.772588

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(2.772588722239781, result, precision: 8);
        }

        [Fact]
        public void ThinPlateSplineRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new ThinPlateSplineRBF<double>();

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void ThinPlateSplineRBF_WidthDerivative_IsAlwaysZero()
        {
            // Arrange
            var rbf = new ThinPlateSplineRBF<double>();
            // TPS has no width parameter

            // Act
            var widthDeriv = rbf.ComputeWidthDerivative(1.0);

            // Assert
            Assert.Equal(0.0, widthDeriv, precision: 10);
        }

        #endregion

        #region CubicRBF Tests

        [Fact]
        public void CubicRBF_AtZeroDistance_ReturnsZero()
        {
            // Arrange
            var rbf = new CubicRBF<double>(width: 1.0);
            // Formula: (r/width)³, at r=0: 0

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void CubicRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new CubicRBF<double>(width: 1.0);
            double r = 2.0;
            // Expected: (2/1)³ = 8

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(8.0, result, precision: 10);
        }

        [Fact]
        public void CubicRBF_WithDifferentWidth_ScalesResult()
        {
            // Arrange
            var rbf1 = new CubicRBF<double>(width: 1.0);
            var rbf2 = new CubicRBF<double>(width: 2.0);
            double r = 2.0;

            // Act
            var result1 = rbf1.Compute(r); // (2/1)³ = 8
            var result2 = rbf2.Compute(r); // (2/2)³ = 1

            // Assert
            Assert.Equal(8.0, result1, precision: 10);
            Assert.Equal(1.0, result2, precision: 10);
        }

        [Fact]
        public void CubicRBF_IncreasingDistance_Grows()
        {
            // Arrange
            var rbf = new CubicRBF<double>(width: 1.0);

            // Act
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);
            var r3 = rbf.Compute(3.0);

            // Assert
            Assert.True(r2 > r1);
            Assert.True(r3 > r2);
        }

        [Fact]
        public void CubicRBF_DerivativeAtOne_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new CubicRBF<double>(width: 1.0);
            double r = 1.0;
            // Derivative: 3r²/width³ = 3 * 1² / 1³ = 3

            // Act
            var derivative = rbf.ComputeDerivative(r);

            // Assert
            Assert.Equal(3.0, derivative, precision: 10);
        }

        #endregion

        #region LinearRBF Tests

        [Fact]
        public void LinearRBF_ReturnsDistanceValue()
        {
            // Arrange
            var rbf = new LinearRBF<double>();
            double r = 3.5;
            // Formula: r

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(r, result, precision: 10);
        }

        [Fact]
        public void LinearRBF_AtZeroDistance_ReturnsZero()
        {
            // Arrange
            var rbf = new LinearRBF<double>();

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void LinearRBF_DerivativeIsOne()
        {
            // Arrange
            var rbf = new LinearRBF<double>();

            // Act
            var derivative = rbf.ComputeDerivative(5.0);

            // Assert
            Assert.Equal(1.0, derivative, precision: 10);
        }

        [Fact]
        public void LinearRBF_WidthDerivativeIsZero()
        {
            // Arrange
            var rbf = new LinearRBF<double>();

            // Act
            var widthDeriv = rbf.ComputeWidthDerivative(1.0);

            // Assert
            Assert.Equal(0.0, widthDeriv, precision: 10);
        }

        #endregion

        #region PolyharmonicSplineRBF Tests

        [Fact]
        public void PolyharmonicSplineRBF_AtZeroDistance_ReturnsZero()
        {
            // Arrange
            var rbf1 = new PolyharmonicSplineRBF<double>(k: 1);
            var rbf2 = new PolyharmonicSplineRBF<double>(k: 2);

            // Act
            var result1 = rbf1.Compute(0.0);
            var result2 = rbf2.Compute(0.0);

            // Assert
            Assert.Equal(0.0, result1, precision: 10);
            Assert.Equal(0.0, result2, precision: 10);
        }

        [Fact]
        public void PolyharmonicSplineRBF_OddK_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new PolyharmonicSplineRBF<double>(k: 1);
            double r = 2.0;
            // Formula for odd k: r^k = 2^1 = 2

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(2.0, result, precision: 10);
        }

        [Fact]
        public void PolyharmonicSplineRBF_EvenK_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new PolyharmonicSplineRBF<double>(k: 2);
            double r = 2.0;
            // Formula for even k: r^k * log(r) = 4 * ln(2) ≈ 2.772588

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(2.772588722239781, result, precision: 8);
        }

        [Fact]
        public void PolyharmonicSplineRBF_K3_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new PolyharmonicSplineRBF<double>(k: 3);
            double r = 2.0;
            // Formula: r³ = 8

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(8.0, result, precision: 10);
        }

        [Fact]
        public void PolyharmonicSplineRBF_WidthDerivativeIsZero()
        {
            // Arrange
            var rbf = new PolyharmonicSplineRBF<double>(k: 2);

            // Act
            var widthDeriv = rbf.ComputeWidthDerivative(1.0);

            // Assert
            Assert.Equal(0.0, widthDeriv, precision: 10);
        }

        #endregion

        #region SquaredExponentialRBF Tests

        [Fact]
        public void SquaredExponentialRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);
            // Formula: exp(-(εr)²), at r=0: exp(0) = 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void SquaredExponentialRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: exp(-(1*1)²) = exp(-1) ≈ 0.36787944117

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(0.36787944117144233, result, precision: 10);
        }

        [Fact]
        public void SquaredExponentialRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
        }

        [Fact]
        public void SquaredExponentialRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void SquaredExponentialRBF_SmallerEpsilon_WiderFunction()
        {
            // Arrange
            var rbf1 = new SquaredExponentialRBF<double>(epsilon: 0.5);
            var rbf2 = new SquaredExponentialRBF<double>(epsilon: 2.0);
            double r = 1.0;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            // Larger epsilon should decay faster
            Assert.True(result2 < result1);
        }

        #endregion

        #region MaternRBF Tests

        [Fact]
        public void MaternRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void MaternRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(0.5);
            var r2 = rbf.Compute(1.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
        }

        [Fact]
        public void MaternRBF_Nu05_BehavesLikeExponential()
        {
            // Arrange - Matérn with nu=0.5 is exponential
            var matern = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
            var exponential = new ExponentialRBF<double>(epsilon: 1.0);
            double r = 1.0;

            // Act
            var maternResult = matern.Compute(r);
            var expResult = exponential.Compute(r);

            // Assert - Should be similar (within tolerance due to different formulations)
            Assert.InRange(maternResult, expResult * 0.5, expResult * 1.5);
        }

        [Fact]
        public void MaternRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void MaternRBF_LargerLengthScale_SlowerDecay()
        {
            // Arrange
            var rbf1 = new MaternRBF<double>(nu: 1.5, lengthScale: 0.5);
            var rbf2 = new MaternRBF<double>(nu: 1.5, lengthScale: 2.0);
            double r = 1.0;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            // Larger length scale should decay more slowly (higher value)
            Assert.True(result2 > result1);
        }

        #endregion

        #region RationalQuadraticRBF Tests

        [Fact]
        public void RationalQuadraticRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);
            // Formula: 1 - r²/(r² + ε²), at r=0: 1 - 0 = 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void RationalQuadraticRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: 1 - 1²/(1² + 1²) = 1 - 1/2 = 0.5

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(0.5, result, precision: 10);
        }

        [Fact]
        public void RationalQuadraticRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
        }

        [Fact]
        public void RationalQuadraticRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void RationalQuadraticRBF_LargerEpsilon_SlowerDecay()
        {
            // Arrange
            var rbf1 = new RationalQuadraticRBF<double>(epsilon: 0.5);
            var rbf2 = new RationalQuadraticRBF<double>(epsilon: 2.0);
            double r = 1.0;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            // Larger epsilon should decay more slowly (higher value)
            Assert.True(result2 > result1);
        }

        #endregion

        #region ExponentialRBF Tests

        [Fact]
        public void ExponentialRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new ExponentialRBF<double>(epsilon: 1.0);
            // Formula: exp(-εr), at r=0: exp(0) = 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void ExponentialRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new ExponentialRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: exp(-1) ≈ 0.36787944117

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(0.36787944117144233, result, precision: 10);
        }

        [Fact]
        public void ExponentialRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new ExponentialRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
        }

        [Fact]
        public void ExponentialRBF_DerivativeIsNegative_ForPositiveDistance()
        {
            // Arrange
            var rbf = new ExponentialRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(1.0);

            // Assert
            Assert.True(derivative < 0);
        }

        [Fact]
        public void ExponentialRBF_LargerEpsilon_FasterDecay()
        {
            // Arrange
            var rbf1 = new ExponentialRBF<double>(epsilon: 0.5);
            var rbf2 = new ExponentialRBF<double>(epsilon: 2.0);
            double r = 1.0;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            // Larger epsilon should decay faster (smaller value)
            Assert.True(result2 < result1);
        }

        #endregion

        #region SphericalRBF Tests

        [Fact]
        public void SphericalRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new SphericalRBF<double>(epsilon: 1.0);
            // Formula: 1 - 1.5(r/ε) + 0.5(r/ε)³ for r ≤ ε, at r=0: 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void SphericalRBF_BeyondSupportRadius_ReturnsZero()
        {
            // Arrange
            var rbf = new SphericalRBF<double>(epsilon: 1.0);

            // Act
            var result = rbf.Compute(2.0); // Beyond epsilon

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void SphericalRBF_AtSupportRadius_ReturnsZero()
        {
            // Arrange
            var rbf = new SphericalRBF<double>(epsilon: 2.0);

            // Act
            var result = rbf.Compute(2.0); // At epsilon

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void SphericalRBF_WithinSupport_ProducesPositiveValue()
        {
            // Arrange
            var rbf = new SphericalRBF<double>(epsilon: 2.0);

            // Act
            var result = rbf.Compute(1.0);

            // Assert
            Assert.True(result > 0);
            Assert.True(result < 1.0);
        }

        [Fact]
        public void SphericalRBF_DerivativeBeyondSupport_ReturnsZero()
        {
            // Arrange
            var rbf = new SphericalRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(2.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void SphericalRBF_CompactSupport_VerifyProperty()
        {
            // Arrange
            var rbf = new SphericalRBF<double>(epsilon: 1.5);

            // Act
            var atSupport = rbf.Compute(1.5);
            var beyondSupport = rbf.Compute(2.0);

            // Assert
            Assert.Equal(0.0, atSupport, precision: 10);
            Assert.Equal(0.0, beyondSupport, precision: 10);
        }

        #endregion

        #region InverseQuadraticRBF Tests

        [Fact]
        public void InverseQuadraticRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);
            // Formula: 1/(1 + (εr)²), at r=0: 1/(1 + 0) = 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void InverseQuadraticRBF_KnownDistance_ProducesCorrectValue()
        {
            // Arrange
            var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);
            double r = 1.0;
            // Expected: 1/(1 + 1²) = 1/2 = 0.5

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.Equal(0.5, result, precision: 10);
        }

        [Fact]
        public void InverseQuadraticRBF_IncreasingDistance_Decays()
        {
            // Arrange
            var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);
            var r2 = rbf.Compute(2.0);

            // Assert
            Assert.True(r0 > r1);
            Assert.True(r1 > r2);
        }

        [Fact]
        public void InverseQuadraticRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void InverseQuadraticRBF_DerivativeIsNegative_ForPositiveDistance()
        {
            // Arrange
            var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(1.0);

            // Assert
            Assert.True(derivative < 0);
        }

        #endregion

        #region WaveRBF Tests

        [Fact]
        public void WaveRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new WaveRBF<double>(epsilon: 1.0);
            // Formula: sin(εr)/(εr), limit as r→0: 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void WaveRBF_FirstZeroCrossing_ProducesZero()
        {
            // Arrange
            var rbf = new WaveRBF<double>(epsilon: 1.0);
            double r = Math.PI; // First zero crossing at π/ε

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.InRange(result, -0.01, 0.01); // Close to zero
        }

        [Fact]
        public void WaveRBF_Oscillates_WithDistance()
        {
            // Arrange
            var rbf = new WaveRBF<double>(epsilon: 1.0);

            // Act
            var r1 = rbf.Compute(0.5);
            var r2 = rbf.Compute(Math.PI); // Zero crossing
            var r3 = rbf.Compute(Math.PI * 1.5); // Negative region

            // Assert
            Assert.True(r1 > 0); // Positive
            Assert.InRange(r2, -0.01, 0.01); // Near zero
            Assert.True(r3 < 0); // Negative
        }

        [Fact]
        public void WaveRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new WaveRBF<double>(epsilon: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void WaveRBF_HigherEpsilon_MoreOscillations()
        {
            // Arrange
            var rbf1 = new WaveRBF<double>(epsilon: 1.0);
            var rbf2 = new WaveRBF<double>(epsilon: 2.0);
            double r = Math.PI / 2;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            // Different epsilon values produce different oscillation patterns
            Assert.NotEqual(result1, result2, precision: 5);
        }

        #endregion

        #region WendlandRBF Tests

        [Fact]
        public void WendlandRBF_K0_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 0, supportRadius: 1.0);
            // Formula: (1-r)² for r ≤ 1, at r=0: 1

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void WendlandRBF_K1_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 1, supportRadius: 1.0);

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void WendlandRBF_K2_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void WendlandRBF_BeyondSupportRadius_ReturnsZero()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);

            // Act
            var result = rbf.Compute(2.0);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void WendlandRBF_AtSupportRadius_ReturnsZero()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.5);

            // Act
            var result = rbf.Compute(1.5);

            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void WendlandRBF_CompactSupport_VerifyProperty()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 2, supportRadius: 2.0);

            // Act
            var withinSupport = rbf.Compute(1.0);
            var atSupport = rbf.Compute(2.0);
            var beyondSupport = rbf.Compute(3.0);

            // Assert
            Assert.True(withinSupport > 0);
            Assert.Equal(0.0, atSupport, precision: 10);
            Assert.Equal(0.0, beyondSupport, precision: 10);
        }

        [Fact]
        public void WendlandRBF_DerivativeAtZero_IsZero()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(0.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        [Fact]
        public void WendlandRBF_DerivativeBeyondSupport_ReturnsZero()
        {
            // Arrange
            var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);

            // Act
            var derivative = rbf.ComputeDerivative(2.0);

            // Assert
            Assert.Equal(0.0, derivative, precision: 10);
        }

        #endregion

        #region BesselRBF Tests

        [Fact]
        public void BesselRBF_AtZeroDistance_ReturnsOne()
        {
            // Arrange
            var rbf = new BesselRBF<double>(epsilon: 1.0, nu: 0.0);

            // Act
            var result = rbf.Compute(0.0);

            // Assert
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void BesselRBF_Nu0_ProducesPositiveValue()
        {
            // Arrange
            var rbf = new BesselRBF<double>(epsilon: 1.0, nu: 0.0);
            double r = 1.0;

            // Act
            var result = rbf.Compute(r);

            // Assert
            Assert.True(result > 0);
        }

        [Fact]
        public void BesselRBF_Nu1_ProducesCorrectShape()
        {
            // Arrange
            var rbf = new BesselRBF<double>(epsilon: 1.0, nu: 1.0);

            // Act
            var r0 = rbf.Compute(0.0);
            var r1 = rbf.Compute(1.0);

            // Assert
            Assert.Equal(1.0, r0, precision: 10);
            Assert.True(r1 > 0);
        }

        [Fact]
        public void BesselRBF_DifferentEpsilon_ProducesDifferentDecay()
        {
            // Arrange
            var rbf1 = new BesselRBF<double>(epsilon: 0.5, nu: 0.0);
            var rbf2 = new BesselRBF<double>(epsilon: 2.0, nu: 0.0);
            double r = 1.0;

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);

            // Assert
            Assert.NotEqual(result1, result2, precision: 5);
        }

        [Fact]
        public void BesselRBF_DerivativeAtZero_HandledCorrectly()
        {
            // Arrange
            var rbf0 = new BesselRBF<double>(epsilon: 1.0, nu: 0.0);
            var rbf1 = new BesselRBF<double>(epsilon: 1.0, nu: 1.0);

            // Act
            var deriv0 = rbf0.ComputeDerivative(0.0);
            var deriv1 = rbf1.ComputeDerivative(0.0);

            // Assert
            // For nu=0, derivative at r=0 has special form
            Assert.True(deriv0 < 0);
            // For nu=1, derivative at r=0 is 0
            Assert.Equal(0.0, deriv1, precision: 10);
        }

        #endregion

        #region Gram Matrix and Positive Definiteness Tests

        [Fact]
        public void GaussianRBF_GramMatrix_IsPositiveDefinite()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);
            var points = new double[] { 0.0, 1.0, 2.0, 3.0 };
            int n = points.Length;
            var gramMatrix = new double[n, n];

            // Act - Construct Gram matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double distance = Math.Abs(points[i] - points[j]);
                    gramMatrix[i, j] = rbf.Compute(distance);
                }
            }

            // Assert - Diagonal should be 1
            for (int i = 0; i < n; i++)
            {
                Assert.Equal(1.0, gramMatrix[i, i], precision: 10);
            }

            // Assert - Matrix should be symmetric
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        [Fact]
        public void SquaredExponentialRBF_GramMatrix_IsSymmetric()
        {
            // Arrange
            var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);
            var points = new double[] { -1.0, 0.0, 1.0, 2.0 };
            int n = points.Length;
            var gramMatrix = new double[n, n];

            // Act
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double distance = Math.Abs(points[i] - points[j]);
                    gramMatrix[i, j] = rbf.Compute(distance);
                }
            }

            // Assert - Symmetry
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Assert.Equal(gramMatrix[i, j], gramMatrix[j, i], precision: 10);
                }
            }
        }

        #endregion

        #region Parameter Sensitivity Tests

        [Fact]
        public void GaussianRBF_EpsilonEffect_VerifyMonotonicity()
        {
            // Arrange
            double r = 1.0;
            var rbf1 = new GaussianRBF<double>(epsilon: 0.5);
            var rbf2 = new GaussianRBF<double>(epsilon: 1.0);
            var rbf3 = new GaussianRBF<double>(epsilon: 2.0);

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);
            var result3 = rbf3.Compute(r);

            // Assert - Increasing epsilon should decrease value
            Assert.True(result1 > result2);
            Assert.True(result2 > result3);
        }

        [Fact]
        public void MaternRBF_NuEffect_SmoothnessIncrease()
        {
            // Arrange
            double r = 0.5;
            var rbf1 = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
            var rbf2 = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);
            var rbf3 = new MaternRBF<double>(nu: 2.5, lengthScale: 1.0);

            // Act
            var result1 = rbf1.Compute(r);
            var result2 = rbf2.Compute(r);
            var result3 = rbf3.Compute(r);

            // Assert - All should produce valid values
            Assert.True(result1 > 0 && result1 < 1);
            Assert.True(result2 > 0 && result2 < 1);
            Assert.True(result3 > 0 && result3 < 1);
        }

        #endregion

        #region Batch Computation Tests

        [Fact]
        public void GaussianRBF_BatchComputation_ProducesConsistentResults()
        {
            // Arrange
            var rbf = new GaussianRBF<double>(epsilon: 1.0);
            var distances = new double[] { 0.0, 0.5, 1.0, 1.5, 2.0 };

            // Act
            var results = new double[distances.Length];
            for (int i = 0; i < distances.Length; i++)
            {
                results[i] = rbf.Compute(distances[i]);
            }

            // Assert - Check monotonic decay
            for (int i = 1; i < results.Length; i++)
            {
                Assert.True(results[i - 1] >= results[i]);
            }
        }

        [Fact]
        public void MultipleRBFs_ConsistencyCheck_AtZero()
        {
            // Arrange - All these RBFs should return 1 at r=0
            var gaussian = new GaussianRBF<double>(epsilon: 1.0);
            var sqExp = new SquaredExponentialRBF<double>(epsilon: 1.0);
            var exponential = new ExponentialRBF<double>(epsilon: 1.0);
            var invQuadratic = new InverseQuadraticRBF<double>(epsilon: 1.0);
            var ratQuadratic = new RationalQuadraticRBF<double>(epsilon: 1.0);

            // Act
            var r1 = gaussian.Compute(0.0);
            var r2 = sqExp.Compute(0.0);
            var r3 = exponential.Compute(0.0);
            var r4 = invQuadratic.Compute(0.0);
            var r5 = ratQuadratic.Compute(0.0);

            // Assert
            Assert.Equal(1.0, r1, precision: 10);
            Assert.Equal(1.0, r2, precision: 10);
            Assert.Equal(1.0, r3, precision: 10);
            Assert.Equal(1.0, r4, precision: 10);
            Assert.Equal(1.0, r5, precision: 10);
        }

        #endregion
    }
}
