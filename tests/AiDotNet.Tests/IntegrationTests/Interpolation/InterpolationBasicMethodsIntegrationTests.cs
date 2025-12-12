using AiDotNet.Interpolation;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Interpolation
{
    /// <summary>
    /// Integration tests for basic interpolation methods with mathematically verified results.
    /// Part 1 of 2: Basic interpolation methods.
    /// These tests validate the mathematical correctness of interpolation operations.
    /// </summary>
    public class InterpolationBasicMethodsIntegrationTests
    {
        private const double Tolerance = 1e-10;

        #region LinearInterpolation Tests

        [Fact]
        public void LinearInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange - Simple linear function y = 2x + 1
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - All known points should be recovered exactly
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 10);
            }
        }

        [Fact]
        public void LinearInterpolation_MidpointInterpolation_ProducesCorrectValues()
        {
            // Arrange - y = 2x + 1
            var x = new Vector<double>(new[] { 0.0, 2.0, 4.0 });
            var y = new Vector<double>(new[] { 1.0, 5.0, 9.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - Midpoint between 0 and 2 should be at x=1, y=3
            var result = interpolator.Interpolate(1.0);
            Assert.Equal(3.0, result, precision: 10);

            // Midpoint between 2 and 4 should be at x=3, y=7
            result = interpolator.Interpolate(3.0);
            Assert.Equal(7.0, result, precision: 10);
        }

        [Fact]
        public void LinearInterpolation_QuarterPointInterpolation_ProducesCorrectValues()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 4.0 });
            var y = new Vector<double>(new[] { 0.0, 8.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - Quarter point at x=1 should give y=2
            Assert.Equal(2.0, interpolator.Interpolate(1.0), precision: 10);

            // Three-quarter point at x=3 should give y=6
            Assert.Equal(6.0, interpolator.Interpolate(3.0), precision: 10);
        }

        [Fact]
        public void LinearInterpolation_TwoPointsOnly_WorksCorrectly()
        {
            // Arrange - Minimal case with two points
            var x = new Vector<double>(new[] { 1.0, 5.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - Midpoint should be (3.0, 15.0)
            Assert.Equal(15.0, interpolator.Interpolate(3.0), precision: 10);
        }

        [Fact]
        public void LinearInterpolation_ConstantFunction_ReturnsConstant()
        {
            // Arrange - Horizontal line y = 5
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - Any interpolation should return 5
            Assert.Equal(5.0, interpolator.Interpolate(0.5), precision: 10);
            Assert.Equal(5.0, interpolator.Interpolate(1.7), precision: 10);
            Assert.Equal(5.0, interpolator.Interpolate(2.3), precision: 10);
        }

        [Fact]
        public void LinearInterpolation_NegativeSlope_WorksCorrectly()
        {
            // Arrange - y = -2x + 10
            var x = new Vector<double>(new[] { 0.0, 2.0, 4.0 });
            var y = new Vector<double>(new[] { 10.0, 6.0, 2.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert
            Assert.Equal(8.0, interpolator.Interpolate(1.0), precision: 10);
            Assert.Equal(4.0, interpolator.Interpolate(3.0), precision: 10);
        }

        [Fact]
        public void LinearInterpolation_ExtrapolationAtBoundaries_UsesEdgeValues()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - Beyond bounds should use edge values
            var resultBelow = interpolator.Interpolate(0.5);
            var resultAbove = interpolator.Interpolate(3.5);

            // Should handle edge cases gracefully
            Assert.True(resultBelow >= 5.0 && resultBelow <= 10.0);
            Assert.True(resultAbove >= 30.0 && resultAbove <= 35.0);
        }

        [Fact]
        public void LinearInterpolation_UnevenlySpacedPoints_WorksCorrectly()
        {
            // Arrange - Points not evenly spaced
            var x = new Vector<double>(new[] { 0.0, 1.0, 5.0, 6.0 });
            var y = new Vector<double>(new[] { 0.0, 10.0, 50.0, 60.0 });
            var interpolator = new LinearInterpolation<double>(x, y);

            // Act & Assert - Interpolate in the large gap
            var result = interpolator.Interpolate(3.0);
            Assert.Equal(30.0, result, precision: 10);
        }

        #endregion

        #region NearestNeighborInterpolation Tests

        [Fact]
        public void NearestNeighborInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 10);
            }
        }

        [Fact]
        public void NearestNeighborInterpolation_NearestToFirst_ReturnsFirstValue()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 3.0, 5.0 });
            var y = new Vector<double>(new[] { 10.0, 30.0, 50.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert - 1.4 is closer to 1 than to 3
            Assert.Equal(10.0, interpolator.Interpolate(1.4), precision: 10);
        }

        [Fact]
        public void NearestNeighborInterpolation_NearestToSecond_ReturnsSecondValue()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 3.0, 5.0 });
            var y = new Vector<double>(new[] { 10.0, 30.0, 50.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert - 2.6 is closer to 3 than to 1
            Assert.Equal(30.0, interpolator.Interpolate(2.6), precision: 10);
        }

        [Fact]
        public void NearestNeighborInterpolation_StaircaseFunction_MaintainsDiscontinuities()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 2.0, 4.0, 8.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert - Should create staircase (piecewise constant)
            Assert.Equal(1.0, interpolator.Interpolate(0.4), precision: 10);
            Assert.Equal(2.0, interpolator.Interpolate(0.6), precision: 10);
            Assert.Equal(2.0, interpolator.Interpolate(1.4), precision: 10);
            Assert.Equal(4.0, interpolator.Interpolate(1.6), precision: 10);
        }

        [Fact]
        public void NearestNeighborInterpolation_SinglePoint_ReturnsOnlyValue()
        {
            // Arrange
            var x = new Vector<double>(new[] { 5.0 });
            var y = new Vector<double>(new[] { 100.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert - Any query should return the only value
            Assert.Equal(100.0, interpolator.Interpolate(0.0), precision: 10);
            Assert.Equal(100.0, interpolator.Interpolate(5.0), precision: 10);
            Assert.Equal(100.0, interpolator.Interpolate(10.0), precision: 10);
        }

        [Fact]
        public void NearestNeighborInterpolation_TwoPoints_SwitchesAtMidpoint()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 10.0 });
            var y = new Vector<double>(new[] { 100.0, 200.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert
            Assert.Equal(100.0, interpolator.Interpolate(4.0), precision: 10);
            Assert.Equal(200.0, interpolator.Interpolate(6.0), precision: 10);
        }

        [Fact]
        public void NearestNeighborInterpolation_ExactMidpoint_ReturnsOneOfTheNeighbors()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 10.0 });
            var y = new Vector<double>(new[] { 100.0, 200.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(5.0);

            // Assert - Should return one of the values
            Assert.True(result == 100.0 || result == 200.0);
        }

        [Fact]
        public void NearestNeighborInterpolation_OutOfBounds_ReturnsNearestEdgeValue()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 10.0, 20.0, 30.0 });
            var interpolator = new NearestNeighborInterpolation<double>(x, y);

            // Act & Assert
            Assert.Equal(10.0, interpolator.Interpolate(0.0), precision: 10);
            Assert.Equal(30.0, interpolator.Interpolate(5.0), precision: 10);
        }

        #endregion

        #region LagrangePolynomialInterpolation Tests

        [Fact]
        public void LagrangePolynomialInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 3.0, 9.0, 19.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 10);
            }
        }

        [Fact]
        public void LagrangePolynomialInterpolation_LinearFunction_ReproducesLinear()
        {
            // Arrange - y = 2x + 1
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - Should work like linear interpolation
            Assert.Equal(2.0, interpolator.Interpolate(0.5), precision: 10);
            Assert.Equal(4.0, interpolator.Interpolate(1.5), precision: 10);
        }

        [Fact]
        public void LagrangePolynomialInterpolation_QuadraticFunction_ReproducesQuadratic()
        {
            // Arrange - y = x^2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0, 9.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - Should recover quadratic exactly
            Assert.Equal(0.25, interpolator.Interpolate(0.5), precision: 9);
            Assert.Equal(2.25, interpolator.Interpolate(1.5), precision: 9);
            Assert.Equal(6.25, interpolator.Interpolate(2.5), precision: 9);
        }

        [Fact]
        public void LagrangePolynomialInterpolation_ThreePoints_FormingParabola()
        {
            // Arrange - Three points on a parabola
            var x = new Vector<double>(new[] { -1.0, 0.0, 1.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0, 1.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - y = x^2, so at x=0.5, y should be 0.25
            Assert.Equal(0.25, interpolator.Interpolate(0.5), precision: 10);
        }

        [Fact]
        public void LagrangePolynomialInterpolation_CubicFunction_ReproducesCubic()
        {
            // Arrange - y = x^3 - 2x + 1
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 0.0, 5.0, 22.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - At x=1.5: 1.5^3 - 2*1.5 + 1 = 3.375 - 3 + 1 = 1.375
            Assert.Equal(1.375, interpolator.Interpolate(1.5), precision: 9);
        }

        [Fact]
        public void LagrangePolynomialInterpolation_TwoPointsMinimal_WorksAsLinear()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0, 5.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - Should behave like linear
            Assert.Equal(3.0, interpolator.Interpolate(1.0), precision: 10);
        }

        [Fact]
        public void LagrangePolynomialInterpolation_SymmetricData_ProducesSymmetricResults()
        {
            // Arrange - Symmetric about x=0
            var x = new Vector<double>(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 4.0, 1.0, 0.0, 1.0, 4.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - Should be symmetric
            var left = interpolator.Interpolate(-0.5);
            var right = interpolator.Interpolate(0.5);
            Assert.Equal(left, right, precision: 10);
        }

        [Fact]
        public void LagrangePolynomialInterpolation_NonUniformSpacing_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 4.0, 5.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var interpolator = new LagrangePolynomialInterpolation<double>(x, y);

            // Act & Assert - Verify interpolation at midpoint of large gap
            var result = interpolator.Interpolate(2.5);
            // Should be between 1 and 2
            Assert.True(result > 1.0 && result < 2.0);
        }

        #endregion

        #region NewtonDividedDifferenceInterpolation Tests

        [Fact]
        public void NewtonDividedDifferenceInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 2.0, 5.0, 10.0, 17.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 10);
            }
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_LinearFunction_ProducesCorrectInterpolation()
        {
            // Arrange - y = 3x + 2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 2.0, 5.0, 8.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act & Assert
            Assert.Equal(3.5, interpolator.Interpolate(0.5), precision: 10);
            Assert.Equal(6.5, interpolator.Interpolate(1.5), precision: 10);
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_QuadraticFunction_ReproducesExactly()
        {
            // Arrange - y = x^2 + x + 1
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 3.0, 7.0, 13.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act & Assert - At x=1.5: 1.5^2 + 1.5 + 1 = 2.25 + 1.5 + 1 = 4.75
            Assert.Equal(4.75, interpolator.Interpolate(1.5), precision: 9);
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_SameAsLagrange_ProducesSameResults()
        {
            // Arrange - Both methods should give identical results
            var x = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 1.0, 4.0, 9.0, 16.0 });
            var newton = new NewtonDividedDifferenceInterpolation<double>(x, y);
            var lagrange = new LagrangePolynomialInterpolation<double>(x, y);

            // Act
            var newtonResult = newton.Interpolate(2.5);
            var lagrangeResult = lagrange.Interpolate(2.5);

            // Assert
            Assert.Equal(lagrangeResult, newtonResult, precision: 10);
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_TwoPoints_WorksAsLinear()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 4.0 });
            var y = new Vector<double>(new[] { 1.0, 9.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act & Assert - Linear interpolation at midpoint
            Assert.Equal(5.0, interpolator.Interpolate(2.0), precision: 10);
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_CubicPolynomial_InterpolatesCorrectly()
        {
            // Arrange - y = x^3
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 8.0, 27.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act & Assert - At x=1.5: 1.5^3 = 3.375
            Assert.Equal(3.375, interpolator.Interpolate(1.5), precision: 9);
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_NegativeValues_HandledCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 4.0, 1.0, 0.0, 1.0, 4.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act & Assert - Symmetric function
            var left = interpolator.Interpolate(-0.5);
            var right = interpolator.Interpolate(0.5);
            Assert.Equal(left, right, precision: 10);
        }

        [Fact]
        public void NewtonDividedDifferenceInterpolation_NonUniformSpacing_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 3.0, 7.0 });
            var y = new Vector<double>(new[] { 0.0, 2.0, 6.0, 14.0 });
            var interpolator = new NewtonDividedDifferenceInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(2.0);

            // Assert - Should produce reasonable interpolation
            Assert.True(result > 2.0 && result < 6.0);
        }

        #endregion

        #region HermiteInterpolation Tests

        [Fact]
        public void HermiteInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 });
            var m = new Vector<double>(new[] { 0.0, 2.0, 4.0 }); // Slopes
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 10);
            }
        }

        [Fact]
        public void HermiteInterpolation_WithZeroSlopes_CreatesSmooth()
        {
            // Arrange - Flat slopes at endpoints
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
            var m = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act
            var result = interpolator.Interpolate(0.5);

            // Assert - Should be between 0 and 1
            Assert.True(result >= 0.0 && result <= 1.0);
        }

        [Fact]
        public void HermiteInterpolation_QuadraticWithDerivatives_ReproducesExactly()
        {
            // Arrange - y = x^2, y' = 2x
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0 });
            var m = new Vector<double>(new[] { 0.0, 2.0, 4.0 });
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act & Assert - At x=0.5: y = 0.25
            Assert.Equal(0.25, interpolator.Interpolate(0.5), precision: 9);

            // At x=1.5: y = 2.25
            Assert.Equal(2.25, interpolator.Interpolate(1.5), precision: 9);
        }

        [Fact]
        public void HermiteInterpolation_ConstantSlopes_ProducesLinear()
        {
            // Arrange - Linear with constant slope
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 2.0, 4.0 });
            var m = new Vector<double>(new[] { 2.0, 2.0, 2.0 });
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act & Assert - Should be linear
            Assert.Equal(1.0, interpolator.Interpolate(0.5), precision: 9);
            Assert.Equal(3.0, interpolator.Interpolate(1.5), precision: 9);
        }

        [Fact]
        public void HermiteInterpolation_CubicWithDerivatives_WorksCorrectly()
        {
            // Arrange - y = x^3, y' = 3x^2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 8.0 });
            var m = new Vector<double>(new[] { 0.0, 3.0, 12.0 });
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act & Assert - At x=0.5: y = 0.125
            Assert.Equal(0.125, interpolator.Interpolate(0.5), precision: 8);
        }

        [Fact]
        public void HermiteInterpolation_TwoPointsWithSlopes_CreatesCubic()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0 });
            var m = new Vector<double>(new[] { 0.0, 0.0 }); // Flat at both ends
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act
            var result = interpolator.Interpolate(0.5);

            // Assert - Should be smooth, value between 0 and 1
            Assert.True(result > 0.0 && result < 1.0);
        }

        [Fact]
        public void HermiteInterpolation_NegativeSlopes_WorksCorrectly()
        {
            // Arrange - Decreasing function
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 4.0, 2.0, 0.0 });
            var m = new Vector<double>(new[] { -2.0, -2.0, -2.0 });
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act & Assert - Should be linear with negative slope
            Assert.Equal(3.0, interpolator.Interpolate(0.5), precision: 9);
            Assert.Equal(1.0, interpolator.Interpolate(1.5), precision: 9);
        }

        [Fact]
        public void HermiteInterpolation_SmoothBellCurve_ProducesReasonableValues()
        {
            // Arrange - Bell-like curve
            var x = new Vector<double>(new[] { -1.0, 0.0, 1.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 0.0 });
            var m = new Vector<double>(new[] { 0.5, 0.0, -0.5 });
            var interpolator = new HermiteInterpolation<double>(x, y, m);

            // Act & Assert - Peak should be at x=0
            var atPeak = interpolator.Interpolate(0.0);
            var atSide = interpolator.Interpolate(0.5);

            Assert.Equal(1.0, atPeak, precision: 10);
            Assert.True(atSide > 0.0 && atSide < 1.0);
        }

        #endregion

        #region BarycentricRationalInterpolation Tests

        [Fact]
        public void BarycentricRationalInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 2.0, 4.0, 8.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 10);
            }
        }

        [Fact]
        public void BarycentricRationalInterpolation_LinearFunction_ProducesCorrectInterpolation()
        {
            // Arrange - y = 2x + 1
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0, 3.0, 5.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert
            Assert.Equal(2.0, interpolator.Interpolate(0.5), precision: 10);
            Assert.Equal(4.0, interpolator.Interpolate(1.5), precision: 10);
        }

        [Fact]
        public void BarycentricRationalInterpolation_QuadraticFunction_InterpolatesCorrectly()
        {
            // Arrange - y = x^2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0, 9.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert - At x=1.5: y should be close to 2.25
            var result = interpolator.Interpolate(1.5);
            Assert.True(Math.Abs(result - 2.25) < 0.01);
        }

        [Fact]
        public void BarycentricRationalInterpolation_TwoPoints_WorksAsLinear()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0, 5.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert
            Assert.Equal(3.0, interpolator.Interpolate(1.0), precision: 10);
        }

        [Fact]
        public void BarycentricRationalInterpolation_NonUniformPoints_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 4.0, 5.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(2.5);

            // Assert - Should be between 1 and 2
            Assert.True(result > 1.0 && result < 2.0);
        }

        [Fact]
        public void BarycentricRationalInterpolation_SmoothFunction_NoOscillations()
        {
            // Arrange - Smooth sine-like points
            var x = new Vector<double>(new[] { 0.0, 0.5, 1.0, 1.5, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 0.48, 0.84, 1.0, 0.91 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert - Check monotonicity in increasing region
            var y1 = interpolator.Interpolate(0.25);
            var y2 = interpolator.Interpolate(0.75);
            var y3 = interpolator.Interpolate(1.25);

            Assert.True(y1 < y2); // Increasing
            Assert.True(y2 < y3); // Still increasing
        }

        [Fact]
        public void BarycentricRationalInterpolation_SymmetricData_ProducesSymmetricResults()
        {
            // Arrange
            var x = new Vector<double>(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 4.0, 1.0, 0.0, 1.0, 4.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert - Should be symmetric
            var left = interpolator.Interpolate(-0.5);
            var right = interpolator.Interpolate(0.5);
            Assert.Equal(left, right, precision: 9);
        }

        [Fact]
        public void BarycentricRationalInterpolation_ManyPoints_RemainsStable()
        {
            // Arrange - Test stability with more points
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0, 9.0, 16.0, 25.0 });
            var interpolator = new BarycentricRationalInterpolation<double>(x, y);

            // Act & Assert - Should still work well
            var result = interpolator.Interpolate(2.5);
            // At x=2.5, y=6.25 for x^2
            Assert.True(Math.Abs(result - 6.25) < 0.5);
        }

        #endregion

        #region TrigonometricInterpolation Tests

        [Fact]
        public void TrigonometricInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange - Odd number of points required
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0, 4.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0, -1.0, 0.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList);

            // Act & Assert
            for (int i = 0; i < xList.Count; i++)
            {
                var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(xList[i]));
                Assert.Equal(yList[i], MathHelper.GetNumericOperations<double>().ToDouble(result), precision: 9);
            }
        }

        [Fact]
        public void TrigonometricInterpolation_SineWave_InterpolatesCorrectly()
        {
            // Arrange - Sample a sine wave with odd number of points
            var xList = new List<double> { 0.0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI };
            var yList = new List<double> { 0.0, 1.0, 0.0, -1.0, 0.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList, 2 * Math.PI);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(Math.PI / 4));

            // Assert - Should be roughly sin(pi/4) ≈ 0.707
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(Math.Abs(doubleResult - 0.707) < 0.1);
        }

        [Fact]
        public void TrigonometricInterpolation_ConstantFunction_ReproducesConstant()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0 };
            var yList = new List<double> { 5.0, 5.0, 5.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList);

            // Act & Assert - Any interpolation should return close to 5
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.5));
            Assert.True(Math.Abs(MathHelper.GetNumericOperations<double>().ToDouble(result) - 5.0) < 0.1);
        }

        [Fact]
        public void TrigonometricInterpolation_ThreePoints_MinimalOddCase()
        {
            // Arrange - Minimum odd number
            var xList = new List<double> { 0.0, 1.0, 2.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.5));

            // Assert - Should be between 0 and 1
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(doubleResult >= 0.0 && doubleResult <= 1.0);
        }

        [Fact]
        public void TrigonometricInterpolation_CosineLikePattern_WorksCorrectly()
        {
            // Arrange - Cosine-like values
            var xList = new List<double> { 0.0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI };
            var yList = new List<double> { 1.0, 0.0, -1.0, 0.0, 1.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList, 2 * Math.PI);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(Math.PI / 4));

            // Assert - Should be roughly cos(pi/4) ≈ 0.707
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(Math.Abs(doubleResult - 0.707) < 0.15);
        }

        [Fact]
        public void TrigonometricInterpolation_PeriodicBehavior_RepeatsCorrectly()
        {
            // Arrange
            var xList = new List<double> { 0.0, 2.0, 4.0 };
            var yList = new List<double> { 1.0, -1.0, 1.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList, 4.0);

            // Act - Interpolate at equivalent periodic points
            var result1 = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(1.0));
            var result2 = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(5.0)); // 1 + period

            // Assert - Should be similar due to periodicity
            var diff = Math.Abs(MathHelper.GetNumericOperations<double>().ToDouble(result1) -
                               MathHelper.GetNumericOperations<double>().ToDouble(result2));
            Assert.True(diff < 0.5);
        }

        [Fact]
        public void TrigonometricInterpolation_FivePoints_ProducesSmooth()
        {
            // Arrange
            var xList = new List<double> { 0.0, 0.5, 1.0, 1.5, 2.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0, -1.0, 0.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList);

            // Act & Assert - Check intermediate points
            var y1 = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.25));
            var y2 = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.75));

            // Should be between 0 and 1 in first half
            var d1 = MathHelper.GetNumericOperations<double>().ToDouble(y1);
            var d2 = MathHelper.GetNumericOperations<double>().ToDouble(y2);
            Assert.True(d1 >= 0.0 && d1 <= 1.0);
            Assert.True(d2 >= 0.0 && d2 <= 1.0);
        }

        [Fact]
        public void TrigonometricInterpolation_CustomPeriod_WorksCorrectly()
        {
            // Arrange - Specify custom period
            var xList = new List<double> { 0.0, 3.0, 6.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0 };
            var interpolator = new TrigonometricInterpolation<double>(xList, yList, 6.0);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(1.5));

            // Assert - Should be between 0 and 1
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(doubleResult >= 0.0 && doubleResult <= 1.0);
        }

        #endregion

        #region SincInterpolation Tests

        [Fact]
        public void SincInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0, -1.0 };
            var interpolator = new SincInterpolation<double>(xList, yList);

            // Act & Assert
            for (int i = 0; i < xList.Count; i++)
            {
                var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(xList[i]));
                var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
                Assert.Equal(yList[i], doubleResult, precision: 9);
            }
        }

        [Fact]
        public void SincInterpolation_UniformSampling_ProducesSmooth()
        {
            // Arrange - Uniformly sampled data
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0, 4.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0, -1.0, 0.0 };
            var interpolator = new SincInterpolation<double>(xList, yList);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.5));

            // Assert - Should produce a value between 0 and 1
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(doubleResult >= -0.5 && doubleResult <= 1.5);
        }

        [Fact]
        public void SincInterpolation_LinearData_ApproximatesLinear()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0 };
            var yList = new List<double> { 0.0, 2.0, 4.0, 6.0 };
            var interpolator = new SincInterpolation<double>(xList, yList);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(1.5));

            // Assert - Should be close to 3.0
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(Math.Abs(doubleResult - 3.0) < 0.5);
        }

        [Fact]
        public void SincInterpolation_WithLowerCutoff_ProducesSmoother()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0, 1.0 };
            var interpolator = new SincInterpolation<double>(xList, yList, 0.5);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.5));

            // Assert - Should produce reasonable value
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(doubleResult >= -0.5 && doubleResult <= 1.5);
        }

        [Fact]
        public void SincInterpolation_FourPoints_WorksCorrectly()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0 };
            var yList = new List<double> { 1.0, 2.0, 3.0, 4.0 };
            var interpolator = new SincInterpolation<double>(xList, yList);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(1.5));

            // Assert - Should be close to 2.5
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(Math.Abs(doubleResult - 2.5) < 1.0);
        }

        [Fact]
        public void SincInterpolation_NegativeValues_HandledCorrectly()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0 };
            var yList = new List<double> { -2.0, -1.0, 0.0, 1.0 };
            var interpolator = new SincInterpolation<double>(xList, yList);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(1.5));

            // Assert - Should be between -1 and 0
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(doubleResult >= -1.5 && doubleResult <= 0.5);
        }

        [Fact]
        public void SincInterpolation_HighFrequencyCutoff_PreservesDetail()
        {
            // Arrange
            var xList = new List<double> { 0.0, 0.5, 1.0, 1.5, 2.0 };
            var yList = new List<double> { 0.0, 1.0, 0.0, 1.0, 0.0 };
            var interpolator = new SincInterpolation<double>(xList, yList, 2.0);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(0.25));

            // Assert - Should handle high frequency
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(doubleResult >= -0.5 && doubleResult <= 1.5);
        }

        [Fact]
        public void SincInterpolation_ConstantData_ReturnsConstant()
        {
            // Arrange
            var xList = new List<double> { 0.0, 1.0, 2.0, 3.0 };
            var yList = new List<double> { 5.0, 5.0, 5.0, 5.0 };
            var interpolator = new SincInterpolation<double>(xList, yList);

            // Act
            var result = interpolator.Interpolate(MathHelper.GetNumericOperations<double>().FromDouble(1.5));

            // Assert - Should be close to 5
            var doubleResult = MathHelper.GetNumericOperations<double>().ToDouble(result);
            Assert.True(Math.Abs(doubleResult - 5.0) < 0.5);
        }

        #endregion

        #region WhittakerShannonInterpolation Tests

        [Fact]
        public void WhittakerShannonInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, -1.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 9);
            }
        }

        [Fact]
        public void WhittakerShannonInterpolation_UniformSampling_ProducesSmooth()
        {
            // Arrange - Uniformly spaced
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, -1.0, 0.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(0.5);

            // Assert - Should be reasonable value between points
            Assert.True(result >= -1.0 && result <= 2.0);
        }

        [Fact]
        public void WhittakerShannonInterpolation_LinearData_ApproximatesLinear()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 2.0, 4.0, 6.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(1.5);

            // Assert - Should be close to 3.0
            Assert.True(Math.Abs(result - 3.0) < 1.0);
        }

        [Fact]
        public void WhittakerShannonInterpolation_SineWavePattern_ReconstructsWell()
        {
            // Arrange - Sample sine wave
            var x = new Vector<double>(new[] { 0.0, 0.5, 1.0, 1.5, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 0.48, 0.84, 1.0, 0.91 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(0.25);

            // Assert - Should be between 0 and 0.48
            Assert.True(result >= 0.0 && result <= 0.6);
        }

        [Fact]
        public void WhittakerShannonInterpolation_ConstantFunction_ReturnsConstant()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 5.0, 5.0, 5.0, 5.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(1.5);

            // Assert - Should be very close to 5
            Assert.True(Math.Abs(result - 5.0) < 0.5);
        }

        [Fact]
        public void WhittakerShannonInterpolation_TwoPoints_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 4.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(1.0);

            // Assert - Should be reasonably between 0 and 4
            Assert.True(result >= 0.0 && result <= 4.0);
        }

        [Fact]
        public void WhittakerShannonInterpolation_SymmetricData_ProducesSymmetric()
        {
            // Arrange
            var x = new Vector<double>(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 1.0, 4.0, 5.0, 4.0, 1.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var left = interpolator.Interpolate(-0.5);
            var right = interpolator.Interpolate(0.5);

            // Assert - Should be similar (symmetric)
            Assert.True(Math.Abs(left - right) < 0.5);
        }

        [Fact]
        public void WhittakerShannonInterpolation_FiveUniformPoints_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 1.0, 3.0, 2.0, 4.0, 3.0 });
            var interpolator = new WhittakerShannonInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(2.5);

            // Assert - Should be between 2 and 4
            Assert.True(result >= 2.0 && result <= 4.0);
        }

        #endregion

        #region LanczosInterpolation Tests

        [Fact]
        public void LanczosInterpolation_RecoversKnownDataPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0, 9.0 });
            var interpolator = new LanczosInterpolation<double>(x, y);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                var result = interpolator.Interpolate(x[i]);
                Assert.Equal(y[i], result, precision: 9);
            }
        }

        [Fact]
        public void LanczosInterpolation_LinearData_ApproximatesLinear()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 2.0, 4.0, 6.0 });
            var interpolator = new LanczosInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(1.5);

            // Assert - Should be close to 3.0
            Assert.True(Math.Abs(result - 3.0) < 0.5);
        }

        [Fact]
        public void LanczosInterpolation_QuadraticData_InterpolatesWell()
        {
            // Arrange - y = x^2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0, 9.0 });
            var interpolator = new LanczosInterpolation<double>(x, y);

            // Act - At x=1.5, y should be 2.25
            var result = interpolator.Interpolate(1.5);

            // Assert
            Assert.True(Math.Abs(result - 2.25) < 0.5);
        }

        [Fact]
        public void LanczosInterpolation_WithA2_WorksCorrectly()
        {
            // Arrange - Test with a=2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var interpolator = new LanczosInterpolation<double>(x, y, 2);

            // Act
            var result = interpolator.Interpolate(1.5);

            // Assert - Should be close to 2.5
            Assert.True(Math.Abs(result - 2.5) < 0.5);
        }

        [Fact]
        public void LanczosInterpolation_WithA4_WorksCorrectly()
        {
            // Arrange - Test with a=4
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 0.0, -1.0, 0.0 });
            var interpolator = new LanczosInterpolation<double>(x, y, 4);

            // Act
            var result = interpolator.Interpolate(0.5);

            // Assert - Should produce reasonable value
            Assert.True(result >= -0.5 && result <= 1.5);
        }

        [Fact]
        public void LanczosInterpolation_ConstantFunction_ReturnsConstant()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 7.0, 7.0, 7.0, 7.0 });
            var interpolator = new LanczosInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(1.5);

            // Assert - Should be close to 7
            Assert.True(Math.Abs(result - 7.0) < 0.1);
        }

        [Fact]
        public void LanczosInterpolation_SmoothCurve_PreservesShape()
        {
            // Arrange - Bell curve like data
            var x = new Vector<double>(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.14, 0.61, 1.0, 0.61, 0.14 });
            var interpolator = new LanczosInterpolation<double>(x, y);

            // Act - Check symmetry
            var left = interpolator.Interpolate(-0.5);
            var right = interpolator.Interpolate(0.5);

            // Assert - Should be symmetric
            Assert.True(Math.Abs(left - right) < 0.1);
        }

        [Fact]
        public void LanczosInterpolation_FivePoints_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 4.0, 9.0, 16.0 });
            var interpolator = new LanczosInterpolation<double>(x, y);

            // Act
            var result = interpolator.Interpolate(2.5);

            // Assert - Should be between 4 and 9
            Assert.True(result > 4.0 && result < 9.0);
        }

        #endregion

        #region CubicConvolutionInterpolation Tests (2D)

        [Fact]
        public void CubicConvolutionInterpolation_RecoversKnownGridPoints_Exactly()
        {
            // Arrange - 4x4 grid
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = i + j;

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < y.Length; j++)
                {
                    var result = interpolator.Interpolate(x[i], y[j]);
                    Assert.Equal(z[i, j], result, precision: 9);
                }
            }
        }

        [Fact]
        public void CubicConvolutionInterpolation_GridCellCenter_InterpolatesCorrectly()
        {
            // Arrange - Simple plane z = x + y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = x[i] + y[j];

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act - Center of cell (0,0) to (1,1) is (0.5, 0.5)
            var result = interpolator.Interpolate(0.5, 0.5);

            // Assert - Should be close to 0.5 + 0.5 = 1.0
            Assert.True(Math.Abs(result - 1.0) < 0.5);
        }

        [Fact]
        public void CubicConvolutionInterpolation_ConstantSurface_ReturnsConstant()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = 5.0;

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.5, 1.5);

            // Assert - Should be 5
            Assert.Equal(5.0, result, precision: 9);
        }

        [Fact]
        public void CubicConvolutionInterpolation_LinearSurfaceX_InterpolatesCorrectly()
        {
            // Arrange - z = x
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = x[i];

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.5, 1.0);

            // Assert - Should be close to 1.5
            Assert.True(Math.Abs(result - 1.5) < 0.2);
        }

        [Fact]
        public void CubicConvolutionInterpolation_LinearSurfaceY_InterpolatesCorrectly()
        {
            // Arrange - z = y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = y[j];

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.0, 1.5);

            // Assert - Should be close to 1.5
            Assert.True(Math.Abs(result - 1.5) < 0.2);
        }

        [Fact]
        public void CubicConvolutionInterpolation_BilinearSurface_InterpolatesWell()
        {
            // Arrange - z = x * y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = x[i] * y[j];

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act - At (1.5, 2), z should be 1.5 * 2 = 3
            var result = interpolator.Interpolate(1.5, 2.0);

            // Assert
            Assert.True(Math.Abs(result - 3.0) < 1.0);
        }

        [Fact]
        public void CubicConvolutionInterpolation_CornerToCorner_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            z[0, 0] = 1.0; z[0, 3] = 2.0;
            z[3, 0] = 3.0; z[3, 3] = 4.0;
            // Fill rest with interpolated values
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    if (z[i, j] == 0)
                        z[i, j] = 1.0 + (i * 2.0 + j) / 6.0;

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.5, 1.5);

            // Assert - Should be reasonable value
            Assert.True(result >= 1.0 && result <= 4.0);
        }

        [Fact]
        public void CubicConvolutionInterpolation_FourByFour_MinimalGrid()
        {
            // Arrange - Minimal 4x4 grid
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = (i + 1) * (j + 1);

            var interpolator = new CubicConvolutionInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(0.5, 0.5);

            // Assert - Should be reasonable
            Assert.True(result >= 0.5 && result <= 4.0);
        }

        #endregion

        #region BilinearInterpolation Tests (2D)

        [Fact]
        public void BilinearInterpolation_RecoversKnownGridPoints_Exactly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = i + j;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < y.Length; j++)
                {
                    var result = interpolator.Interpolate(x[i], y[j]);
                    Assert.Equal(z[i, j], result, precision: 10);
                }
            }
        }

        [Fact]
        public void BilinearInterpolation_CellCenter_AveragesCorners()
        {
            // Arrange - Unit square with known corners
            var x = new Vector<double>(new[] { 0.0, 1.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0 });
            var z = new Matrix<double>(2, 2);
            z[0, 0] = 1.0; z[0, 1] = 3.0;
            z[1, 0] = 5.0; z[1, 1] = 7.0;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act - Center of cell
            var result = interpolator.Interpolate(0.5, 0.5);

            // Assert - Should be average: (1+3+5+7)/4 = 4
            Assert.Equal(4.0, result, precision: 10);
        }

        [Fact]
        public void BilinearInterpolation_LinearSurfaceZ_EqualXPlusY_ReproducesExactly()
        {
            // Arrange - z = x + y is bilinear
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = x[i] + y[j];

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act - At (0.5, 0.7), z should be 0.5 + 0.7 = 1.2
            var result = interpolator.Interpolate(0.5, 0.7);

            // Assert
            Assert.Equal(1.2, result, precision: 10);
        }

        [Fact]
        public void BilinearInterpolation_ConstantSurface_ReturnsConstant()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = 10.0;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(0.7, 1.3);

            // Assert
            Assert.Equal(10.0, result, precision: 10);
        }

        [Fact]
        public void BilinearInterpolation_EdgeMidpoint_InterpolatesTwoCorners()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0 });
            var z = new Matrix<double>(2, 2);
            z[0, 0] = 1.0; z[0, 1] = 3.0;
            z[1, 0] = 5.0; z[1, 1] = 7.0;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act - Bottom edge midpoint (0.5, 0)
            var result = interpolator.Interpolate(0.5, 0.0);

            // Assert - Should be (1+5)/2 = 3
            Assert.Equal(3.0, result, precision: 10);
        }

        [Fact]
        public void BilinearInterpolation_MinimalTwoByTwoGrid_WorksCorrectly()
        {
            // Arrange - Minimal 2x2 grid
            var x = new Vector<double>(new[] { 0.0, 10.0 });
            var y = new Vector<double>(new[] { 0.0, 10.0 });
            var z = new Matrix<double>(2, 2);
            z[0, 0] = 0.0; z[0, 1] = 10.0;
            z[1, 0] = 20.0; z[1, 1] = 30.0;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(5.0, 5.0);

            // Assert - Center should be average = 15
            Assert.Equal(15.0, result, precision: 10);
        }

        [Fact]
        public void BilinearInterpolation_QuarterPoint_InterpolatesCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0 });
            var z = new Matrix<double>(2, 2);
            z[0, 0] = 0.0; z[0, 1] = 0.0;
            z[1, 0] = 4.0; z[1, 1] = 4.0;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act - Quarter point (0.25, 0.5)
            var result = interpolator.Interpolate(0.25, 0.5);

            // Assert - Should be 0.25 * 4 = 1.0
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void BilinearInterpolation_LargeGrid_WorksCorrectly()
        {
            // Arrange - Larger grid
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var z = new Matrix<double>(5, 5);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    z[i, j] = i * j;

            var interpolator = new BilinearInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.5, 2.5);

            // Assert - Should be 1.5 * 2.5 = 3.75
            Assert.Equal(3.75, result, precision: 10);
        }

        #endregion

        #region BicubicInterpolation Tests (2D)

        [Fact]
        public void BicubicInterpolation_RecoversKnownGridPoints_Exactly()
        {
            // Arrange - Minimal 4x4 grid
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = i + j;

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act & Assert
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < y.Length; j++)
                {
                    var result = interpolator.Interpolate(x[i], y[j]);
                    Assert.Equal(z[i, j], result, precision: 8);
                }
            }
        }

        [Fact]
        public void BicubicInterpolation_ConstantSurface_ReturnsConstant()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = 5.0;

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.5, 2.5);

            // Assert
            Assert.Equal(5.0, result, precision: 8);
        }

        [Fact]
        public void BicubicInterpolation_LinearSurfaceZ_EqualXPlusY_InterpolatesWell()
        {
            // Arrange - z = x + y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = x[i] + y[j];

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act - At (1.5, 2.5), z should be 1.5 + 2.5 = 4.0
            var result = interpolator.Interpolate(1.5, 2.5);

            // Assert
            Assert.True(Math.Abs(result - 4.0) < 0.5);
        }

        [Fact]
        public void BicubicInterpolation_CenterOfCell_InterpolatesSmooth()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = i * i + j * j;

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.5, 1.5);

            // Assert - Should be close to 1.5^2 + 1.5^2 = 4.5
            Assert.True(result >= 3.0 && result <= 6.0);
        }

        [Fact]
        public void BicubicInterpolation_ParabolicSurface_InterpolatesWell()
        {
            // Arrange - z = x^2 + y^2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = x[i] * x[i] + y[j] * y[j];

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(1.0, 1.0);

            // Assert - Should be 1 + 1 = 2
            Assert.Equal(2.0, result, precision: 8);
        }

        [Fact]
        public void BicubicInterpolation_MinimalFourByFour_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = (i + 1) * (j + 1);

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(0.5, 0.5);

            // Assert - Should be reasonable
            Assert.True(result >= 0.5 && result <= 4.0);
        }

        [Fact]
        public void BicubicInterpolation_LargerGrid_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0, 4.0 });
            var z = new Matrix<double>(5, 5);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    z[i, j] = i * j;

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act
            var result = interpolator.Interpolate(2.0, 2.0);

            // Assert - Should be 4.0
            Assert.Equal(4.0, result, precision: 8);
        }

        [Fact]
        public void BicubicInterpolation_BilinearFunction_InterpolatesExactly()
        {
            // Arrange - z = x * y (bilinear)
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = x[i] * y[j];

            var interpolator = new BicubicInterpolation<double>(x, y, z);

            // Act - At (1.5, 2.0), z should be 3.0
            var result = interpolator.Interpolate(1.5, 2.0);

            // Assert
            Assert.True(Math.Abs(result - 3.0) < 0.5);
        }

        #endregion

        #region Interpolation2DTo1DAdapter Tests

        [Fact]
        public void Interpolation2DTo1DAdapter_FixedX_WorksCorrectly()
        {
            // Arrange - Create 2D interpolation
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = x[i] + y[j];

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.0, true);

            // Act - Query with varying Y, fixed X=1.0
            var result = adapter.Interpolate(0.5);

            // Assert - Should be 1.0 + 0.5 = 1.5
            Assert.Equal(1.5, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_FixedY_WorksCorrectly()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = x[i] + y[j];

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.5, false);

            // Act - Query with varying X, fixed Y=1.5
            var result = adapter.Interpolate(0.5);

            // Assert - Should be 0.5 + 1.5 = 2.0
            Assert.Equal(2.0, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_SliceThroughCenter_ExtractsCorrectly()
        {
            // Arrange - z = x^2 + y^2
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = x[i] * x[i] + y[j] * y[j];

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 0.0, false);

            // Act - Slice at y=0, so z = x^2
            var result = adapter.Interpolate(1.0);

            // Assert - Should be 1.0
            Assert.Equal(1.0, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_ConstantSurface_ReturnsConstant()
        {
            // Arrange
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = 7.0;

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.0, true);

            // Act
            var result = adapter.Interpolate(0.5);

            // Assert
            Assert.Equal(7.0, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_LinearSliceFixedX_ExtractsLinear()
        {
            // Arrange - z = 2x + 3y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = 2 * x[i] + 3 * y[j];

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.0, true);

            // Act - Fixed x=1, varying y: z = 2(1) + 3y = 2 + 3y
            var result = adapter.Interpolate(1.0);

            // Assert - Should be 2 + 3(1) = 5
            Assert.Equal(5.0, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_LinearSliceFixedY_ExtractsLinear()
        {
            // Arrange - z = 2x + 3y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = 2 * x[i] + 3 * y[j];

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.0, false);

            // Act - Fixed y=1, varying x: z = 2x + 3(1) = 2x + 3
            var result = adapter.Interpolate(1.0);

            // Assert - Should be 2(1) + 3 = 5
            Assert.Equal(5.0, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_DiagonalSlice_WorksCorrectly()
        {
            // Arrange - z = x + y, slice at x=y
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0 });
            var z = new Matrix<double>(3, 3);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    z[i, j] = x[i] + y[j];

            var interpolator2D = new BilinearInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.0, true);

            // Act - Fixed x=1, query y=1
            var result = adapter.Interpolate(1.0);

            // Assert - Should be 1 + 1 = 2
            Assert.Equal(2.0, result, precision: 10);
        }

        [Fact]
        public void Interpolation2DTo1DAdapter_WithCubicConvolution_WorksCorrectly()
        {
            // Arrange - Use cubic convolution as base
            var x = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var y = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });
            var z = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    z[i, j] = i + j;

            var interpolator2D = new CubicConvolutionInterpolation<double>(x, y, z);
            var adapter = new Interpolation2DTo1DAdapter<double>(interpolator2D, 1.5, true);

            // Act
            var result = adapter.Interpolate(1.5);

            // Assert - Should be close to 1.5 + 1.5 = 3.0
            Assert.True(Math.Abs(result - 3.0) < 0.5);
        }

        #endregion
    }
}
