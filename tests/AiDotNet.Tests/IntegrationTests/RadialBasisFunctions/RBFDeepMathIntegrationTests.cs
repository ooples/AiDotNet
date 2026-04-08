using AiDotNet.RadialBasisFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RadialBasisFunctions;

/// <summary>
/// Deep mathematical integration tests for all Radial Basis Function implementations.
/// Verifies hand-calculated values, derivative correctness via numerical differentiation,
/// monotonicity, boundary values, and mathematical identities.
/// </summary>
public class RBFDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double DerivTolerance = 1e-4; // Numerical differentiation tolerance

    // =========================================================================
    // Helper: numerical derivative via central difference
    // =========================================================================
    private static double NumericalDerivative(Func<double, double> f, double x, double h = 1e-6)
    {
        return (f(x + h) - f(x - h)) / (2 * h);
    }

    // =========================================================================
    // GaussianRBF: f(r) = exp(-ε*r²)
    // =========================================================================

    [Fact]
    public void GaussianRBF_HandCalculated_Epsilon1_r1()
    {
        // f(1) = exp(-1*1²) = exp(-1) ≈ 0.367879441
        var rbf = new GaussianRBF<double>(1.0);
        Assert.Equal(Math.Exp(-1.0), rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void GaussianRBF_HandCalculated_Epsilon2_r1()
    {
        // f(1) = exp(-2*1²) = exp(-2) ≈ 0.135335283
        var rbf = new GaussianRBF<double>(2.0);
        Assert.Equal(Math.Exp(-2.0), rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void GaussianRBF_HandCalculated_Epsilon1_r2()
    {
        // f(2) = exp(-1*4) = exp(-4) ≈ 0.018315639
        var rbf = new GaussianRBF<double>(1.0);
        Assert.Equal(Math.Exp(-4.0), rbf.Compute(2.0), Tolerance);
    }

    [Fact]
    public void GaussianRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = -2εr * exp(-ε*r²)
        // At r=1.5, ε=1: f'(1.5) = -3 * exp(-2.25) ≈ -0.31672
        var rbf = new GaussianRBF<double>(1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void GaussianRBF_Derivative_HandCalculated_r1_epsilon1()
    {
        // f'(1) = -2*1*1 * exp(-1) = -2*exp(-1) ≈ -0.735758882
        var rbf = new GaussianRBF<double>(1.0);
        double expected = -2.0 * Math.Exp(-1.0);
        Assert.Equal(expected, rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void GaussianRBF_WidthDerivative_MatchesNumerical()
    {
        // df/dε = -r² * exp(-ε*r²)
        // At r=2, ε=1: df/dε = -4 * exp(-4) ≈ -0.07326
        double r = 2.0;
        double epsilon = 1.0;
        var rbf = new GaussianRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        // Numerical: (f(r; ε+h) - f(r; ε-h)) / (2h)
        double h = 1e-6;
        var rbfPlus = new GaussianRBF<double>(epsilon + h);
        var rbfMinus = new GaussianRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void GaussianRBF_WidthDerivative_HandCalculated()
    {
        // df/dε at r=1, ε=1: -1² * exp(-1) = -exp(-1) ≈ -0.367879441
        var rbf = new GaussianRBF<double>(1.0);
        Assert.Equal(-Math.Exp(-1.0), rbf.ComputeWidthDerivative(1.0), Tolerance);
    }

    [Fact]
    public void GaussianRBF_Symmetry_PositiveEqualsNegativeR()
    {
        var rbf = new GaussianRBF<double>(1.5);
        Assert.Equal(rbf.Compute(2.0), rbf.Compute(-2.0), Tolerance);
    }

    // =========================================================================
    // MultiquadricRBF: f(r) = √(r² + ε²)
    // =========================================================================

    [Fact]
    public void MultiquadricRBF_HandCalculated_AtZero()
    {
        // f(0) = √(0 + 1²) = 1
        var rbf = new MultiquadricRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void MultiquadricRBF_HandCalculated_r3_epsilon4()
    {
        // f(3) = √(9 + 16) = √25 = 5
        var rbf = new MultiquadricRBF<double>(4.0);
        Assert.Equal(5.0, rbf.Compute(3.0), Tolerance);
    }

    [Fact]
    public void MultiquadricRBF_HandCalculated_r1_epsilon1()
    {
        // f(1) = √(1 + 1) = √2 ≈ 1.41421356
        var rbf = new MultiquadricRBF<double>(1.0);
        Assert.Equal(Math.Sqrt(2.0), rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void MultiquadricRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = r / √(r² + ε²)
        var rbf = new MultiquadricRBF<double>(1.0);
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void MultiquadricRBF_Derivative_HandCalculated()
    {
        // f'(3) with ε=4: 3/√(9+16) = 3/5 = 0.6
        var rbf = new MultiquadricRBF<double>(4.0);
        Assert.Equal(0.6, rbf.ComputeDerivative(3.0), Tolerance);
    }

    [Fact]
    public void MultiquadricRBF_WidthDerivative_MatchesNumerical()
    {
        // df/dε = ε / √(r² + ε²)
        double r = 2.0;
        double epsilon = 1.5;
        var rbf = new MultiquadricRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new MultiquadricRBF<double>(epsilon + h);
        var rbfMinus = new MultiquadricRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void MultiquadricRBF_WidthDerivative_HandCalculated()
    {
        // df/dε at r=3, ε=4: 4/√(9+16) = 4/5 = 0.8
        var rbf = new MultiquadricRBF<double>(4.0);
        Assert.Equal(0.8, rbf.ComputeWidthDerivative(3.0), Tolerance);
    }

    [Fact]
    public void MultiquadricRBF_Monotonically_Increasing()
    {
        var rbf = new MultiquadricRBF<double>(1.0);
        double prev = rbf.Compute(0.0);
        for (double r = 0.5; r <= 5.0; r += 0.5)
        {
            double curr = rbf.Compute(r);
            Assert.True(curr > prev, $"MultiquadricRBF must increase: f({r})={curr} <= f({r - 0.5})={prev}");
            prev = curr;
        }
    }

    // =========================================================================
    // InverseMultiquadricRBF: f(r) = 1/√(r² + ε²)
    // =========================================================================

    [Fact]
    public void InverseMultiquadricRBF_HandCalculated_AtZero()
    {
        // f(0) = 1/√(0 + 1) = 1
        var rbf = new InverseMultiquadricRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void InverseMultiquadricRBF_HandCalculated_r3_epsilon4()
    {
        // f(3) = 1/√(9 + 16) = 1/5 = 0.2
        var rbf = new InverseMultiquadricRBF<double>(4.0);
        Assert.Equal(0.2, rbf.Compute(3.0), Tolerance);
    }

    [Fact]
    public void InverseMultiquadricRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = -r / (r² + ε²)^(3/2)
        var rbf = new InverseMultiquadricRBF<double>(1.0);
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void InverseMultiquadricRBF_Derivative_HandCalculated()
    {
        // f'(3) with ε=4: -3 / (9+16)^(3/2) = -3 / 125 = -0.024
        var rbf = new InverseMultiquadricRBF<double>(4.0);
        Assert.Equal(-3.0 / 125.0, rbf.ComputeDerivative(3.0), Tolerance);
    }

    [Fact]
    public void InverseMultiquadricRBF_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.5;
        var rbf = new InverseMultiquadricRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new InverseMultiquadricRBF<double>(epsilon + h);
        var rbfMinus = new InverseMultiquadricRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void InverseMultiquadricRBF_IsReciprocal_OfMultiquadric()
    {
        // InverseMultiquadric(r) = 1 / Multiquadric(r)
        var mq = new MultiquadricRBF<double>(2.0);
        var imq = new InverseMultiquadricRBF<double>(2.0);
        for (double r = 0.0; r <= 5.0; r += 0.5)
        {
            Assert.Equal(1.0 / mq.Compute(r), imq.Compute(r), Tolerance);
        }
    }

    // =========================================================================
    // LinearRBF: f(r) = r
    // =========================================================================

    [Fact]
    public void LinearRBF_ComputeIsIdentity()
    {
        var rbf = new LinearRBF<double>();
        Assert.Equal(0.0, rbf.Compute(0.0), Tolerance);
        Assert.Equal(3.7, rbf.Compute(3.7), Tolerance);
        Assert.Equal(100.0, rbf.Compute(100.0), Tolerance);
    }

    [Fact]
    public void LinearRBF_DerivativeAlwaysOne()
    {
        var rbf = new LinearRBF<double>();
        Assert.Equal(1.0, rbf.ComputeDerivative(0.0), Tolerance);
        Assert.Equal(1.0, rbf.ComputeDerivative(5.0), Tolerance);
        Assert.Equal(1.0, rbf.ComputeDerivative(999.0), Tolerance);
    }

    [Fact]
    public void LinearRBF_WidthDerivativeAlwaysZero()
    {
        var rbf = new LinearRBF<double>();
        Assert.Equal(0.0, rbf.ComputeWidthDerivative(0.0), Tolerance);
        Assert.Equal(0.0, rbf.ComputeWidthDerivative(5.0), Tolerance);
    }

    // =========================================================================
    // CubicRBF: f(r) = (r/width)³
    // =========================================================================

    [Fact]
    public void CubicRBF_HandCalculated_Width1()
    {
        // f(2) = (2/1)³ = 8
        var rbf = new CubicRBF<double>(1.0);
        Assert.Equal(8.0, rbf.Compute(2.0), Tolerance);
    }

    [Fact]
    public void CubicRBF_HandCalculated_Width2()
    {
        // f(4) = (4/2)³ = 8
        var rbf = new CubicRBF<double>(2.0);
        Assert.Equal(8.0, rbf.Compute(4.0), Tolerance);
    }

    [Fact]
    public void CubicRBF_AtZero_ReturnsZero()
    {
        var rbf = new CubicRBF<double>(1.0);
        Assert.Equal(0.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void CubicRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = 3r²/width³
        var rbf = new CubicRBF<double>(1.5);
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void CubicRBF_Derivative_HandCalculated()
    {
        // f'(3) with width=1: 3*9/1 = 27
        var rbf = new CubicRBF<double>(1.0);
        Assert.Equal(27.0, rbf.ComputeDerivative(3.0), Tolerance);
    }

    [Fact]
    public void CubicRBF_WidthDerivative_MatchesNumerical()
    {
        // df/dwidth = -3r³/width⁴
        double r = 2.0;
        double width = 1.5;
        var rbf = new CubicRBF<double>(width);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new CubicRBF<double>(width + h);
        var rbfMinus = new CubicRBF<double>(width - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void CubicRBF_WidthDerivative_HandCalculated()
    {
        // df/dwidth at r=2, width=1: -3*8/1 = -24
        var rbf = new CubicRBF<double>(1.0);
        Assert.Equal(-24.0, rbf.ComputeWidthDerivative(2.0), Tolerance);
    }

    // =========================================================================
    // ExponentialRBF: f(r) = exp(-ε*r)
    // =========================================================================

    [Fact]
    public void ExponentialRBF_HandCalculated_AtZero()
    {
        // f(0) = exp(0) = 1
        var rbf = new ExponentialRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void ExponentialRBF_HandCalculated_Epsilon1_r1()
    {
        // f(1) = exp(-1) ≈ 0.367879441
        var rbf = new ExponentialRBF<double>(1.0);
        Assert.Equal(Math.Exp(-1.0), rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void ExponentialRBF_HandCalculated_Epsilon2_r3()
    {
        // f(3) = exp(-2*3) = exp(-6) ≈ 0.002478752
        var rbf = new ExponentialRBF<double>(2.0);
        Assert.Equal(Math.Exp(-6.0), rbf.Compute(3.0), Tolerance);
    }

    [Fact]
    public void ExponentialRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = -ε * exp(-ε*r)
        var rbf = new ExponentialRBF<double>(1.5);
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void ExponentialRBF_Derivative_HandCalculated()
    {
        // f'(1) with ε=1: -1 * exp(-1) ≈ -0.367879441
        var rbf = new ExponentialRBF<double>(1.0);
        Assert.Equal(-Math.Exp(-1.0), rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void ExponentialRBF_WidthDerivative_MatchesNumerical()
    {
        // df/dε = -r * exp(-ε*r)
        double r = 2.0;
        double epsilon = 1.0;
        var rbf = new ExponentialRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new ExponentialRBF<double>(epsilon + h);
        var rbfMinus = new ExponentialRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void ExponentialRBF_WidthDerivative_HandCalculated()
    {
        // df/dε at r=3, ε=1: -3 * exp(-3) ≈ -0.149361
        var rbf = new ExponentialRBF<double>(1.0);
        Assert.Equal(-3.0 * Math.Exp(-3.0), rbf.ComputeWidthDerivative(3.0), Tolerance);
    }

    [Fact]
    public void ExponentialRBF_MonotonicallyDecreasing()
    {
        var rbf = new ExponentialRBF<double>(1.0);
        double prev = rbf.Compute(0.0);
        for (double r = 0.5; r <= 5.0; r += 0.5)
        {
            double curr = rbf.Compute(r);
            Assert.True(curr < prev, $"ExponentialRBF must decrease: f({r})={curr} >= f({r - 0.5})={prev}");
            prev = curr;
        }
    }

    // =========================================================================
    // ThinPlateSplineRBF: f(r) = r² * log(r)
    // =========================================================================

    [Fact]
    public void ThinPlateSplineRBF_AtZero_ReturnsZero()
    {
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.Equal(0.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_HandCalculated_r1()
    {
        // f(1) = 1² * log(1) = 0
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.Equal(0.0, rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_HandCalculated_rE()
    {
        // f(e) = e² * log(e) = e² * 1 = e² ≈ 7.389056
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.Equal(Math.E * Math.E, rbf.Compute(Math.E), Tolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_HandCalculated_r2()
    {
        // f(2) = 4 * log(2) ≈ 2.772589
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.Equal(4.0 * Math.Log(2.0), rbf.Compute(2.0), Tolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_NegativeForSmallR()
    {
        // For 0 < r < 1, log(r) < 0 so r²*log(r) < 0
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.True(rbf.Compute(0.5) < 0.0, "TPS should be negative for 0 < r < 1");
    }

    [Fact]
    public void ThinPlateSplineRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = r * (2*log(r) + 1)
        var rbf = new ThinPlateSplineRBF<double>();
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_Derivative_HandCalculated_r1()
    {
        // f'(1) = 1 * (2*log(1) + 1) = 1 * (0 + 1) = 1
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.Equal(1.0, rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void ThinPlateSplineRBF_WidthDerivativeAlwaysZero()
    {
        var rbf = new ThinPlateSplineRBF<double>();
        Assert.Equal(0.0, rbf.ComputeWidthDerivative(0.0), Tolerance);
        Assert.Equal(0.0, rbf.ComputeWidthDerivative(2.5), Tolerance);
    }

    // =========================================================================
    // InverseQuadraticRBF: f(r) = 1/(1 + (εr)²)
    // =========================================================================

    [Fact]
    public void InverseQuadraticRBF_HandCalculated_AtZero()
    {
        // f(0) = 1/(1 + 0) = 1
        var rbf = new InverseQuadraticRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_HandCalculated_r1_epsilon1()
    {
        // f(1) = 1/(1 + 1) = 0.5
        var rbf = new InverseQuadraticRBF<double>(1.0);
        Assert.Equal(0.5, rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_HandCalculated_r2_epsilon1()
    {
        // f(2) = 1/(1 + 4) = 0.2
        var rbf = new InverseQuadraticRBF<double>(1.0);
        Assert.Equal(0.2, rbf.Compute(2.0), Tolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_HandCalculated_r1_epsilon2()
    {
        // f(1) = 1/(1 + (2*1)²) = 1/(1 + 4) = 0.2
        var rbf = new InverseQuadraticRBF<double>(2.0);
        Assert.Equal(0.2, rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = -2ε²r / (1 + (εr)²)²
        var rbf = new InverseQuadraticRBF<double>(1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_Derivative_HandCalculated()
    {
        // f'(1) with ε=1: -2*1*1 / (1+1)² = -2/4 = -0.5
        var rbf = new InverseQuadraticRBF<double>(1.0);
        Assert.Equal(-0.5, rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.0;
        var rbf = new InverseQuadraticRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new InverseQuadraticRBF<double>(epsilon + h);
        var rbfMinus = new InverseQuadraticRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void InverseQuadraticRBF_WidthDerivative_HandCalculated()
    {
        // df/dε at r=1, ε=1: -2*1*1 / (1+1)² = -2/4 = -0.5
        var rbf = new InverseQuadraticRBF<double>(1.0);
        Assert.Equal(-0.5, rbf.ComputeWidthDerivative(1.0), Tolerance);
    }

    // =========================================================================
    // SquaredExponentialRBF: f(r) = exp(-(εr)²)
    // =========================================================================

    [Fact]
    public void SquaredExponentialRBF_HandCalculated_AtZero()
    {
        var rbf = new SquaredExponentialRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_HandCalculated_r1_epsilon1()
    {
        // f(1) = exp(-(1*1)²) = exp(-1) ≈ 0.367879441
        var rbf = new SquaredExponentialRBF<double>(1.0);
        Assert.Equal(Math.Exp(-1.0), rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_HandCalculated_r1_epsilon2()
    {
        // f(1) = exp(-(2*1)²) = exp(-4) ≈ 0.018315639
        var rbf = new SquaredExponentialRBF<double>(2.0);
        Assert.Equal(Math.Exp(-4.0), rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_Derivative_MatchesNumerical()
    {
        // f'(r) = -2ε²r * exp(-(εr)²)
        var rbf = new SquaredExponentialRBF<double>(1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_Derivative_HandCalculated()
    {
        // f'(1) with ε=1: -2*1*1 * exp(-1) = -2*exp(-1) ≈ -0.735759
        var rbf = new SquaredExponentialRBF<double>(1.0);
        Assert.Equal(-2.0 * Math.Exp(-1.0), rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.0;
        var rbf = new SquaredExponentialRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new SquaredExponentialRBF<double>(epsilon + h);
        var rbfMinus = new SquaredExponentialRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void SquaredExponentialRBF_Relationship_ToGaussianRBF()
    {
        // SquaredExponential f(r)=exp(-(εr)²) and GaussianRBF f(r)=exp(-ε*r²)
        // For ε_SE = 1: exp(-(1*r)²) = exp(-r²)
        // For ε_Gauss = 1: exp(-1*r²) = exp(-r²)
        // So with both ε=1, they should be identical
        var se = new SquaredExponentialRBF<double>(1.0);
        var gauss = new GaussianRBF<double>(1.0);
        for (double r = 0.0; r <= 3.0; r += 0.5)
        {
            Assert.Equal(gauss.Compute(r), se.Compute(r), Tolerance);
        }
    }

    [Fact]
    public void SquaredExponentialRBF_Relationship_ToGaussianRBF_DifferentEpsilon()
    {
        // SquaredExponential f(r) = exp(-(εr)²) = exp(-ε²*r²)
        // Gaussian f(r) = exp(-ε*r²)
        // So SquaredExponential(ε) = Gaussian(ε²)
        double eps = 2.0;
        var se = new SquaredExponentialRBF<double>(eps);
        var gauss = new GaussianRBF<double>(eps * eps);  // ε² = 4
        for (double r = 0.0; r <= 3.0; r += 0.5)
        {
            Assert.Equal(gauss.Compute(r), se.Compute(r), Tolerance);
        }
    }

    // =========================================================================
    // RationalQuadraticRBF: f(r) = 1 - r²/(r² + ε²)
    // =========================================================================

    [Fact]
    public void RationalQuadraticRBF_HandCalculated_AtZero()
    {
        var rbf = new RationalQuadraticRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_HandCalculated_r1_epsilon1()
    {
        // f(1) = 1 - 1/(1+1) = 1 - 0.5 = 0.5
        var rbf = new RationalQuadraticRBF<double>(1.0);
        Assert.Equal(0.5, rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_HandCalculated_r3_epsilon4()
    {
        // f(3) = 1 - 9/(9+16) = 1 - 9/25 = 1 - 0.36 = 0.64
        var rbf = new RationalQuadraticRBF<double>(4.0);
        Assert.Equal(0.64, rbf.Compute(3.0), Tolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_SimplifiesTo_EpsilonSquaredOverSum()
    {
        // f(r) = 1 - r²/(r²+ε²) = ε²/(r²+ε²)
        var rbf = new RationalQuadraticRBF<double>(3.0);
        double r = 4.0;
        double expected = 9.0 / (16.0 + 9.0); // ε²/(r²+ε²) = 9/25 = 0.36
        Assert.Equal(expected, rbf.Compute(r), Tolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_Derivative_MatchesNumerical()
    {
        var rbf = new RationalQuadraticRBF<double>(1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_Derivative_HandCalculated()
    {
        // f'(r) = -2rε²/(r²+ε²)²
        // At r=1, ε=1: -2*1*1/(1+1)² = -2/4 = -0.5
        var rbf = new RationalQuadraticRBF<double>(1.0);
        Assert.Equal(-0.5, rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.5;
        var rbf = new RationalQuadraticRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new RationalQuadraticRBF<double>(epsilon + h);
        var rbfMinus = new RationalQuadraticRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void RationalQuadraticRBF_WidthDerivative_IsPositive()
    {
        // Increasing ε should increase f(r) for r > 0 (wider spread)
        var rbf = new RationalQuadraticRBF<double>(1.0);
        Assert.True(rbf.ComputeWidthDerivative(1.0) > 0);
        Assert.True(rbf.ComputeWidthDerivative(3.0) > 0);
    }

    // =========================================================================
    // SphericalRBF: f(r) = 1 - 1.5(r/ε) + 0.5(r/ε)³ for r ≤ ε, else 0
    // =========================================================================

    [Fact]
    public void SphericalRBF_HandCalculated_AtZero()
    {
        var rbf = new SphericalRBF<double>(1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void SphericalRBF_HandCalculated_AtBoundary()
    {
        // f(ε) = 1 - 1.5*1 + 0.5*1 = 1 - 1.5 + 0.5 = 0
        var rbf = new SphericalRBF<double>(1.0);
        Assert.Equal(0.0, rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void SphericalRBF_BeyondBoundary_ReturnsZero()
    {
        var rbf = new SphericalRBF<double>(1.0);
        Assert.Equal(0.0, rbf.Compute(1.5), Tolerance);
        Assert.Equal(0.0, rbf.Compute(10.0), Tolerance);
    }

    [Fact]
    public void SphericalRBF_HandCalculated_HalfRadius()
    {
        // f(0.5) with ε=1: 1 - 1.5*0.5 + 0.5*0.125 = 1 - 0.75 + 0.0625 = 0.3125
        var rbf = new SphericalRBF<double>(1.0);
        Assert.Equal(0.3125, rbf.Compute(0.5), Tolerance);
    }

    [Fact]
    public void SphericalRBF_Derivative_MatchesNumerical()
    {
        var rbf = new SphericalRBF<double>(2.0);
        double analytical = rbf.ComputeDerivative(0.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 0.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void SphericalRBF_Derivative_HandCalculated_AtZero()
    {
        // f'(0) = (1.5/ε)[(0)² - 1] = 1.5 * (-1) = -1.5
        var rbf = new SphericalRBF<double>(1.0);
        Assert.Equal(-1.5, rbf.ComputeDerivative(0.0), Tolerance);
    }

    [Fact]
    public void SphericalRBF_Derivative_AtBoundary_IsZero()
    {
        // f'(ε) = (1.5/ε)[(ε/ε)² - 1] = (1.5/ε)[1 - 1] = 0
        var rbf = new SphericalRBF<double>(1.0);
        Assert.Equal(0.0, rbf.ComputeDerivative(1.0), Tolerance);
    }

    [Fact]
    public void SphericalRBF_WidthDerivative_MatchesNumerical()
    {
        double r = 0.5;
        double epsilon = 1.0;
        var rbf = new SphericalRBF<double>(epsilon);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new SphericalRBF<double>(epsilon + h);
        var rbfMinus = new SphericalRBF<double>(epsilon - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void SphericalRBF_CompactSupport()
    {
        // Values inside support must be positive, outside must be zero
        var rbf = new SphericalRBF<double>(2.0);
        Assert.True(rbf.Compute(0.5) > 0.0);
        Assert.True(rbf.Compute(1.0) > 0.0);
        Assert.True(rbf.Compute(1.5) > 0.0);
        Assert.Equal(0.0, rbf.Compute(2.5), Tolerance);
        Assert.Equal(0.0, rbf.Compute(5.0), Tolerance);
    }

    // =========================================================================
    // WendlandRBF: compact support with different smoothness orders
    // =========================================================================

    [Fact]
    public void WendlandRBF_K0_HandCalculated_AtZero()
    {
        // f(0) = (1-0)² = 1
        var rbf = new WendlandRBF<double>(k: 0, supportRadius: 1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void WendlandRBF_K0_HandCalculated_AtHalf()
    {
        // f(0.5) = (1-0.5)² = 0.25
        var rbf = new WendlandRBF<double>(k: 0, supportRadius: 1.0);
        Assert.Equal(0.25, rbf.Compute(0.5), Tolerance);
    }

    [Fact]
    public void WendlandRBF_K0_AtBoundary_ReturnsZero()
    {
        // f(1) = (1-1)² = 0
        var rbf = new WendlandRBF<double>(k: 0, supportRadius: 1.0);
        Assert.Equal(0.0, rbf.Compute(1.0), Tolerance);
    }

    [Fact]
    public void WendlandRBF_K1_HandCalculated_AtZero()
    {
        // f(0) = (1-0)⁴*(1+4*0) = 1*1 = 1  ... wait: k=1 formula is (1-r)^4*(1+4r)
        // f(0) = 1^4 * (1+0) = 1
        var rbf = new WendlandRBF<double>(k: 1, supportRadius: 1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void WendlandRBF_K1_HandCalculated_AtHalf()
    {
        // f(0.5) = (0.5)⁴ * (1+2) = 0.0625 * 3 = 0.1875
        var rbf = new WendlandRBF<double>(k: 1, supportRadius: 1.0);
        Assert.Equal(0.1875, rbf.Compute(0.5), Tolerance);
    }

    [Fact]
    public void WendlandRBF_K2_HandCalculated_AtZero()
    {
        // f(0) = (1-0)⁶*(3+18*0+35*0²) = 1*3 = 3
        var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);
        Assert.Equal(3.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void WendlandRBF_K2_HandCalculated_AtHalf()
    {
        // f(0.5) = (0.5)⁶ * (3 + 9 + 8.75) = (1/64) * 20.75 = 0.32421875
        var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);
        double expected = Math.Pow(0.5, 6) * (3.0 + 18.0 * 0.5 + 35.0 * 0.25);
        Assert.Equal(expected, rbf.Compute(0.5), Tolerance);
    }

    [Fact]
    public void WendlandRBF_CompactSupport_AllK()
    {
        // Beyond support radius, all k values should give zero
        for (int k = 0; k <= 2; k++)
        {
            var rbf = new WendlandRBF<double>(k: k, supportRadius: 2.0);
            Assert.Equal(0.0, rbf.Compute(2.0), Tolerance);
            Assert.Equal(0.0, rbf.Compute(3.0), Tolerance);
        }
    }

    [Fact]
    public void WendlandRBF_K0_Derivative_MatchesNumerical()
    {
        var rbf = new WendlandRBF<double>(k: 0, supportRadius: 1.0);
        double analytical = rbf.ComputeDerivative(0.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 0.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void WendlandRBF_K1_Derivative_MatchesNumerical()
    {
        var rbf = new WendlandRBF<double>(k: 1, supportRadius: 1.0);
        double analytical = rbf.ComputeDerivative(0.3);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 0.3);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void WendlandRBF_K2_Derivative_MatchesNumerical()
    {
        var rbf = new WendlandRBF<double>(k: 2, supportRadius: 1.0);
        double analytical = rbf.ComputeDerivative(0.3);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 0.3);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void WendlandRBF_K0_Derivative_HandCalculated()
    {
        // f'(r) = -2(1-r) at r=0.5: -2*0.5 = -1.0
        var rbf = new WendlandRBF<double>(k: 0, supportRadius: 1.0);
        Assert.Equal(-1.0, rbf.ComputeDerivative(0.5), Tolerance);
    }

    // =========================================================================
    // PolyharmonicSplineRBF: odd k: r^k, even k: r^k*log(r)
    // =========================================================================

    [Fact]
    public void PolyharmonicSplineRBF_K1_IsLinear()
    {
        // k=1: f(r) = r
        var rbf = new PolyharmonicSplineRBF<double>(k: 1);
        Assert.Equal(0.0, rbf.Compute(0.0), Tolerance);
        Assert.Equal(3.5, rbf.Compute(3.5), Tolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_K2_IsThinPlateSpline()
    {
        // k=2: f(r) = r²*log(r)
        var rbf = new PolyharmonicSplineRBF<double>(k: 2);
        var tps = new ThinPlateSplineRBF<double>();
        for (double r = 0.5; r <= 5.0; r += 0.5)
        {
            Assert.Equal(tps.Compute(r), rbf.Compute(r), Tolerance);
        }
    }

    [Fact]
    public void PolyharmonicSplineRBF_K3_IsCubic()
    {
        // k=3: f(r) = r³
        var rbf = new PolyharmonicSplineRBF<double>(k: 3);
        Assert.Equal(0.0, rbf.Compute(0.0), Tolerance);
        Assert.Equal(8.0, rbf.Compute(2.0), Tolerance);  // 2³ = 8
        Assert.Equal(27.0, rbf.Compute(3.0), Tolerance); // 3³ = 27
    }

    [Fact]
    public void PolyharmonicSplineRBF_K4_HandCalculated()
    {
        // k=4 (even): f(r) = r⁴*log(r)
        var rbf = new PolyharmonicSplineRBF<double>(k: 4);
        // f(2) = 16 * log(2) ≈ 11.0903549
        Assert.Equal(16.0 * Math.Log(2.0), rbf.Compute(2.0), Tolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_Derivative_K1_MatchesNumerical()
    {
        var rbf = new PolyharmonicSplineRBF<double>(k: 1);
        // k=1: derivative is 1 everywhere
        Assert.Equal(1.0, rbf.ComputeDerivative(5.0), Tolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_Derivative_K3_MatchesNumerical()
    {
        var rbf = new PolyharmonicSplineRBF<double>(k: 3);
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_Derivative_K2_MatchesNumerical()
    {
        var rbf = new PolyharmonicSplineRBF<double>(k: 2);
        double analytical = rbf.ComputeDerivative(2.0);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 2.0);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_Derivative_K3_HandCalculated()
    {
        // k=3: f'(r) = 3r², f'(2) = 12
        var rbf = new PolyharmonicSplineRBF<double>(k: 3);
        Assert.Equal(12.0, rbf.ComputeDerivative(2.0), Tolerance);
    }

    [Fact]
    public void PolyharmonicSplineRBF_WidthDerivativeAlwaysZero()
    {
        for (int k = 1; k <= 4; k++)
        {
            var rbf = new PolyharmonicSplineRBF<double>(k: k);
            Assert.Equal(0.0, rbf.ComputeWidthDerivative(2.0), Tolerance);
        }
    }

    // =========================================================================
    // MaternRBF: special cases at ν=0.5, 1.5, 2.5
    // =========================================================================

    [Fact]
    public void MaternRBF_Nu05_AtZero_ReturnsOne()
    {
        var rbf = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tolerance);
    }

    [Fact]
    public void MaternRBF_Nu05_IsExponential()
    {
        // ν=0.5: k(r) = exp(-√(2*0.5)*r/l) = exp(-r/l)
        var rbf = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
        double r = 2.0;
        double expected = Math.Exp(-r); // √(2*0.5) = 1
        Assert.Equal(expected, rbf.Compute(r), Tolerance);
    }

    [Fact]
    public void MaternRBF_Nu15_HandCalculated()
    {
        // ν=1.5: k(r) = (1 + √3*r/l) * exp(-√3*r/l)
        var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);
        double r = 1.0;
        double x = Math.Sqrt(3.0) * r;
        double expected = (1.0 + x) * Math.Exp(-x);
        Assert.Equal(expected, rbf.Compute(r), Tolerance);
    }

    [Fact]
    public void MaternRBF_Nu25_HandCalculated()
    {
        // ν=2.5: k(r) = (1 + √5*r/l + 5r²/(3l²)) * exp(-√5*r/l)
        var rbf = new MaternRBF<double>(nu: 2.5, lengthScale: 1.0);
        double r = 1.0;
        double x = Math.Sqrt(5.0) * r;
        double expected = (1.0 + x + x * x / 3.0) * Math.Exp(-x);
        Assert.Equal(expected, rbf.Compute(r), Tolerance);
    }

    [Fact]
    public void MaternRBF_Nu05_Derivative_MatchesNumerical()
    {
        var rbf = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void MaternRBF_Nu15_Derivative_MatchesNumerical()
    {
        var rbf = new MaternRBF<double>(nu: 1.5, lengthScale: 1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void MaternRBF_Nu25_Derivative_MatchesNumerical()
    {
        var rbf = new MaternRBF<double>(nu: 2.5, lengthScale: 1.0);
        double analytical = rbf.ComputeDerivative(1.5);
        double numerical = NumericalDerivative(r => rbf.Compute(r), 1.5);
        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void MaternRBF_Nu05_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double ls = 1.0;
        var rbf = new MaternRBF<double>(nu: 0.5, lengthScale: ls);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new MaternRBF<double>(nu: 0.5, lengthScale: ls + h);
        var rbfMinus = new MaternRBF<double>(nu: 0.5, lengthScale: ls - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void MaternRBF_AllNu_DerivativeAtZero_IsZero()
    {
        // Symmetry: derivative at r=0 should be zero for all ν
        foreach (double nu in new[] { 0.5, 1.5, 2.5 })
        {
            var rbf = new MaternRBF<double>(nu: nu, lengthScale: 1.0);
            Assert.Equal(0.0, rbf.ComputeDerivative(0.0), Tolerance);
        }
    }

    [Fact]
    public void MaternRBF_MonotonicallyDecreasing()
    {
        foreach (double nu in new[] { 0.5, 1.5, 2.5 })
        {
            var rbf = new MaternRBF<double>(nu: nu, lengthScale: 1.0);
            double prev = rbf.Compute(0.0);
            for (double r = 0.5; r <= 5.0; r += 0.5)
            {
                double curr = rbf.Compute(r);
                Assert.True(curr < prev, $"MaternRBF(ν={nu}) must decrease: f({r})={curr} >= f({r - 0.5})={prev}");
                prev = curr;
            }
        }
    }

    // =========================================================================
    // Cross-RBF identity tests
    // =========================================================================

    [Fact]
    public void PolyharmonicK1_Matches_LinearRBF()
    {
        var poly = new PolyharmonicSplineRBF<double>(k: 1);
        var linear = new LinearRBF<double>();
        for (double r = 0.0; r <= 5.0; r += 0.5)
        {
            Assert.Equal(linear.Compute(r), poly.Compute(r), Tolerance);
        }
    }

    [Fact]
    public void CubicRBF_Width1_Matches_PolyharmonicK3()
    {
        var cubic = new CubicRBF<double>(1.0);
        var poly = new PolyharmonicSplineRBF<double>(k: 3);
        for (double r = 0.0; r <= 5.0; r += 0.5)
        {
            Assert.Equal(poly.Compute(r), cubic.Compute(r), Tolerance);
        }
    }

    [Fact]
    public void MaternNu05_Matches_ExponentialRBF()
    {
        // Matérn(ν=0.5, l=1) = exp(-r) and Exponential(ε=1) = exp(-r)
        var matern = new MaternRBF<double>(nu: 0.5, lengthScale: 1.0);
        var expo = new ExponentialRBF<double>(1.0);
        for (double r = 0.0; r <= 5.0; r += 0.5)
        {
            Assert.Equal(expo.Compute(r), matern.Compute(r), Tolerance);
        }
    }

    // =========================================================================
    // Derivative correctness for Wendland width derivatives
    // =========================================================================

    [Fact]
    public void WendlandRBF_K0_WidthDerivative_MatchesNumerical()
    {
        double r = 0.5;
        double sr = 1.0;
        var rbf = new WendlandRBF<double>(k: 0, supportRadius: sr);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new WendlandRBF<double>(k: 0, supportRadius: sr + h);
        var rbfMinus = new WendlandRBF<double>(k: 0, supportRadius: sr - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void WendlandRBF_K1_WidthDerivative_MatchesNumerical()
    {
        double r = 0.3;
        double sr = 1.0;
        var rbf = new WendlandRBF<double>(k: 1, supportRadius: sr);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new WendlandRBF<double>(k: 1, supportRadius: sr + h);
        var rbfMinus = new WendlandRBF<double>(k: 1, supportRadius: sr - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    [Fact]
    public void WendlandRBF_K2_WidthDerivative_MatchesNumerical()
    {
        double r = 0.3;
        double sr = 1.0;
        var rbf = new WendlandRBF<double>(k: 2, supportRadius: sr);
        double analytical = rbf.ComputeWidthDerivative(r);

        double h = 1e-6;
        var rbfPlus = new WendlandRBF<double>(k: 2, supportRadius: sr + h);
        var rbfMinus = new WendlandRBF<double>(k: 2, supportRadius: sr - h);
        double numerical = (rbfPlus.Compute(r) - rbfMinus.Compute(r)) / (2 * h);

        Assert.Equal(numerical, analytical, DerivTolerance);
    }

    // =========================================================================
    // Boundary and special value tests
    // =========================================================================

    [Fact]
    public void AllDecayingRBFs_AtZero_ReturnOne()
    {
        // All bell-shaped RBFs should return 1 at r=0
        Assert.Equal(1.0, new GaussianRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(1.0, new ExponentialRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(1.0, new InverseQuadraticRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(1.0, new SquaredExponentialRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(1.0, new RationalQuadraticRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(1.0, new SphericalRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(1.0, new MaternRBF<double>(1.5, 1.0).Compute(0.0), Tolerance);
    }

    [Fact]
    public void AllDecayingRBFs_DerivativeAtZero_IsZero()
    {
        // Symmetric RBFs should have zero derivative at r=0
        Assert.Equal(0.0, new GaussianRBF<double>(1.0).ComputeDerivative(0.0), Tolerance);
        Assert.Equal(0.0, new InverseQuadraticRBF<double>(1.0).ComputeDerivative(0.0), Tolerance);
        Assert.Equal(0.0, new SquaredExponentialRBF<double>(1.0).ComputeDerivative(0.0), Tolerance);
        Assert.Equal(0.0, new RationalQuadraticRBF<double>(1.0).ComputeDerivative(0.0), Tolerance);
        Assert.Equal(0.0, new MaternRBF<double>(1.5, 1.0).ComputeDerivative(0.0), Tolerance);
    }

    [Fact]
    public void AllGrowingRBFs_AtZero_ReturnZero()
    {
        // Growing RBFs should return 0 at r=0
        Assert.Equal(0.0, new LinearRBF<double>().Compute(0.0), Tolerance);
        Assert.Equal(0.0, new CubicRBF<double>(1.0).Compute(0.0), Tolerance);
        Assert.Equal(0.0, new ThinPlateSplineRBF<double>().Compute(0.0), Tolerance);
    }

    [Fact]
    public void AllWidthFreeRBFs_WidthDerivative_IsZero()
    {
        // RBFs without width parameter should always have zero width derivative
        Assert.Equal(0.0, new LinearRBF<double>().ComputeWidthDerivative(2.0), Tolerance);
        Assert.Equal(0.0, new ThinPlateSplineRBF<double>().ComputeWidthDerivative(2.0), Tolerance);
        Assert.Equal(0.0, new PolyharmonicSplineRBF<double>(k: 3).ComputeWidthDerivative(2.0), Tolerance);
    }

    // =========================================================================
    // Numerical stability / large values
    // =========================================================================

    [Fact]
    public void GaussianRBF_LargeR_ApproachesZero()
    {
        var rbf = new GaussianRBF<double>(1.0);
        double val = rbf.Compute(10.0);
        Assert.True(val >= 0.0, "GaussianRBF should never be negative");
        Assert.True(val < 1e-10, $"GaussianRBF at r=10 should be very small, got {val}");
    }

    [Fact]
    public void ExponentialRBF_LargeR_ApproachesZero()
    {
        var rbf = new ExponentialRBF<double>(1.0);
        double val = rbf.Compute(20.0);
        Assert.True(val >= 0.0, "ExponentialRBF should never be negative");
        Assert.True(val < 1e-8, $"ExponentialRBF at r=20 should be very small, got {val}");
    }

    [Fact]
    public void InverseQuadraticRBF_LargeR_ApproachesZero()
    {
        var rbf = new InverseQuadraticRBF<double>(1.0);
        double val = rbf.Compute(100.0);
        Assert.True(val > 0.0, "InverseQuadraticRBF should be positive");
        Assert.True(val < 0.001, $"InverseQuadraticRBF at r=100 should be very small, got {val}");
    }
}
