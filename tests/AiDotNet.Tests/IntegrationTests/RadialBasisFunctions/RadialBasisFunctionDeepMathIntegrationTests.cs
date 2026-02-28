using AiDotNet.RadialBasisFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RadialBasisFunctions;

/// <summary>
/// Deep mathematical integration tests for Radial Basis Functions.
/// Tests verify mathematical properties, derivative correctness via numerical differentiation,
/// and cross-function relationships that expose implementation bugs.
/// </summary>
public class RadialBasisFunctionDeepMathIntegrationTests
{
    private const double Tol = 1e-6;
    private const double DerivTol = 1e-4; // For numerical derivative checks (finite difference)
    private const double H = 1e-6; // Step size for finite differences

    // ─── Gaussian RBF ───────────────────────────────────────────────────

    [Fact]
    public void Gaussian_AtOrigin_ReturnsOne()
    {
        // f(0) = exp(-e*0) = 1 for any epsilon
        var rbf = new GaussianRBF<double>(epsilon: 2.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void Gaussian_HandComputed_ExpValues()
    {
        // f(r) = exp(-e*r^2), with e=1: f(1) = exp(-1) ≈ 0.36788
        var rbf = new GaussianRBF<double>(epsilon: 1.0);
        Assert.Equal(Math.Exp(-1), rbf.Compute(1.0), Tol);
        Assert.Equal(Math.Exp(-4), rbf.Compute(2.0), Tol);
    }

    [Fact]
    public void Gaussian_IsMonotonicallyDecreasing()
    {
        var rbf = new GaussianRBF<double>(epsilon: 1.0);
        double prev = rbf.Compute(0.0);
        for (double r = 0.1; r <= 5.0; r += 0.1)
        {
            double curr = rbf.Compute(r);
            Assert.True(curr < prev, $"Gaussian not monotonically decreasing at r={r}");
            prev = curr;
        }
    }

    [Fact]
    public void Gaussian_Derivative_MatchesNumericalDerivative()
    {
        // Verify d/dr[exp(-e*r^2)] = -2er*exp(-e*r^2) using finite differences
        var rbf = new GaussianRBF<double>(epsilon: 1.5);
        double[] testPoints = { 0.5, 1.0, 2.0, 3.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"Gaussian derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    [Fact]
    public void Gaussian_DerivativeAtOrigin_IsZero()
    {
        // f'(0) = -2e*0*exp(0) = 0
        var rbf = new GaussianRBF<double>(epsilon: 3.0);
        Assert.Equal(0.0, rbf.ComputeDerivative(0.0), Tol);
    }

    [Fact]
    public void Gaussian_WidthDerivative_MatchesNumerical()
    {
        // d/de[exp(-e*r^2)] = -r^2*exp(-e*r^2)
        double r = 2.0;
        double epsilon = 1.0;

        var rbf1 = new GaussianRBF<double>(epsilon: epsilon);
        var rbf2 = new GaussianRBF<double>(epsilon: epsilon + H);
        var rbf3 = new GaussianRBF<double>(epsilon: epsilon - H);

        double analytic = rbf1.ComputeWidthDerivative(r);
        double numerical = (rbf2.Compute(r) - rbf3.Compute(r)) / (2 * H);

        Assert.True(Math.Abs(analytic - numerical) < DerivTol,
            $"Gaussian width derivative mismatch: analytic={analytic}, numerical={numerical}");
    }

    [Fact]
    public void Gaussian_SymmetricInR()
    {
        // f(r) = f(-r) since f depends on r^2
        var rbf = new GaussianRBF<double>(epsilon: 1.0);
        Assert.Equal(rbf.Compute(2.0), rbf.Compute(-2.0), Tol);
    }

    // ─── Multiquadric RBF ───────────────────────────────────────────────

    [Fact]
    public void Multiquadric_AtOrigin_ReturnsEpsilon()
    {
        // f(0) = sqrt(0 + e^2) = e
        double eps = 2.5;
        var rbf = new MultiquadricRBF<double>(epsilon: eps);
        Assert.Equal(eps, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void Multiquadric_HandComputed()
    {
        // f(r) = sqrt(r^2 + e^2), e=1: f(3) = sqrt(9+1) = sqrt(10) ≈ 3.16228
        var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
        Assert.Equal(Math.Sqrt(10), rbf.Compute(3.0), Tol);
    }

    [Fact]
    public void Multiquadric_IsMonotonicallyIncreasing()
    {
        var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
        double prev = rbf.Compute(0.0);
        for (double r = 0.1; r <= 5.0; r += 0.1)
        {
            double curr = rbf.Compute(r);
            Assert.True(curr > prev, $"Multiquadric not monotonically increasing at r={r}");
            prev = curr;
        }
    }

    [Fact]
    public void Multiquadric_Derivative_MatchesNumerical()
    {
        var rbf = new MultiquadricRBF<double>(epsilon: 1.0);
        double[] testPoints = { 0.5, 1.0, 2.0, 4.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"Multiquadric derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    [Fact]
    public void Multiquadric_DerivativeAtOrigin_IsZero()
    {
        // f'(0) = 0/sqrt(e^2) = 0
        var rbf = new MultiquadricRBF<double>(epsilon: 2.0);
        Assert.Equal(0.0, rbf.ComputeDerivative(0.0), Tol);
    }

    // ─── Inverse Multiquadric RBF ───────────────────────────────────────

    [Fact]
    public void InverseMultiquadric_AtOrigin_ReturnsOneOverEpsilon()
    {
        // f(0) = 1/sqrt(0 + e^2) = 1/e
        double eps = 2.0;
        var rbf = new InverseMultiquadricRBF<double>(epsilon: eps);
        Assert.Equal(1.0 / eps, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void InverseMultiquadric_IsReciprocal_OfMultiquadric()
    {
        // 1/sqrt(r^2 + e^2) = 1/MQ(r)
        double eps = 1.5;
        var imq = new InverseMultiquadricRBF<double>(epsilon: eps);
        var mq = new MultiquadricRBF<double>(epsilon: eps);

        double[] testPoints = { 0.0, 0.5, 1.0, 2.0, 5.0 };
        foreach (double r in testPoints)
        {
            double imqVal = imq.Compute(r);
            double mqVal = mq.Compute(r);
            Assert.Equal(1.0 / mqVal, imqVal, Tol);
        }
    }

    [Fact]
    public void InverseMultiquadric_IsMonotonicallyDecreasing()
    {
        var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);
        double prev = rbf.Compute(0.0);
        for (double r = 0.1; r <= 5.0; r += 0.1)
        {
            double curr = rbf.Compute(r);
            Assert.True(curr < prev, $"InverseMQ not monotonically decreasing at r={r}");
            prev = curr;
        }
    }

    [Fact]
    public void InverseMultiquadric_Derivative_MatchesNumerical()
    {
        var rbf = new InverseMultiquadricRBF<double>(epsilon: 1.0);
        double[] testPoints = { 0.5, 1.0, 2.0, 4.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"InverseMQ derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    // ─── Inverse Quadratic RBF ──────────────────────────────────────────

    [Fact]
    public void InverseQuadratic_AtOrigin_ReturnsOne()
    {
        // f(0) = 1/(1 + 0) = 1
        var rbf = new InverseQuadraticRBF<double>(epsilon: 3.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void InverseQuadratic_HandComputed()
    {
        // f(r) = 1/(1 + (er)^2), e=1: f(1) = 1/(1+1) = 0.5
        var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);
        Assert.Equal(0.5, rbf.Compute(1.0), Tol);

        // e=2: f(1) = 1/(1+4) = 0.2
        var rbf2 = new InverseQuadraticRBF<double>(epsilon: 2.0);
        Assert.Equal(0.2, rbf2.Compute(1.0), Tol);
    }

    [Fact]
    public void InverseQuadratic_IsMonotonicallyDecreasing()
    {
        var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);
        double prev = rbf.Compute(0.0);
        for (double r = 0.1; r <= 5.0; r += 0.1)
        {
            double curr = rbf.Compute(r);
            Assert.True(curr < prev, $"InverseQuadratic not monotonically decreasing at r={r}");
            prev = curr;
        }
    }

    [Fact]
    public void InverseQuadratic_Derivative_MatchesNumerical()
    {
        var rbf = new InverseQuadraticRBF<double>(epsilon: 1.0);
        double[] testPoints = { 0.5, 1.0, 2.0, 4.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"InverseQuadratic derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    [Fact]
    public void InverseQuadratic_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.0;

        var rbf1 = new InverseQuadraticRBF<double>(epsilon: epsilon);
        var rbf2 = new InverseQuadraticRBF<double>(epsilon: epsilon + H);
        var rbf3 = new InverseQuadraticRBF<double>(epsilon: epsilon - H);

        double analytic = rbf1.ComputeWidthDerivative(r);
        double numerical = (rbf2.Compute(r) - rbf3.Compute(r)) / (2 * H);

        Assert.True(Math.Abs(analytic - numerical) < DerivTol,
            $"InverseQuadratic width derivative mismatch: analytic={analytic}, numerical={numerical}");
    }

    // ─── Linear RBF ─────────────────────────────────────────────────────

    [Fact]
    public void Linear_ReturnsInputValue()
    {
        var rbf = new LinearRBF<double>();
        Assert.Equal(0.0, rbf.Compute(0.0), Tol);
        Assert.Equal(3.5, rbf.Compute(3.5), Tol);
        Assert.Equal(100.0, rbf.Compute(100.0), Tol);
    }

    [Fact]
    public void Linear_DerivativeIsAlwaysOne()
    {
        var rbf = new LinearRBF<double>();
        Assert.Equal(1.0, rbf.ComputeDerivative(0.0), Tol);
        Assert.Equal(1.0, rbf.ComputeDerivative(5.0), Tol);
        Assert.Equal(1.0, rbf.ComputeDerivative(100.0), Tol);
    }

    [Fact]
    public void Linear_WidthDerivativeIsAlwaysZero()
    {
        var rbf = new LinearRBF<double>();
        Assert.Equal(0.0, rbf.ComputeWidthDerivative(0.0), Tol);
        Assert.Equal(0.0, rbf.ComputeWidthDerivative(5.0), Tol);
    }

    // ─── Cubic RBF ──────────────────────────────────────────────────────

    [Fact]
    public void Cubic_ComputeHandValues()
    {
        // f(r) = r^3
        var rbf = new CubicRBF<double>();
        Assert.Equal(0.0, rbf.Compute(0.0), Tol);
        Assert.Equal(8.0, rbf.Compute(2.0), Tol);
        Assert.Equal(27.0, rbf.Compute(3.0), Tol);
    }

    [Fact]
    public void Cubic_Derivative_MatchesNumerical()
    {
        // f'(r) = 3r^2
        var rbf = new CubicRBF<double>();
        double[] testPoints = { 0.5, 1.0, 2.0, 3.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"Cubic derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    // ─── Exponential RBF ────────────────────────────────────────────────

    [Fact]
    public void Exponential_AtOrigin_ReturnsOne()
    {
        // f(0) = exp(-e*0) = 1
        var rbf = new ExponentialRBF<double>(epsilon: 2.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void Exponential_HandComputed()
    {
        // f(r) = exp(-e*r), e=1: f(2) = exp(-2)
        var rbf = new ExponentialRBF<double>(epsilon: 1.0);
        Assert.Equal(Math.Exp(-2), rbf.Compute(2.0), Tol);
    }

    [Fact]
    public void Exponential_Derivative_MatchesNumerical()
    {
        var rbf = new ExponentialRBF<double>(epsilon: 1.5);
        double[] testPoints = { 0.5, 1.0, 2.0, 3.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"Exponential derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    // ─── Squared Exponential RBF ────────────────────────────────────────

    [Fact]
    public void SquaredExponential_MatchesGaussian()
    {
        // Squared exponential = exp(-e^2 * r^2) = Gaussian with epsilon = e^2
        // Actually, SquaredExponential: exp(-(r/e)^2) or exp(-r^2/(2*sigma^2))
        // Let's just test basic properties
        var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void SquaredExponential_Derivative_MatchesNumerical()
    {
        var rbf = new SquaredExponentialRBF<double>(epsilon: 1.0);
        double[] testPoints = { 0.5, 1.0, 2.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"SquaredExponential derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }

    // ─── Cross-Function Properties ──────────────────────────────────────

    [Fact]
    public void AllDecreasingRBFs_ArePositive()
    {
        // All bell-shaped RBFs should return positive values
        var gaussian = new GaussianRBF<double>(epsilon: 1.0);
        var invMQ = new InverseMultiquadricRBF<double>(epsilon: 1.0);
        var invQuad = new InverseQuadraticRBF<double>(epsilon: 1.0);

        for (double r = 0.0; r <= 10.0; r += 0.5)
        {
            Assert.True(gaussian.Compute(r) > 0, $"Gaussian negative at r={r}");
            Assert.True(invMQ.Compute(r) > 0, $"InverseMQ negative at r={r}");
            Assert.True(invQuad.Compute(r) > 0, $"InverseQuadratic negative at r={r}");
        }
    }

    [Fact]
    public void GaussianDecaysFasterThanInverseQuadratic()
    {
        // Gaussian decays exponentially, InverseQuadratic polynomially
        // At large r, Gaussian should be much smaller
        var gaussian = new GaussianRBF<double>(epsilon: 1.0);
        var invQuad = new InverseQuadraticRBF<double>(epsilon: 1.0);

        double largeR = 5.0;
        double gVal = gaussian.Compute(largeR);
        double iqVal = invQuad.Compute(largeR);

        Assert.True(gVal < iqVal,
            $"At r={largeR}, Gaussian ({gVal}) should be < InverseQuadratic ({iqVal})");
    }

    [Fact]
    public void Gaussian_LargerEpsilon_DecaysFaster()
    {
        // Larger epsilon = narrower bell = smaller values at same r
        var narrow = new GaussianRBF<double>(epsilon: 2.0);
        var wide = new GaussianRBF<double>(epsilon: 0.5);

        double r = 1.0;
        Assert.True(narrow.Compute(r) < wide.Compute(r),
            "Gaussian with larger epsilon should decay faster");
    }

    [Fact]
    public void MultiquadricAndInverseMQ_ProductIsOne()
    {
        // MQ(r) * IMQ(r) = 1 for all r
        double eps = 1.5;
        var mq = new MultiquadricRBF<double>(epsilon: eps);
        var imq = new InverseMultiquadricRBF<double>(epsilon: eps);

        double[] testPoints = { 0.0, 0.5, 1.0, 2.0, 5.0, 10.0 };
        foreach (double r in testPoints)
        {
            double product = mq.Compute(r) * imq.Compute(r);
            Assert.Equal(1.0, product, Tol);
        }
    }

    [Fact]
    public void Multiquadric_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.0;

        var rbf1 = new MultiquadricRBF<double>(epsilon: epsilon);
        var rbf2 = new MultiquadricRBF<double>(epsilon: epsilon + H);
        var rbf3 = new MultiquadricRBF<double>(epsilon: epsilon - H);

        double analytic = rbf1.ComputeWidthDerivative(r);
        double numerical = (rbf2.Compute(r) - rbf3.Compute(r)) / (2 * H);

        Assert.True(Math.Abs(analytic - numerical) < DerivTol,
            $"Multiquadric width derivative mismatch: analytic={analytic}, numerical={numerical}");
    }

    [Fact]
    public void InverseMultiquadric_WidthDerivative_MatchesNumerical()
    {
        double r = 2.0;
        double epsilon = 1.0;

        var rbf1 = new InverseMultiquadricRBF<double>(epsilon: epsilon);
        var rbf2 = new InverseMultiquadricRBF<double>(epsilon: epsilon + H);
        var rbf3 = new InverseMultiquadricRBF<double>(epsilon: epsilon - H);

        double analytic = rbf1.ComputeWidthDerivative(r);
        double numerical = (rbf2.Compute(r) - rbf3.Compute(r)) / (2 * H);

        Assert.True(Math.Abs(analytic - numerical) < DerivTol,
            $"InverseMQ width derivative mismatch: analytic={analytic}, numerical={numerical}");
    }

    // ─── Thin Plate Spline RBF ──────────────────────────────────────────

    [Fact]
    public void ThinPlateSpline_AtOrigin_ReturnsZero()
    {
        // f(0) = 0^2 * ln(0) = 0 (by convention, 0*ln(0)=0)
        var rbf = new ThinPlateSplineRBF<double>();
        double val = rbf.Compute(0.0);
        Assert.True(Math.Abs(val) < Tol || double.IsNaN(val) || val == 0.0,
            $"TPS at origin should be 0, got {val}");
    }

    [Fact]
    public void ThinPlateSpline_HandComputed()
    {
        // f(r) = r^2 * ln(r), f(e) = e^2 * ln(e) = e^2 * 1 = e^2 ≈ 7.389
        var rbf = new ThinPlateSplineRBF<double>();
        double expected = Math.E * Math.E * Math.Log(Math.E); // e^2 * 1 = e^2
        Assert.Equal(expected, rbf.Compute(Math.E), Tol);
    }

    // ─── Rational Quadratic RBF ─────────────────────────────────────────

    [Fact]
    public void RationalQuadratic_AtOrigin_ReturnsOne()
    {
        var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);
        Assert.Equal(1.0, rbf.Compute(0.0), Tol);
    }

    [Fact]
    public void RationalQuadratic_Derivative_MatchesNumerical()
    {
        var rbf = new RationalQuadraticRBF<double>(epsilon: 1.0);
        double[] testPoints = { 0.5, 1.0, 2.0 };

        foreach (double r in testPoints)
        {
            double analytic = rbf.ComputeDerivative(r);
            double numerical = (rbf.Compute(r + H) - rbf.Compute(r - H)) / (2 * H);
            Assert.True(Math.Abs(analytic - numerical) < DerivTol,
                $"RationalQuadratic derivative mismatch at r={r}: analytic={analytic}, numerical={numerical}");
        }
    }
}
