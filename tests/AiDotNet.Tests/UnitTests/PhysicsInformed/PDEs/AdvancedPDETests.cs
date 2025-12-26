using System;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PDEs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.PDEs;

/// <summary>
/// Tests for advanced PDE implementations: Black-Scholes, Linear Elasticity,
/// Advection-Diffusion, and Korteweg-de Vries equations.
/// </summary>
public class AdvancedPDETests
{
    private const double Tolerance = 1e-10;

    #region Black-Scholes Equation Tests

    /// <summary>
    /// Tests Black-Scholes constructor with valid parameters.
    /// </summary>
    [Fact]
    public void BlackScholesEquation_Constructor_WithValidParameters_Succeeds()
    {
        var pde = new BlackScholesEquation<double>(0.2, 0.05);

        Assert.Equal(2, pde.InputDimension);
        Assert.Equal(1, pde.OutputDimension);
        Assert.Contains("Black-Scholes", pde.Name);
    }

    /// <summary>
    /// Tests Black-Scholes constructor with double convenience overload.
    /// </summary>
    [Fact]
    public void BlackScholesEquation_Constructor_DoubleOverload_Succeeds()
    {
        var pde = new BlackScholesEquation<double>(volatility: 0.3, riskFreeRate: 0.02);

        Assert.NotNull(pde);
        Assert.Contains("0.3", pde.Name);
        Assert.Contains("0.02", pde.Name);
    }

    /// <summary>
    /// Tests Black-Scholes constructor throws on non-positive volatility.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    public void BlackScholesEquation_Constructor_InvalidVolatility_Throws(double volatility)
    {
        Assert.Throws<ArgumentException>(() =>
            new BlackScholesEquation<double>(volatility, 0.05));
    }

    /// <summary>
    /// Tests Black-Scholes with a constant option value (trivial case).
    /// For constant V with all derivatives zero except dV/dt = 0, the residual is -rV.
    /// </summary>
    [Fact]
    public void BlackScholesEquation_ConstantValue_ComputesExpectedResidual()
    {
        double sigma = 0.2;
        double r = 0.05;
        var pde = new BlackScholesEquation<double>(sigma, r);

        double S = 100.0;
        double t = 0.5;
        double V = 10.0;

        var inputs = new[] { S, t };
        var outputs = new[] { V };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.0, 0.0 } },  // dV/dS = 0, dV/dt = 0
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } });  // d2V/dS2 = 0

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        // For constant V: residual = 0 + 0 + 0 - rV = -rV
        double expected = -r * V;
        Assert.True(Math.Abs(residual - expected) < Tolerance,
            $"Expected residual {expected}, got {residual}");
    }

    /// <summary>
    /// Tests Black-Scholes with linear option value V = S.
    /// This is like a forward contract at maturity.
    /// </summary>
    [Fact]
    public void BlackScholesEquation_LinearValue_ComputesCorrectResidual()
    {
        double sigma = 0.2;
        double r = 0.05;
        var pde = new BlackScholesEquation<double>(sigma, r);

        double S = 100.0;
        double t = 0.5;
        double V = S;  // V = S

        var inputs = new[] { S, t };
        var outputs = new[] { V };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 1.0, 0.0 } },  // dV/dS = 1, dV/dt = 0
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } });  // d2V/dS2 = 0

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        // For V = S: residual = 0 + 0 + rS*1 - rS = 0
        Assert.True(Math.Abs(residual) < Tolerance,
            $"Black-Scholes residual should be ~0 for V=S, got {residual}");
    }

    /// <summary>
    /// Tests Black-Scholes gradient computation.
    /// </summary>
    [Fact]
    public void BlackScholesEquation_GradientComputation_ReturnsExpectedValues()
    {
        double sigma = 0.2;
        double r = 0.05;
        var pde = new BlackScholesEquation<double>(sigma, r);

        double S = 100.0;
        double t = 0.5;

        var inputs = new[] { S, t };
        var outputs = new[] { 10.0 };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.5, 0.1 } },
            new[, ,] { { { 0.02, 0.0 }, { 0.0, 0.0 } } });

        var gradient = pde.ComputeResidualGradient(inputs, outputs, derivatives);

        // Check gradient values
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], 6);  // dR/d(dV/dt) = 1
        Assert.Equal(r * S, gradient.FirstDerivatives[0, 0], 6);  // dR/d(dV/dS) = rS
        Assert.Equal(0.5 * sigma * sigma * S * S, gradient.SecondDerivatives[0, 0, 0], 6);  // dR/d(d2V/dS2)
        Assert.Equal(-r, gradient.OutputGradients[0], 6);  // dR/dV = -r
    }

    #endregion

    #region Linear Elasticity Equation Tests

    /// <summary>
    /// Tests Linear Elasticity constructor with valid Lame parameters.
    /// </summary>
    [Fact]
    public void LinearElasticityEquation_Constructor_WithValidParameters_Succeeds()
    {
        var pde = new LinearElasticityEquation<double>(1.0, 1.0);

        Assert.Equal(2, pde.InputDimension);
        Assert.Equal(2, pde.OutputDimension);
        Assert.Contains("Linear Elasticity", pde.Name);
    }

    /// <summary>
    /// Tests Linear Elasticity constructor throws on non-positive shear modulus.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    public void LinearElasticityEquation_Constructor_InvalidMu_Throws(double mu)
    {
        Assert.Throws<ArgumentException>(() =>
            new LinearElasticityEquation<double>(1.0, mu));
    }

    /// <summary>
    /// Tests Linear Elasticity factory method from engineering constants.
    /// </summary>
    [Fact]
    public void LinearElasticityEquation_FromEngineeringConstants_CreatesValidEquation()
    {
        double E = 200e9;  // Steel-like Young's modulus
        double nu = 0.3;    // Poisson's ratio

        var pde = LinearElasticityEquation<double>.FromEngineeringConstants(E, nu);

        Assert.NotNull(pde);
        Assert.Equal(2, pde.InputDimension);
        Assert.Equal(2, pde.OutputDimension);
    }

    /// <summary>
    /// Tests Linear Elasticity factory throws on invalid Poisson's ratio.
    /// </summary>
    [Theory]
    [InlineData(-1.1)]
    [InlineData(0.5)]
    [InlineData(0.6)]
    public void LinearElasticityEquation_FromEngineeringConstants_InvalidPoisson_Throws(double nu)
    {
        Assert.Throws<ArgumentException>(() =>
            LinearElasticityEquation<double>.FromEngineeringConstants(200e9, nu));
    }

    /// <summary>
    /// Tests Linear Elasticity with zero displacement (trivial equilibrium).
    /// </summary>
    [Fact]
    public void LinearElasticityEquation_ZeroDisplacement_ReturnsZeroResidual()
    {
        var pde = new LinearElasticityEquation<double>(1.0, 1.0);

        var inputs = new[] { 0.5, 0.5 };
        var outputs = new[] { 0.0, 0.0 };  // u = 0, v = 0

        // All derivatives zero
        var firstDerivatives = new double[2, 2];
        var secondDerivatives = new double[2, 2, 2];

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Linear elasticity residual should be ~0 for zero displacement, got {residual}");
    }

    /// <summary>
    /// Tests Linear Elasticity with uniform strain (constant displacement gradient).
    /// </summary>
    [Fact]
    public void LinearElasticityEquation_UniformStrain_HasZeroSecondDerivatives()
    {
        var pde = new LinearElasticityEquation<double>(1.0, 1.0);

        var inputs = new[] { 0.5, 0.5 };
        var outputs = new[] { 0.1, 0.05 };  // Non-zero displacements

        // Uniform strain: first derivatives constant, second derivatives zero
        var firstDerivatives = new double[2, 2];
        firstDerivatives[0, 0] = 0.01;  // du/dx
        firstDerivatives[0, 1] = 0.0;   // du/dy
        firstDerivatives[1, 0] = 0.0;   // dv/dx
        firstDerivatives[1, 1] = 0.01;  // dv/dy

        var secondDerivatives = new double[2, 2, 2];  // All zero

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        // With zero second derivatives and no body forces, residual should be zero
        Assert.True(Math.Abs(residual) < Tolerance,
            $"Linear elasticity residual should be ~0 for uniform strain, got {residual}");
    }

    #endregion

    #region Advection-Diffusion Equation Tests

    /// <summary>
    /// Tests 1D Advection-Diffusion constructor.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_1D_Constructor_Succeeds()
    {
        var pde = new AdvectionDiffusionEquation<double>(0.1, 1.0);

        Assert.Equal(2, pde.InputDimension);  // [x, t]
        Assert.Equal(1, pde.OutputDimension);  // [c]
        Assert.Contains("1D", pde.Name);
    }

    /// <summary>
    /// Tests 2D Advection-Diffusion constructor.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_2D_Constructor_Succeeds()
    {
        // Explicitly specify all 4 parameters to ensure 2D constructor is called
        var pde = new AdvectionDiffusionEquation<double>(
            diffusionCoeff: 0.1,
            velocityX: 1.0,
            velocityY: 0.5,
            sourceTerm: 0.0);

        Assert.Equal(3, pde.InputDimension);  // [x, y, t]
        Assert.Equal(1, pde.OutputDimension);  // [c]
        Assert.Contains("2D", pde.Name);
    }

    /// <summary>
    /// Tests Advection-Diffusion constructor throws on negative diffusion coefficient.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_NegativeDiffusion_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new AdvectionDiffusionEquation<double>(-0.1, 1.0));
    }

    /// <summary>
    /// Tests 1D pure diffusion (no advection) with Gaussian solution.
    /// For pure diffusion, the equation reduces to heat equation.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_1D_PureDiffusion_ConstantSolution_ZeroResidual()
    {
        double D = 0.1;
        var pde = new AdvectionDiffusionEquation<double>(D, 0.0);  // No advection

        // Constant concentration: all derivatives zero
        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 1.0 };  // c = 1

        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.0, 0.0 } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Advection-diffusion residual should be ~0 for constant concentration, got {residual}");
    }

    /// <summary>
    /// Tests 1D pure advection (no diffusion) with traveling wave.
    /// Solution: c(x,t) = f(x - vt) for any function f.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_1D_PureAdvection_TravelingWave_ZeroResidual()
    {
        double v = 1.0;
        var pde = new AdvectionDiffusionEquation<double>(0.0, v);  // No diffusion

        double x = 0.5;
        double t = 0.1;
        double xi = x - v * t;  // Characteristic coordinate

        // Traveling wave: c = sin(x - vt)
        double c = Math.Sin(xi);
        double dcdx = Math.Cos(xi);
        double dcdt = -v * Math.Cos(xi);

        var inputs = new[] { x, t };
        var outputs = new[] { c };

        var derivatives = CreateDerivatives(1, 2,
            new[,] { { dcdx, dcdt } },
            new[, ,] { { { -Math.Sin(xi), 0.0 }, { 0.0, 0.0 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Advection equation residual should be ~0 for traveling wave, got {residual}");
    }

    /// <summary>
    /// Tests 2D advection-diffusion with uniform concentration.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_2D_UniformConcentration_ZeroResidual()
    {
        // Explicitly specify all 4 parameters to ensure 2D constructor is called
        var pde = new AdvectionDiffusionEquation<double>(
            diffusionCoeff: 0.1,
            velocityX: 1.0,
            velocityY: 0.5,
            sourceTerm: 0.0);

        var inputs = new[] { 0.5, 0.5, 0.1 };
        var outputs = new[] { 1.0 };

        // All derivatives zero for uniform concentration
        var firstDerivatives = new double[1, 3];
        var secondDerivatives = new double[1, 3, 3];

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"2D Advection-diffusion residual should be ~0 for uniform concentration, got {residual}");
    }

    /// <summary>
    /// Tests Advection-Diffusion gradient computation for 1D case.
    /// </summary>
    [Fact]
    public void AdvectionDiffusionEquation_1D_GradientComputation_ReturnsExpectedValues()
    {
        double D = 0.1;
        double v = 1.0;
        var pde = new AdvectionDiffusionEquation<double>(D, v);

        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 1.0 };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.5, 0.3 } },
            new[, ,] { { { 0.2, 0.0 }, { 0.0, 0.0 } } });

        var gradient = pde.ComputeResidualGradient(inputs, outputs, derivatives);

        // Expected gradients for 1D: R = dc/dt + v*dc/dx - D*d2c/dx2
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], 6);  // dR/d(dc/dt) = 1
        Assert.Equal(v, gradient.FirstDerivatives[0, 0], 6);    // dR/d(dc/dx) = v
        Assert.Equal(-D, gradient.SecondDerivatives[0, 0, 0], 6);  // dR/d(d2c/dx2) = -D
    }

    #endregion

    #region Korteweg-de Vries Equation Tests

    /// <summary>
    /// Tests KdV equation constructor with default canonical form.
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_Canonical_HasExpectedParameters()
    {
        var pde = KortewegDeVriesEquation<double>.Canonical();

        Assert.Equal(2, pde.InputDimension);
        Assert.Equal(1, pde.OutputDimension);
        Assert.Contains("Korteweg-de Vries", pde.Name);
        Assert.Contains("6", pde.Name);  // alpha = 6 in canonical form
    }

    /// <summary>
    /// Tests KdV equation constructor with physical form.
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_Physical_HasExpectedParameters()
    {
        var pde = KortewegDeVriesEquation<double>.Physical();

        Assert.Contains("1", pde.Name);  // alpha = 1 in physical form
    }

    /// <summary>
    /// Tests KdV equation constructor throws on zero dispersion coefficient.
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_ZeroDispersion_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new KortewegDeVriesEquation<double>(6.0, 0.0));
    }

    /// <summary>
    /// Tests KdV with constant solution (all derivatives zero).
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_ConstantSolution_ZeroResidual()
    {
        var pde = KortewegDeVriesEquation<double>.Canonical();

        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 1.0 };  // u = constant

        var derivatives = CreateDerivativesWithThird(1, 2,
            new[,] { { 0.0, 0.0 } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } },
            new[, , ,] { { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"KdV residual should be ~0 for constant solution, got {residual}");
    }

    /// <summary>
    /// Tests KdV with zero solution.
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_ZeroSolution_ZeroResidual()
    {
        var pde = KortewegDeVriesEquation<double>.Canonical();

        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 0.0 };  // u = 0

        var derivatives = CreateDerivativesWithThird(1, 2,
            new[,] { { 0.0, 0.0 } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } },
            new[, , ,] { { { { 0.0, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"KdV residual should be ~0 for zero solution, got {residual}");
    }

    /// <summary>
    /// Tests KdV residual computation for non-trivial case.
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_ComputeResidual_ReturnsFiniteValue()
    {
        var pde = KortewegDeVriesEquation<double>.Canonical();

        double x = 0.5;
        double t = 0.1;
        double u = 0.5;

        // Some non-zero derivatives
        double dudx = 0.1;
        double dudt = 0.2;
        double d3udx3 = 0.05;

        var inputs = new[] { x, t };
        var outputs = new[] { u };

        var derivatives = CreateDerivativesWithThird(1, 2,
            new[,] { { dudx, dudt } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } },
            new[, , ,] { { { { d3udx3, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        // Residual = du/dt + 6*u*du/dx + d3u/dx3
        double expected = dudt + 6.0 * u * dudx + d3udx3;
        Assert.True(Math.Abs(residual - expected) < Tolerance,
            $"Expected residual {expected}, got {residual}");
    }

    /// <summary>
    /// Tests KdV gradient computation.
    /// </summary>
    [Fact]
    public void KortewegDeVriesEquation_GradientComputation_ReturnsExpectedValues()
    {
        double alpha = 6.0;
        double beta = 1.0;
        var pde = new KortewegDeVriesEquation<double>(alpha, beta);

        double u = 0.5;
        double dudx = 0.1;

        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { u };

        var derivatives = CreateDerivativesWithThird(1, 2,
            new[,] { { dudx, 0.2 } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } },
            new[, , ,] { { { { 0.05, 0.0 }, { 0.0, 0.0 } }, { { 0.0, 0.0 }, { 0.0, 0.0 } } } });

        var gradient = pde.ComputeResidualGradient(inputs, outputs, derivatives);

        // Expected gradients for KdV: R = du/dt + alpha*u*du/dx + beta*d3u/dx3
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], 6);  // dR/d(du/dt) = 1
        Assert.Equal(alpha * u, gradient.FirstDerivatives[0, 0], 6);  // dR/d(du/dx) = alpha*u
        Assert.Equal(beta, gradient.ThirdDerivatives[0, 0, 0, 0], 6);  // dR/d(d3u/dx3) = beta
        Assert.Equal(alpha * dudx, gradient.OutputGradients[0], 6);  // dR/du = alpha*du/dx
    }

    #endregion

    #region Integration Tests

    /// <summary>
    /// Verifies all new PDEs implement required interfaces correctly.
    /// </summary>
    [Fact]
    public void AllNewPDEs_ImplementRequiredInterfaces()
    {
        // Black-Scholes
        var bs = new BlackScholesEquation<double>(0.2, 0.05);
        Assert.IsAssignableFrom<IPDESpecification<double>>(bs);
        Assert.IsAssignableFrom<IPDEResidualGradient<double>>(bs);

        // Linear Elasticity
        var le = new LinearElasticityEquation<double>(1.0, 1.0);
        Assert.IsAssignableFrom<IPDESpecification<double>>(le);
        Assert.IsAssignableFrom<IPDEResidualGradient<double>>(le);

        // Advection-Diffusion
        var ad = new AdvectionDiffusionEquation<double>(0.1, 1.0);
        Assert.IsAssignableFrom<IPDESpecification<double>>(ad);
        Assert.IsAssignableFrom<IPDEResidualGradient<double>>(ad);

        // Korteweg-de Vries
        var kdv = KortewegDeVriesEquation<double>.Canonical();
        Assert.IsAssignableFrom<IPDESpecification<double>>(kdv);
        Assert.IsAssignableFrom<IPDEResidualGradient<double>>(kdv);
    }

    /// <summary>
    /// Verifies all new PDEs have sensible Name properties.
    /// </summary>
    [Fact]
    public void AllNewPDEs_HaveDescriptiveNames()
    {
        var bs = new BlackScholesEquation<double>(0.2, 0.05);
        Assert.Contains("Black-Scholes", bs.Name);

        var le = new LinearElasticityEquation<double>(1.0, 1.0);
        Assert.Contains("Elasticity", le.Name);

        var ad = new AdvectionDiffusionEquation<double>(0.1, 1.0);
        Assert.Contains("Advection", ad.Name);
        Assert.Contains("Diffusion", ad.Name);

        var kdv = KortewegDeVriesEquation<double>.Canonical();
        Assert.Contains("Korteweg", kdv.Name);
    }

    #endregion

    #region Helper Methods

    private static PDEDerivatives<double> CreateDerivatives(
        int numOutputs,
        int numInputs,
        double[,] firstDerivatives,
        double[,,] secondDerivatives)
    {
        return new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };
    }

    private static PDEDerivatives<double> CreateDerivativesWithThird(
        int numOutputs,
        int numInputs,
        double[,] firstDerivatives,
        double[,,] secondDerivatives,
        double[,,,] thirdDerivatives)
    {
        return new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives,
            ThirdDerivatives = thirdDerivatives
        };
    }

    #endregion
}
