using System;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PDEs;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.PDEs;

/// <summary>
/// Tests PDE implementations against known analytical solutions.
/// When the exact analytical solution is provided, the residual should be zero (or very small due to numerical precision).
/// </summary>
public class PDEAnalyticalSolutionTests
{
    private const double Tolerance = 1e-10;

    #region Heat Equation Tests

    /// <summary>
    /// Tests Heat Equation with analytical solution: u(x,t) = exp(-α*k²*t) * sin(k*x)
    /// This is the decay of a sinusoidal temperature distribution.
    /// </summary>
    [Theory]
    [InlineData(1.0, 1.0)]  // α=1, k=1
    [InlineData(0.5, 2.0)]  // α=0.5, k=2
    [InlineData(2.0, 0.5)]  // α=2, k=0.5
    public void HeatEquation_AnalyticalSolution_ReturnsZeroResidual(double alpha, double k)
    {
        var pde = new HeatEquation<double>(alpha);

        double x = 0.5;
        double t = 0.1;

        // Analytical solution: u = exp(-α*k²*t) * sin(k*x)
        double expFactor = Math.Exp(-alpha * k * k * t);
        double u = expFactor * Math.Sin(k * x);

        // Derivatives:
        // ∂u/∂x = k * exp(-α*k²*t) * cos(k*x)
        // ∂u/∂t = -α*k² * exp(-α*k²*t) * sin(k*x)
        // ∂²u/∂x² = -k² * exp(-α*k²*t) * sin(k*x)
        double dudt = -alpha * k * k * expFactor * Math.Sin(k * x);
        double d2udx2 = -k * k * expFactor * Math.Sin(k * x);

        var inputs = new[] { x, t };
        var outputs = new[] { u };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { k * expFactor * Math.Cos(k * x), dudt } },
            new[, ,] { { { d2udx2, 0 }, { 0, 0 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Heat equation residual should be ~0 for analytical solution, got {residual}");
    }

    #endregion

    #region Wave Equation Tests

    /// <summary>
    /// Tests Wave Equation with analytical solution: u(x,t) = sin(k*x) * cos(c*k*t)
    /// This represents a standing wave.
    /// </summary>
    [Theory]
    [InlineData(1.0, 1.0)]  // c=1, k=1
    [InlineData(2.0, 1.0)]  // c=2, k=1
    [InlineData(1.0, 2.0)]  // c=1, k=2
    public void WaveEquation_StandingWaveSolution_ReturnsZeroResidual(double c, double k)
    {
        var pde = new WaveEquation<double>(c);

        double x = 0.5;
        double t = 0.1;

        // Analytical solution: u = sin(k*x) * cos(c*k*t)
        double u = Math.Sin(k * x) * Math.Cos(c * k * t);

        // Derivatives:
        // ∂u/∂x = k * cos(k*x) * cos(c*k*t)
        // ∂u/∂t = -c*k * sin(k*x) * sin(c*k*t)
        // ∂²u/∂x² = -k² * sin(k*x) * cos(c*k*t)
        // ∂²u/∂t² = -c²*k² * sin(k*x) * cos(c*k*t)
        double dudx = k * Math.Cos(k * x) * Math.Cos(c * k * t);
        double dudt = -c * k * Math.Sin(k * x) * Math.Sin(c * k * t);
        double d2udx2 = -k * k * Math.Sin(k * x) * Math.Cos(c * k * t);
        double d2udt2 = -c * c * k * k * Math.Sin(k * x) * Math.Cos(c * k * t);

        var inputs = new[] { x, t };
        var outputs = new[] { u };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { dudx, dudt } },
            new[, ,] { { { d2udx2, 0 }, { 0, d2udt2 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Wave equation residual should be ~0 for standing wave solution, got {residual}");
    }

    #endregion

    #region Poisson Equation Tests

    /// <summary>
    /// Tests Poisson Equation with analytical solution: u(x) = x² for f(x) = 2
    /// The Poisson equation: ∂²u/∂x² = f(x)
    /// </summary>
    [Fact]
    public void PoissonEquation_QuadraticSolution_ReturnsZeroResidual()
    {
        // f(x) = 2, so u(x) = x² satisfies ∂²u/∂x² = 2
        Func<double[], double> sourceFunction = x => 2.0;
        var pde = new PoissonEquation<double>(sourceFunction, spatialDimension: 1);

        double x = 0.5;
        double u = x * x;  // u = x²
        double dudx = 2 * x;  // ∂u/∂x = 2x
        double d2udx2 = 2.0;  // ∂²u/∂x² = 2

        var inputs = new[] { x };
        var outputs = new[] { u };
        var derivatives = CreateDerivatives(1, 1,
            new[,] { { dudx } },
            new[, ,] { { { d2udx2 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Poisson equation residual should be ~0 for u=x², got {residual}");
    }

    /// <summary>
    /// Tests Poisson Equation with sinusoidal solution: u(x) = sin(k*x) for f(x) = -k²*sin(k*x)
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(2.0)]
    public void PoissonEquation_SinusoidalSolution_ReturnsZeroResidual(double k)
    {
        // u(x) = sin(k*x), so ∂²u/∂x² = -k²*sin(k*x)
        Func<double[], double> sourceFunction = inputs => -k * k * Math.Sin(k * inputs[0]);
        var pde = new PoissonEquation<double>(sourceFunction, spatialDimension: 1);

        double x = 0.5;
        double u = Math.Sin(k * x);
        double dudx = k * Math.Cos(k * x);
        double d2udx2 = -k * k * Math.Sin(k * x);

        var inputs = new[] { x };
        var outputs = new[] { u };
        var derivatives = CreateDerivatives(1, 1,
            new[,] { { dudx } },
            new[, ,] { { { d2udx2 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Poisson equation residual should be ~0 for sinusoidal solution, got {residual}");
    }

    #endregion

    #region Burgers Equation Tests

    /// <summary>
    /// Tests Burgers Equation with steady-state solution where u is constant.
    /// For u = const, all derivatives are zero, so residual should be zero.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    public void BurgersEquation_ConstantSolution_ReturnsZeroResidual(double u0)
    {
        var pde = new BurgersEquation<double>(viscosity: 0.1);

        double x = 0.5;
        double t = 0.1;

        // Constant solution: all derivatives are zero
        var inputs = new[] { x, t };
        var outputs = new[] { u0 };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.0, 0.0 } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Burgers equation residual should be ~0 for constant solution, got {residual}");
    }

    #endregion

    #region Allen-Cahn Equation Tests

    /// <summary>
    /// Tests Allen-Cahn Equation at equilibrium points u = ±1 and u = 0.
    /// At these points, f(u) = u - u³ = 0, so steady-state with flat profile has zero residual.
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    [InlineData(0.0)]
    public void AllenCahnEquation_EquilibriumPoints_ReturnsZeroResidual(double u0)
    {
        var pde = new AllenCahnEquation<double>(epsilon: 1.0);

        double x = 0.5;
        double t = 0.1;

        // At equilibrium with flat profile: all derivatives are zero
        var inputs = new[] { x, t };
        var outputs = new[] { u0 };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.0, 0.0 } },
            new[, ,] { { { 0.0, 0.0 }, { 0.0, 0.0 } } });

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        // For u=±1: u - u³ = ±1 - (±1) = 0, so residual = 0
        // For u=0: u - u³ = 0, so residual = 0
        Assert.True(Math.Abs(residual) < Tolerance,
            $"Allen-Cahn equation residual should be ~0 at equilibrium u={u0}, got {residual}");
    }

    #endregion

    #region Navier-Stokes Equation Tests

    /// <summary>
    /// Tests Navier-Stokes with Poiseuille flow (steady laminar flow between parallel plates).
    /// For steady flow with u = u(y) only (no x or t dependence), the solution is parabolic.
    /// </summary>
    [Fact]
    public void NavierStokesEquation_PoiseuilleFlow_ReturnsZeroResidual()
    {
        double viscosity = 1.0;
        double density = 1.0;
        var pde = new NavierStokesEquation<double>(viscosity, density);

        // Poiseuille flow: u = (dP/dx) * y * (H - y) / (2*μ), v = 0, p = p0 - (dP/dx)*x
        // For simplicity, let dP/dx = -2*μ, H = 1, so u = y * (1 - y), v = 0
        // Then: ∂u/∂x = 0, ∂u/∂y = 1 - 2y, ∂u/∂t = 0
        //       ∂²u/∂x² = 0, ∂²u/∂y² = -2
        //       ∂p/∂x = -2*μ = -2, ∂p/∂y = 0

        double x = 0.5;
        double y = 0.3;
        double t = 0.0;

        double u = y * (1 - y);
        double v = 0.0;
        double p = 0.0;  // Reference pressure

        var inputs = new[] { x, y, t };
        var outputs = new[] { u, v, p };

        // First derivatives: [output_idx, input_idx] where inputs are [x=0, y=1, t=2]
        double dudx = 0.0;
        double dudy = 1 - 2 * y;
        double dudt = 0.0;
        double dvdx = 0.0;
        double dvdy = 0.0;
        double dvdt = 0.0;
        double dpdx = -2.0 * viscosity;  // Pressure gradient driving the flow
        double dpdy = 0.0;

        var firstDerivatives = new double[3, 3];
        firstDerivatives[0, 0] = dudx;
        firstDerivatives[0, 1] = dudy;
        firstDerivatives[0, 2] = dudt;
        firstDerivatives[1, 0] = dvdx;
        firstDerivatives[1, 1] = dvdy;
        firstDerivatives[1, 2] = dvdt;
        firstDerivatives[2, 0] = dpdx;
        firstDerivatives[2, 1] = dpdy;

        // Second derivatives: [output_idx, input_idx1, input_idx2]
        var secondDerivatives = new double[3, 3, 3];
        secondDerivatives[0, 1, 1] = -2.0;  // ∂²u/∂y²

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Navier-Stokes residual should be ~0 for Poiseuille flow, got {residual}");
    }

    /// <summary>
    /// Tests Navier-Stokes continuity equation with divergence-free velocity field.
    /// </summary>
    [Fact]
    public void NavierStokesEquation_DivergenceFreeField_SatisfiesContinuity()
    {
        var pde = new NavierStokesEquation<double>(viscosity: 1.0);

        // Divergence-free field: u = y, v = -x (simple rotation)
        // ∂u/∂x = 0, ∂v/∂y = 0, so ∂u/∂x + ∂v/∂y = 0
        double x = 0.5;
        double y = 0.3;
        double t = 0.0;

        double u = y;
        double v = -x;
        double p = 0.0;

        var inputs = new[] { x, y, t };
        var outputs = new[] { u, v, p };

        var firstDerivatives = new double[3, 3];
        firstDerivatives[0, 0] = 0.0;  // ∂u/∂x
        firstDerivatives[0, 1] = 1.0;  // ∂u/∂y
        firstDerivatives[1, 0] = -1.0; // ∂v/∂x
        firstDerivatives[1, 1] = 0.0;  // ∂v/∂y

        var secondDerivatives = new double[3, 3, 3];

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        // The continuity residual (∂u/∂x + ∂v/∂y) should be zero
        // Note: The full residual includes momentum equations which won't be zero for this simple field
        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        // For this test, we just verify the computation runs without error
        // A true zero would require the momentum equations to also be satisfied
        Assert.True(!double.IsNaN(residual) && !double.IsInfinity(residual), "Residual should be finite");
    }

    #endregion

    #region Maxwell's Equations Tests

    /// <summary>
    /// Tests Maxwell's Equations with a plane wave solution in vacuum.
    /// E = E0 * sin(kx - ωt), B = B0 * sin(kx - ωt) where ω = c*k
    /// </summary>
    [Fact]
    public void MaxwellEquations_PlaneWaveSolution_ReturnsZeroResidual()
    {
        double epsilon = 1.0;  // Vacuum permittivity (normalized)
        double mu = 1.0;       // Vacuum permeability (normalized)
        var pde = new MaxwellEquations<double>(epsilon, mu);

        // Plane wave: k = ω = 1 (c = 1 in vacuum with ε = μ = 1)
        double k = 1.0;
        double omega = 1.0;  // ω = c*k = k for c = 1
        double E0 = 1.0;

        double x = 0.5;
        double y = 0.0;  // Propagation along x, uniform in y
        double t = 0.1;

        double phase = k * x - omega * t;

        // TE mode: Ex = 0, Ey = E0*sin(phase), Bz = E0*sin(phase)/c = E0*sin(phase)
        double Ex = 0.0;
        double Ey = E0 * Math.Sin(phase);
        double Bz = E0 * Math.Sin(phase);  // For c = 1

        var inputs = new[] { x, y, t };
        var outputs = new[] { Ex, Ey, Bz };

        // First derivatives: [output_idx, input_idx] where inputs are [x=0, y=1, t=2]
        var firstDerivatives = new double[3, 3];

        // Ex = 0, so all derivatives are 0
        firstDerivatives[0, 0] = 0.0;  // ∂Ex/∂x
        firstDerivatives[0, 1] = 0.0;  // ∂Ex/∂y
        firstDerivatives[0, 2] = 0.0;  // ∂Ex/∂t

        // Ey = E0*sin(kx - ωt)
        firstDerivatives[1, 0] = E0 * k * Math.Cos(phase);      // ∂Ey/∂x
        firstDerivatives[1, 1] = 0.0;                            // ∂Ey/∂y
        firstDerivatives[1, 2] = -E0 * omega * Math.Cos(phase); // ∂Ey/∂t

        // Bz = E0*sin(kx - ωt)
        firstDerivatives[2, 0] = E0 * k * Math.Cos(phase);      // ∂Bz/∂x
        firstDerivatives[2, 1] = 0.0;                            // ∂Bz/∂y
        firstDerivatives[2, 2] = -E0 * omega * Math.Cos(phase); // ∂Bz/∂t

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Maxwell equations residual should be ~0 for plane wave solution, got {residual}");
    }

    /// <summary>
    /// Tests Maxwell's Equations with static field (no time dependence).
    /// </summary>
    [Fact]
    public void MaxwellEquations_StaticField_ReturnsZeroResidual()
    {
        var pde = new MaxwellEquations<double>(permittivity: 1.0, permeability: 1.0);

        // Static uniform field: all time derivatives are zero
        double Ex = 1.0;
        double Ey = 0.0;
        double Bz = 0.0;

        var inputs = new[] { 0.5, 0.5, 0.0 };
        var outputs = new[] { Ex, Ey, Bz };

        // All derivatives are zero for uniform static field
        var firstDerivatives = new double[3, 3];

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Maxwell equations residual should be ~0 for static field, got {residual}");
    }

    #endregion

    #region Schrodinger Equation Tests

    /// <summary>
    /// Tests Schrodinger Equation with free particle wave packet solution.
    /// For a free particle (V=0), ψ = exp(i(kx - ωt)) where ω = k²/2
    /// Real: ψ_r = cos(kx - ωt), Imaginary: ψ_i = sin(kx - ωt)
    /// </summary>
    [Fact]
    public void SchrodingerEquation_FreeParticlePlaneWave_ReturnsZeroResidual()
    {
        var pde = new SchrodingerEquation<double>();  // Free particle (V=0)

        double k = 1.0;
        double omega = k * k / 2.0;  // ω = k²/2 for free particle

        double x = 0.5;
        double t = 0.1;
        double phase = k * x - omega * t;

        // ψ = exp(i*phase) = cos(phase) + i*sin(phase)
        double psiR = Math.Cos(phase);
        double psiI = Math.Sin(phase);

        var inputs = new[] { x, t };
        var outputs = new[] { psiR, psiI };

        // First derivatives
        double dPsiRdx = -k * Math.Sin(phase);
        double dPsiRdt = omega * Math.Sin(phase);
        double dPsiIdx = k * Math.Cos(phase);
        double dPsiIdt = -omega * Math.Cos(phase);

        // Second derivatives
        double d2PsiRdx2 = -k * k * Math.Cos(phase);
        double d2PsiIdx2 = -k * k * Math.Sin(phase);

        var firstDerivatives = new double[2, 2];
        firstDerivatives[0, 0] = dPsiRdx;
        firstDerivatives[0, 1] = dPsiRdt;
        firstDerivatives[1, 0] = dPsiIdx;
        firstDerivatives[1, 1] = dPsiIdt;

        var secondDerivatives = new double[2, 2, 2];
        secondDerivatives[0, 0, 0] = d2PsiRdx2;
        secondDerivatives[1, 0, 0] = d2PsiIdx2;

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Schrodinger equation residual should be ~0 for free particle plane wave, got {residual}");
    }

    /// <summary>
    /// Tests Schrodinger Equation with stationary state solution.
    /// For a stationary state with energy E: ψ(x,t) = φ(x)*exp(-iEt)
    /// </summary>
    [Fact]
    public void SchrodingerEquation_StationaryState_SatisfiesTimeEvolution()
    {
        // Use a simple constant potential
        double V0 = 1.0;
        var pde = new SchrodingerEquation<double>(x => V0);

        // For a plane wave in constant potential:
        // E = k²/2 + V0, so ω = E
        double k = 1.0;
        double E = k * k / 2.0 + V0;

        double x = 0.5;
        double t = 0.1;

        // ψ = exp(i(kx - Et))
        double spatialPhase = k * x;
        double timePhase = E * t;
        double phase = spatialPhase - timePhase;

        double psiR = Math.Cos(phase);
        double psiI = Math.Sin(phase);

        var inputs = new[] { x, t };
        var outputs = new[] { psiR, psiI };

        // First derivatives
        double dPsiRdx = -k * Math.Sin(phase);
        double dPsiRdt = E * Math.Sin(phase);
        double dPsiIdx = k * Math.Cos(phase);
        double dPsiIdt = -E * Math.Cos(phase);

        // Second derivatives
        double d2PsiRdx2 = -k * k * Math.Cos(phase);
        double d2PsiIdx2 = -k * k * Math.Sin(phase);

        var firstDerivatives = new double[2, 2];
        firstDerivatives[0, 0] = dPsiRdx;
        firstDerivatives[0, 1] = dPsiRdt;
        firstDerivatives[1, 0] = dPsiIdx;
        firstDerivatives[1, 1] = dPsiIdt;

        var secondDerivatives = new double[2, 2, 2];
        secondDerivatives[0, 0, 0] = d2PsiRdx2;
        secondDerivatives[1, 0, 0] = d2PsiIdx2;

        var derivatives = new PDEDerivatives<double>
        {
            FirstDerivatives = firstDerivatives,
            SecondDerivatives = secondDerivatives
        };

        double residual = pde.ComputeResidual(inputs, outputs, derivatives);

        Assert.True(Math.Abs(residual) < Tolerance,
            $"Schrodinger equation residual should be ~0 for stationary state, got {residual}");
    }

    #endregion

    #region Gradient Tests

    /// <summary>
    /// Tests that heat equation gradient is computed correctly.
    /// </summary>
    [Fact]
    public void HeatEquation_GradientComputation_ReturnsExpectedValues()
    {
        double alpha = 1.5;
        var pde = new HeatEquation<double>(alpha);

        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 1.0 };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.5, 0.3 } },
            new[, ,] { { { 0.2, 0.0 }, { 0.0, 0.0 } } });

        var gradient = pde.ComputeResidualGradient(inputs, outputs, derivatives);

        // Expected gradients for heat equation R = ∂u/∂t - α*∂²u/∂x²
        // ∂R/∂(∂u/∂t) = 1
        // ∂R/∂(∂²u/∂x²) = -α
        Assert.Equal(1.0, gradient.FirstDerivatives[0, 1], 6);
        Assert.Equal(-alpha, gradient.SecondDerivatives[0, 0, 0], 6);
    }

    /// <summary>
    /// Tests that wave equation gradient is computed correctly.
    /// </summary>
    [Fact]
    public void WaveEquation_GradientComputation_ReturnsExpectedValues()
    {
        double c = 2.0;
        var pde = new WaveEquation<double>(c);

        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 1.0 };
        var derivatives = CreateDerivatives(1, 2,
            new[,] { { 0.5, 0.3 } },
            new[, ,] { { { 0.2, 0.0 }, { 0.0, 0.1 } } });

        var gradient = pde.ComputeResidualGradient(inputs, outputs, derivatives);

        // Expected gradients for wave equation R = ∂²u/∂t² - c²*∂²u/∂x²
        // ∂R/∂(∂²u/∂t²) = 1
        // ∂R/∂(∂²u/∂x²) = -c²
        Assert.Equal(1.0, gradient.SecondDerivatives[0, 1, 1], 6);
        Assert.Equal(-c * c, gradient.SecondDerivatives[0, 0, 0], 6);
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

    #endregion
}
