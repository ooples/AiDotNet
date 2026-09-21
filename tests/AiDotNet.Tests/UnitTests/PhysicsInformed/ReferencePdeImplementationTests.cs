using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.PhysicsInformed.PDEs;
using AiDotNet.PhysicsInformed.PINNs;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed;

/// <summary>
/// Covers the reference implementations that make the PINN families constructible.
/// </summary>
/// <remarks>
/// <para>
/// <c>MultiScalePINN</c> and <c>InverseProblemPINN</c> take an <c>IMultiScalePDE</c> and an
/// <c>IInverseProblem</c>, and five PINN types take <c>IBoundaryCondition[]</c>. The library implemented
/// none of those interfaces, so all three could be documented and none could be constructed
/// (#2105, #2099). These tests build each PINN for real rather than only checking the new types in
/// isolation, because "the example compiles" was already true when nothing worked.
/// </para>
/// </remarks>
public class ReferencePdeImplementationTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 1);

    private static PDEDerivatives<double> Derivatives(double dudt, double d2udx2)
    {
        var d = new PDEDerivatives<double>
        {
            FirstDerivatives = new double[1, 2],
            SecondDerivatives = new double[1, 2, 2]
        };
        d.FirstDerivatives[0, 1] = dudt;
        d.SecondDerivatives[0, 0, 0] = d2udx2;
        return d;
    }

    // ── The multi-scale equation ──────────────────────────────────────────────────────────────────

    [Fact]
    public void TheMultiScalePinnCanBeConstructed()
    {
        var pinn = new MultiScalePINN<double>(
            Architecture(),
            new MultiScaleReactionDiffusion<double>(),
            [new DirichletBoundaryCondition<double>()]);

        Assert.NotNull(pinn);
    }

    /// <summary>
    /// The two scales must actually differ — a "multi-scale" equation whose scales computed the same
    /// residual would be a single-scale one with extra steps.
    /// </summary>
    [Fact]
    public void TheTwoScalesUseDifferentDiffusivities()
    {
        var pde = new MultiScaleReactionDiffusion<double>(coarseDiffusivity: 1.0, fineDiffusivity: 0.01);
        var inputs = new[] { 0.5, 0.1 };
        var outputs = new[] { 1.0 };
        var derivatives = Derivatives(dudt: 2.0, d2udx2: 3.0);

        double coarse = pde.ComputeScaleResidual(0, inputs, outputs, derivatives);
        double fine = pde.ComputeScaleResidual(1, inputs, outputs, derivatives);

        // ∂u/∂t − k ∂²u/∂x²  =  2 − k·3
        Assert.Equal(2.0 - 1.0 * 3.0, coarse, 10);
        Assert.Equal(2.0 - 0.01 * 3.0, fine, 10);
        Assert.NotEqual(coarse, fine);
    }

    [Fact]
    public void TheCombinedResidualIsBothScalesTogether()
    {
        var pde = new MultiScaleReactionDiffusion<double>();
        var inputs = new Vector<double>(new[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new[] { 1.0 });
        var derivatives = Derivatives(dudt: 2.0, d2udx2: 3.0);

        double combined = pde.ComputeResidual(inputs, outputs, derivatives);
        double expected =
            pde.ComputeScaleResidual(0, inputs.ToArray(), outputs.ToArray(), derivatives)
            + pde.ComputeScaleResidual(1, inputs.ToArray(), outputs.ToArray(), derivatives);

        Assert.Equal(expected, combined, 10);
    }

    /// <summary>
    /// The fine scale is weighted up on purpose: its residual is numerically smaller, so left equal it
    /// would be optimised away.
    /// </summary>
    [Fact]
    public void TheFineScaleIsWeightedAboveTheCoarseOne()
    {
        var pde = new MultiScaleReactionDiffusion<double>();

        Assert.True(pde.GetScaleLossWeight(1) > pde.GetScaleLossWeight(0));
        Assert.Equal(2, pde.NumberOfScales);
        Assert.Equal(2, pde.ScaleCharacteristicLengths.Length);
        Assert.True(pde.ScaleCharacteristicLengths[1] < pde.ScaleCharacteristicLengths[0]);
    }

    [Fact]
    public void TheCouplingIsZeroWhenTheScalesAgree()
    {
        var pde = new MultiScaleReactionDiffusion<double>();
        var derivatives = Derivatives(0.0, 0.0);

        double agreeing = pde.ComputeScaleCoupling(
            0, 1, [0.5, 0.1], [1.0], [1.0], derivatives, derivatives);
        double disagreeing = pde.ComputeScaleCoupling(
            0, 1, [0.5, 0.1], [1.0], [0.4], derivatives, derivatives);

        Assert.Equal(0.0, agreeing, 10);
        Assert.True(disagreeing > 0);
    }

    // ── The inverse problem ───────────────────────────────────────────────────────────────────────

    private static HeatConductivityInverseProblem<double> Problem() =>
        new(
            [
                (location: new[] { 0.25, 0.1 }, value: new[] { 0.82 }),
                (location: new[] { 0.50, 0.1 }, value: new[] { 0.95 }),
                (location: new[] { 0.75, 0.1 }, value: new[] { 0.80 })
            ],
            initialGuess: 0.5);

    [Fact]
    public void TheInverseProblemPinnCanBeConstructed()
    {
        var pinn = new InverseProblemPINN<double>(
            Architecture(), Problem(), [new NeumannBoundaryCondition<double>()]);

        Assert.NotNull(pinn);
    }

    [Fact]
    public void ItCarriesTheObservationsItWasGiven()
    {
        var problem = Problem();

        Assert.Equal(3, problem.Observations.Count);
        Assert.Equal(1, problem.NumberOfParameters);
        Assert.Equal("thermal_conductivity", problem.ParameterNames.Single());
        Assert.Equal(0.5, problem.InitialParameterGuesses.Single(), 10);
    }

    /// <summary>
    /// The parameterised PDE is the whole point: the search substitutes a guess and asks what it
    /// predicts, so the guess has to reach the equation.
    /// </summary>
    [Fact]
    public void TheParameterisedEquationUsesTheGuess()
    {
        var problem = Problem();
        var inputs = new Vector<double>(new[] { 0.5, 0.1 });
        var outputs = new Vector<double>(new[] { 1.0 });
        var derivatives = Derivatives(dudt: 2.0, d2udx2: 3.0);

        double atLowConductivity = problem.CreateParameterizedPDE([0.5])
            .ComputeResidual(inputs, outputs, derivatives);
        double atHighConductivity = problem.CreateParameterizedPDE([2.0])
            .ComputeResidual(inputs, outputs, derivatives);

        Assert.Equal(2.0 - 0.5 * 3.0, atLowConductivity, 10);
        Assert.Equal(2.0 - 2.0 * 3.0, atHighConductivity, 10);
    }

    [Fact]
    public void ParametersOutsideTheBoundsAreRejected()
    {
        var problem = Problem();

        Assert.True(problem.ValidateParameters([1.0]));
        Assert.False(problem.ValidateParameters([0.0]));
        Assert.False(problem.ValidateParameters([1000.0]));
        Assert.False(problem.ValidateParameters([]));

        Assert.Throws<ArgumentException>(() => problem.CreateParameterizedPDE([-1.0]));
    }

    [Fact]
    public void AProblemWithNoObservations_IsRejected()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            new HeatConductivityInverseProblem<double>([]));

        Assert.Contains("at least one observation", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStartingGuessOutsideTheBounds_IsRejected()
    {
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HeatConductivityInverseProblem<double>(
                [(location: new[] { 0.5, 0.1 }, value: new[] { 0.8 })],
                initialGuess: 500.0,
                lowerBound: 0.001,
                upperBound: 100.0));

        Assert.Contains("outside the bounds", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// An invented noise level is worse than none: it tells the fit how much to trust the data.
    /// </summary>
    [Fact]
    public void NoiseLevelIsAbsentUnlessGiven()
    {
        Assert.False(Problem().HasMeasurementNoiseLevel);

        var withNoise = new HeatConductivityInverseProblem<double>(
            [(location: new[] { 0.5, 0.1 }, value: new[] { 0.8 })],
            measurementNoiseLevel: 0.02);

        Assert.True(withNoise.HasMeasurementNoiseLevel);
        Assert.Equal(0.02, withNoise.MeasurementNoiseLevel, 10);
    }

    // ── Boundary conditions ───────────────────────────────────────────────────────────────────────

    [Fact]
    public void DirichletSelectsTheEdgesAndMeasuresTheGapFromItsValue()
    {
        var bc = new DirichletBoundaryCondition<double>(boundaryValue: 0.0, lowerBound: 0.0, upperBound: 1.0);

        Assert.True(bc.IsOnBoundary(new Vector<double>(new[] { 0.0, 0.5 })));
        Assert.True(bc.IsOnBoundary(new Vector<double>(new[] { 1.0, 0.5 })));
        Assert.False(bc.IsOnBoundary(new Vector<double>(new[] { 0.5, 0.5 })));

        double residual = bc.ComputeBoundaryResidual(
            new Vector<double>(new[] { 0.0, 0.5 }),
            new Vector<double>(new[] { 0.3 }),
            Derivatives(0.0, 0.0));

        // Held at 0, solution says 0.3 — the violation is 0.3.
        Assert.Equal(0.3, residual, 10);
    }

    [Fact]
    public void NeumannMeasuresTheGapFromItsFlux()
    {
        var insulated = new NeumannBoundaryCondition<double>(flux: 0.0, lowerBound: 0.0, upperBound: 1.0);

        var derivatives = new PDEDerivatives<double> { FirstDerivatives = new double[1, 2] };
        derivatives.FirstDerivatives[0, 0] = 0.25;

        double residual = insulated.ComputeBoundaryResidual(
            new Vector<double>(new[] { 0.0, 0.5 }),
            new Vector<double>(new[] { 1.0 }),
            derivatives);

        // Insulated means zero slope; the solution slopes at 0.25, so that is the violation.
        Assert.Equal(0.25, residual, 10);
    }

    /// <summary>
    /// Collocation points are sampled, so they land near a boundary rather than exactly on it. A
    /// tolerance of zero would select none of them and silently apply no boundary condition at all.
    /// </summary>
    [Fact]
    public void APointJustInsideTheToleranceCountsAsOnTheBoundary()
    {
        var bc = new DirichletBoundaryCondition<double>(lowerBound: 0.0, upperBound: 1.0, tolerance: 1e-3);

        Assert.True(bc.IsOnBoundary(new Vector<double>(new[] { 0.0005, 0.5 })));
        Assert.False(bc.IsOnBoundary(new Vector<double>(new[] { 0.05, 0.5 })));
    }

    [Fact]
    public void ANonPositiveToleranceIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new DirichletBoundaryCondition<double>(tolerance: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new NeumannBoundaryCondition<double>(tolerance: -1.0));
    }

    /// <summary>
    /// The ordinary PINN takes boundary conditions too, and was equally unconstructible without one.
    /// </summary>
    [Fact]
    public void TheOrdinaryPinnCanBeConstructedWithAShippedBoundaryCondition()
    {
        var pinn = new PhysicsInformedNeuralNetwork<double>(
            Architecture(),
            new HeatEquation<double>(),
            [new DirichletBoundaryCondition<double>(), new NeumannBoundaryCondition<double>(coordinate: 1)]);

        Assert.NotNull(pinn);
    }
}
