#nullable disable
using AiDotNet.Control;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Integration tests for the algebraic Riccati solvers and the linear quadratic regulator.
/// </summary>
/// <remarks>
/// CRITICAL: The expected values here come from three independent sources, never from the solver's
/// own output:
///   1. Closed-form solutions of the Riccati equation, derived by hand for scalar and 2x2 systems.
///   2. The residual of the Riccati equation itself — substituting the answer back in must give
///      zero, which is a check the solver cannot fake by converging to the wrong place.
///   3. Direct simulation of the closed loop, accumulating the actual cost. The cost-to-go matrix
///      predicts that number before the simulation runs, and a perturbed gain must do worse.
/// If a test fails, FIX THE SOLVER — do not relax the assertion.
/// </remarks>
public class LinearQuadraticRegulatorIntegrationTests
{
    private const double Tolerance = 1e-9;

    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    private static Matrix<double> M(double[,] values)
    {
        var matrix = new Matrix<double>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    #region Discrete-time Riccati, closed form

    /// <summary>
    /// The scalar discrete Riccati equation with a = b = q = r = 1 reads
    /// p = p − p²/(1 + p) + 1, so p² − p − 1 = 0 and p is the golden ratio (1 + √5)/2.
    /// The gain is then p/(1 + p) = 1/φ and the closed-loop pole is 1 − 1/φ = 1/φ².
    /// </summary>
    [Fact]
    public void DiscreteRiccati_ScalarSystem_GivesTheGoldenRatio()
    {
        var solution = new DiscreteAlgebraicRiccatiSolver<double>().Solve(
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }));

        double goldenRatio = (1.0 + Math.Sqrt(5.0)) / 2.0;

        Assert.True(solution.Converged);
        Assert.Equal(goldenRatio, solution.Solution[0, 0], 10);
        Assert.True(
            solution.Residual < Tolerance,
            $"Substituting the answer back into the Riccati equation left {solution.Residual}.");
    }

    /// <summary>
    /// The same system through the regulator: the gain must be 1/φ ≈ 0.618034 and the closed-loop
    /// pole 1/φ² ≈ 0.381966, which is inside the unit circle.
    /// </summary>
    [Fact]
    public void DiscreteRegulator_ScalarSystem_GivesTheReciprocalGoldenGain()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            ControlTimeDomain.Discrete);

        double goldenRatio = (1.0 + Math.Sqrt(5.0)) / 2.0;

        Assert.Equal(1.0 / goldenRatio, regulator.Gain[0, 0], 10);
        Assert.Equal(1.0 / (goldenRatio * goldenRatio), regulator.ClosedLoopMatrix[0, 0], 10);
    }

    /// <summary>
    /// A system that is already stable and costs nothing to leave alone: with q = 0 the optimal
    /// input is to do nothing, so the gain must be exactly zero.
    /// </summary>
    [Fact]
    public void DiscreteRegulator_ZeroStateCost_LeavesTheSystemAlone()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            M(new[,] { { 0.5 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 0.0 } }), M(new[,] { { 1.0 } }),
            ControlTimeDomain.Discrete);

        Assert.Equal(0.0, regulator.Gain[0, 0], 10);
    }

    /// <summary>
    /// With no dynamics at all (a = 0) the state is already at rest each step, so the cost-to-go is
    /// just the state cost and the gain is zero.
    /// </summary>
    [Fact]
    public void DiscreteRiccati_DeadBeatSystem_CostToGoIsTheStateCost()
    {
        var solution = new DiscreteAlgebraicRiccatiSolver<double>().Solve(
            M(new[,] { { 0.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 7.0 } }), M(new[,] { { 1.0 } }));

        Assert.Equal(7.0, solution.Solution[0, 0], 10);
    }

    #endregion

    #region Continuous-time Riccati, closed form

    /// <summary>
    /// The scalar continuous Riccati equation with a = −1, b = q = r = 1 reads
    /// −2p − p² + 1 = 0, so p² + 2p − 1 = 0 and p = √2 − 1. The closed-loop pole is then
    /// −1 − p = −√2.
    /// </summary>
    [Fact]
    public void ContinuousRiccati_ScalarSystem_MatchesTheClosedForm()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            M(new[,] { { -1.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            ControlTimeDomain.Continuous);

        Assert.True(regulator.RiccatiSolution.Converged);
        Assert.Equal(Math.Sqrt(2.0) - 1.0, regulator.CostToGoMatrix[0, 0], 10);
        Assert.Equal(Math.Sqrt(2.0) - 1.0, regulator.Gain[0, 0], 10);
        Assert.Equal(-Math.Sqrt(2.0), regulator.ClosedLoopMatrix[0, 0], 10);
        Assert.True(regulator.RiccatiSolution.Residual < Tolerance);
    }

    /// <summary>
    /// An integrator, a = 0: the equation reduces to 1 − p² = 0, so p = 1 and the closed-loop pole
    /// is −1.
    /// </summary>
    [Fact]
    public void ContinuousRiccati_Integrator_GivesUnitSolution()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            M(new[,] { { 0.0 } }), M(new[,] { { 1.0 } }),
            M(new[,] { { 1.0 } }), M(new[,] { { 1.0 } }),
            ControlTimeDomain.Continuous);

        Assert.Equal(1.0, regulator.CostToGoMatrix[0, 0], 10);
        Assert.Equal(-1.0, regulator.ClosedLoopMatrix[0, 0], 10);
    }

    /// <summary>
    /// The continuous double integrator — position and velocity, force input — with Q = I and R = 1.
    /// Writing P = [[p11, p12], [p12, p22]] and expanding AᵀP + PA − PBBᵀP + Q = 0 gives
    /// 1 − p12² = 0, p11 − p12·p22 = 0, and 2·p12 − p22² + 1 = 0, so p12 = 1, p22 = √3, p11 = √3.
    /// The gain is then BᵀP = [1, √3].
    /// </summary>
    [Fact]
    public void ContinuousRiccati_DoubleIntegrator_MatchesTheHandDerivedSolution()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            M(new[,] { { 0.0, 1.0 }, { 0.0, 0.0 } }),
            M(new[,] { { 0.0 }, { 1.0 } }),
            Matrix<double>.CreateIdentity(2),
            Matrix<double>.CreateIdentity(1),
            ControlTimeDomain.Continuous);

        double root3 = Math.Sqrt(3.0);

        Assert.Equal(root3, regulator.CostToGoMatrix[0, 0], 8);
        Assert.Equal(1.0, regulator.CostToGoMatrix[0, 1], 8);
        Assert.Equal(1.0, regulator.CostToGoMatrix[1, 0], 8);
        Assert.Equal(root3, regulator.CostToGoMatrix[1, 1], 8);

        Assert.Equal(1.0, regulator.Gain[0, 0], 8);
        Assert.Equal(root3, regulator.Gain[0, 1], 8);

        Assert.True(regulator.RiccatiSolution.Residual < 1e-8);
    }

    #endregion

    #region Verification by simulation

    /// <summary>
    /// The cost-to-go matrix claims that the total cost of running the closed loop forever from x₀
    /// is exactly x₀ᵀPx₀. This runs the loop and adds the cost up. The two numbers must agree — a
    /// check that involves none of the Riccati machinery and would catch a P that satisfies the
    /// equation but means something other than what it is documented to mean.
    /// </summary>
    [Fact]
    public void DiscreteRegulator_SimulatedCost_MatchesTheCostToGoPrediction()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r, ControlTimeDomain.Discrete);

        var start = V(3.0, -1.0);
        double predicted = regulator.CostToGo(start);
        double simulated = SimulateCost(a, b, q, r, regulator.Gain, start, steps: 2000);

        Assert.Equal(predicted, simulated, 6);
    }

    /// <summary>
    /// The defining property: no other gain is cheaper. This perturbs the optimal gain in both
    /// directions along each entry and confirms every perturbation costs more over the same horizon.
    /// </summary>
    [Theory]
    [InlineData(0, 0.05)]
    [InlineData(0, -0.05)]
    [InlineData(1, 0.05)]
    [InlineData(1, -0.05)]
    public void DiscreteRegulator_PerturbedGain_AlwaysCostsMore(int entry, double delta)
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r, ControlTimeDomain.Discrete);
        var start = V(3.0, -1.0);

        double optimalCost = SimulateCost(a, b, q, r, regulator.Gain, start, steps: 2000);

        var perturbed = new Matrix<double>(1, 2);
        perturbed[0, 0] = regulator.Gain[0, 0];
        perturbed[0, 1] = regulator.Gain[0, 1];
        perturbed[0, entry] += delta;

        double perturbedCost = SimulateCost(a, b, q, r, perturbed, start, steps: 2000);

        Assert.True(
            perturbedCost > optimalCost,
            $"Perturbing gain entry {entry} by {delta} cost {perturbedCost}, which is not more " +
            $"than the optimal {optimalCost}. The LQR gain is supposed to be optimal.");
    }

    /// <summary>
    /// The closed loop must be stable: driven from a non-zero state with no disturbance, it must
    /// return to the origin. This is the property that makes the controller usable at all, and it
    /// holds for an unstable open-loop system — here A has a repeated eigenvalue at 1.
    /// </summary>
    [Fact]
    public void DiscreteRegulator_ClosedLoop_DrivesAnUnstableSystemToRest()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });

        var regulator = new LinearQuadraticRegulator<double>(
            a, b, Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            ControlTimeDomain.Discrete);

        var state = V(10.0, -5.0);
        for (int step = 0; step < 500; step++)
        {
            var input = regulator.ComputeControl(state);
            state = Advance(a, b, state, input);
        }

        Assert.True(
            Math.Abs(state[0]) < 1e-8 && Math.Abs(state[1]) < 1e-8,
            $"The closed loop did not settle: ({state[0]}, {state[1]}).");
    }

    /// <summary>
    /// The continuous closed loop must decay too. Integrating ẋ = (A − BK)x with small explicit
    /// steps is crude, but a stable matrix is exactly one under which this decays, so it is enough
    /// to distinguish a correct gain from a wrong one.
    /// </summary>
    [Fact]
    public void ContinuousRegulator_ClosedLoop_Decays()
    {
        var a = M(new[,] { { 0.0, 1.0 }, { 0.0, 0.0 } });
        var b = M(new[,] { { 0.0 }, { 1.0 } });

        var regulator = new LinearQuadraticRegulator<double>(
            a, b, Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            ControlTimeDomain.Continuous);

        var closedLoop = regulator.ClosedLoopMatrix;
        var state = V(1.0, 0.0);

        const double StepSize = 0.001;
        for (int step = 0; step < 20000; step++)
        {
            double dx0 = closedLoop[0, 0] * state[0] + closedLoop[0, 1] * state[1];
            double dx1 = closedLoop[1, 0] * state[0] + closedLoop[1, 1] * state[1];
            state = V(state[0] + StepSize * dx0, state[1] + StepSize * dx1);
        }

        Assert.True(
            Math.Abs(state[0]) < 1e-6 && Math.Abs(state[1]) < 1e-6,
            $"The continuous closed loop did not decay: ({state[0]}, {state[1]}).");
    }

    #endregion

    #region Structural properties

    /// <summary>
    /// Only the ratio of Q to R matters. Scaling both by the same factor must leave the gain exactly
    /// where it was, while scaling the cost-to-go matrix by that factor — the cost changes, the
    /// optimal policy does not.
    /// </summary>
    [Fact]
    public void Regulator_ScalingBothCosts_LeavesTheGainUnchanged()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });

        var baseline = new LinearQuadraticRegulator<double>(
            a, b, Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        var scaled = new LinearQuadraticRegulator<double>(
            a, b,
            M(new[,] { { 100.0, 0.0 }, { 0.0, 100.0 } }),
            M(new[,] { { 100.0 } }));

        Assert.Equal(baseline.Gain[0, 0], scaled.Gain[0, 0], 8);
        Assert.Equal(baseline.Gain[0, 1], scaled.Gain[0, 1], 8);

        Assert.Equal(
            baseline.CostToGoMatrix[0, 0] * 100.0, scaled.CostToGoMatrix[0, 0], 6);
    }

    /// <summary>
    /// Expensive control means gentle control: raising R relative to Q must shrink the gain.
    /// </summary>
    [Fact]
    public void Regulator_ExpensiveControl_ProducesASmallerGain()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });
        var q = Matrix<double>.CreateIdentity(2);

        var cheap = new LinearQuadraticRegulator<double>(
            a, b, q, M(new[,] { { 0.01 } }));
        var expensive = new LinearQuadraticRegulator<double>(
            a, b, q, M(new[,] { { 100.0 } }));

        Assert.True(
            Math.Abs(expensive.Gain[0, 0]) < Math.Abs(cheap.Gain[0, 0]),
            "Making control expensive should produce a gentler controller, not a more aggressive " +
            "one.");
    }

    /// <summary>
    /// The cost-to-go matrix is symmetric as a matter of theory. Rounding can only break that if the
    /// implementation lets it, so this checks it exactly rather than approximately.
    /// </summary>
    [Fact]
    public void Regulator_CostToGoMatrix_IsExactlySymmetric()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            M(new[,] { { 1.0, 1.0, 0.0 }, { 0.0, 1.0, 1.0 }, { 0.0, 0.0, 0.9 } }),
            M(new[,] { { 0.0 }, { 0.0 }, { 1.0 } }),
            Matrix<double>.CreateIdentity(3),
            Matrix<double>.CreateIdentity(1));

        var p = regulator.CostToGoMatrix;
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(p[r, c], p[c, r]);
            }
        }
    }

    /// <summary>
    /// A larger system, verified the only way that scales: by residual. Three states, two inputs,
    /// non-diagonal costs.
    /// </summary>
    [Fact]
    public void DiscreteRiccati_MultiInputSystem_SatisfiesTheEquation()
    {
        var solution = new DiscreteAlgebraicRiccatiSolver<double>().Solve(
            M(new[,] { { 0.9, 0.3, 0.0 }, { 0.0, 0.8, 0.4 }, { 0.1, 0.0, 1.1 } }),
            M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 0.5, 0.5 } }),
            M(new[,] { { 2.0, 0.5, 0.0 }, { 0.5, 1.0, 0.0 }, { 0.0, 0.0, 3.0 } }),
            M(new[,] { { 1.0, 0.2 }, { 0.2, 2.0 } }));

        Assert.True(solution.Converged);
        Assert.True(
            solution.Residual < 1e-9,
            $"The Riccati residual was {solution.Residual}, which is too large to call solved.");
    }

    /// <summary>
    /// The same for the continuous solver.
    /// </summary>
    [Fact]
    public void ContinuousRiccati_MultiInputSystem_SatisfiesTheEquation()
    {
        var solution = new ContinuousAlgebraicRiccatiSolver<double>().Solve(
            M(new[,] { { -0.5, 1.0, 0.0 }, { 0.0, -0.2, 1.0 }, { 0.3, 0.0, 0.4 } }),
            M(new[,] { { 1.0, 0.0 }, { 0.0, 1.0 }, { 0.5, 0.5 } }),
            M(new[,] { { 2.0, 0.5, 0.0 }, { 0.5, 1.0, 0.0 }, { 0.0, 0.0, 3.0 } }),
            M(new[,] { { 1.0, 0.2 }, { 0.2, 2.0 } }));

        Assert.True(solution.Converged);
        Assert.True(
            solution.Residual < 1e-8,
            $"The Riccati residual was {solution.Residual}, which is too large to call solved.");
    }

    /// <summary>
    /// Both solvers converge quadratically, so a well-posed problem should need very few iterations.
    /// This guards the doubling and sign-function structure: a linearly convergent implementation
    /// would still reach the right answer, but would take far more steps to do it.
    /// </summary>
    [Fact]
    public void Riccati_WellPosedProblems_ConvergeInFewIterations()
    {
        var discrete = new DiscreteAlgebraicRiccatiSolver<double>().Solve(
            M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } }), M(new[,] { { 0.5 }, { 1.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        var continuous = new ContinuousAlgebraicRiccatiSolver<double>().Solve(
            M(new[,] { { 0.0, 1.0 }, { 0.0, 0.0 } }), M(new[,] { { 0.0 }, { 1.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        Assert.InRange(discrete.Iterations, 1, 30);
        Assert.InRange(continuous.Iterations, 1, 30);
    }

    [Fact(Timeout = 120000)]
    public async Task ContinuousRiccati_ScalingChangesOnlyThePath()
    {
        await Task.Yield();

        var state = M(new[,] { { -1000.0, 1.0 }, { 0.0, -0.001 } });
        var input = M(new[,] { { 1.0 }, { 1.0 } });
        var stateCost = Matrix<double>.CreateIdentity(2);
        var inputCost = Matrix<double>.CreateIdentity(1);

        var scaled = new ContinuousAlgebraicRiccatiSolver<double>(
            new AlgebraicRiccatiSolverOptions { UseSignFunctionScaling = true })
            .Solve(state, input, stateCost, inputCost);
        var unscaled = new ContinuousAlgebraicRiccatiSolver<double>(
            new AlgebraicRiccatiSolverOptions { UseSignFunctionScaling = false })
            .Solve(state, input, stateCost, inputCost);

        Assert.True(scaled.Converged);
        Assert.True(unscaled.Converged);
        Assert.True(scaled.Iterations <= unscaled.Iterations);
        for (int row = 0; row < 2; row++)
        {
            for (int column = 0; column < 2; column++)
            {
                Assert.Equal(unscaled.Solution[row, column], scaled.Solution[row, column], 8);
            }
        }
    }

    #endregion

    #region Validation

    [Fact(Timeout = 120000)]
    public async Task ContinuousRiccati_RankDeficientInvariantSubspace_ThrowsNamedDiagnostic()
    {
        await Task.Yield();

        var rankDeficientSign = M(new[,] { { 0.0, 0.0 }, { 0.0, -1.0 } });

        var exception = Assert.Throws<InvalidOperationException>(
            () => ContinuousAlgebraicRiccatiSolver<double>.ExtractSolution(
                rankDeficientSign, n: 1));

        Assert.Contains("rank-deficient", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task Riccati_AsymmetricCost_ThrowsNamedArgument()
    {
        await Task.Yield();

        var exception = Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0 }, { 1.0 } }),
                M(new[,] { { 1.0, 0.25 }, { 0.0, 1.0 } }),
                Matrix<double>.CreateIdentity(1)));

        Assert.Equal("stateCost", exception.ParamName);
        Assert.Contains("symmetric", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task Riccati_NegativeInputCost_ThrowsNamedArgument()
    {
        await Task.Yield();

        var exception = Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                M(new[,] { { 0.2 } }), M(new[,] { { 1.0 } }),
                M(new[,] { { 0.1 } }), M(new[,] { { -1.0 } })));

        Assert.Equal("inputCost", exception.ParamName);
        Assert.Contains("positive-definite", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task Riccati_IndefiniteStateCost_ThrowsNamedArgument()
    {
        await Task.Yield();

        var exception = Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0 }, { 0.0 } }),
                M(new[,] { { 1.0, 0.0 }, { 0.0, -1.0 } }),
                Matrix<double>.CreateIdentity(1)));

        Assert.Equal("stateCost", exception.ParamName);
        Assert.Contains("positive-semidefinite", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 120000)]
    public async Task RiccatiOptions_CopyPreservesAllConfiguration()
    {
        await Task.Yield();

        var original = new AlgebraicRiccatiSolverOptions
        {
            Seed = 42,
            MaxIterations = 17,
            Tolerance = 2e-9,
            UseSignFunctionScaling = false,
        };

        var copy = new AlgebraicRiccatiSolverOptions(original);

        Assert.Equal(original.Seed, copy.Seed);
        Assert.Equal(original.MaxIterations, copy.MaxIterations);
        Assert.Equal(original.Tolerance, copy.Tolerance);
        Assert.Equal(original.UseSignFunctionScaling, copy.UseSignFunctionScaling);
    }

    [Fact]
    public void Riccati_NonSquareStateMatrix_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                M(new[,] { { 1.0, 0.0 } }), M(new[,] { { 1.0 } }),
                Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1)));
    }

    [Fact]
    public void Riccati_InputMatrixWithWrongRowCount_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0 } }),
                Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1)));
    }

    [Fact]
    public void Riccati_StateCostWithWrongSize_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                Matrix<double>.CreateIdentity(2), M(new[,] { { 1.0 }, { 1.0 } }),
                Matrix<double>.CreateIdentity(3), Matrix<double>.CreateIdentity(1)));
    }

    [Fact(Timeout = 120000)]
    public async Task Riccati_SingularInputCost_ThrowsNamedArgument()
    {
        await Task.Yield();

        // R must be positive definite: a direction of zero control cost would let the controller
        // apply unbounded effort for free, so there is no finite optimum to find.
        var exception = Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>().Solve(
                Matrix<double>.CreateIdentity(1), M(new[,] { { 1.0 } }),
                Matrix<double>.CreateIdentity(1), M(new[,] { { 0.0 } })));

        Assert.Equal("inputCost", exception.ParamName);
        Assert.Contains("positive-definite", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Regulator_StateOfWrongLength_Throws()
    {
        var regulator = new LinearQuadraticRegulator<double>(
            Matrix<double>.CreateIdentity(2), M(new[,] { { 0.0 }, { 1.0 } }),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        Assert.Throws<ArgumentException>(() => regulator.ComputeControl(V(1.0)));
    }

    [Fact]
    public void Riccati_NonPositiveIterationLimit_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new DiscreteAlgebraicRiccatiSolver<double>(
                new AlgebraicRiccatiSolverOptions { MaxIterations = 0 }));
    }

    #endregion

    #region Simulation helpers

    private static Vector<double> Advance(
        Matrix<double> a, Matrix<double> b, Vector<double> state, Vector<double> input)
    {
        var next = new Vector<double>(state.Length);
        for (int r = 0; r < state.Length; r++)
        {
            double value = 0.0;
            for (int c = 0; c < state.Length; c++) value += a[r, c] * state[c];
            for (int c = 0; c < input.Length; c++) value += b[r, c] * input[c];
            next[r] = value;
        }

        return next;
    }

    /// <summary>
    /// Runs the closed loop under a given gain and accumulates the quadratic cost.
    /// </summary>
    private static double SimulateCost(
        Matrix<double> a,
        Matrix<double> b,
        Matrix<double> q,
        Matrix<double> r,
        Matrix<double> gain,
        Vector<double> start,
        int steps)
    {
        var state = start;
        double total = 0.0;

        for (int step = 0; step < steps; step++)
        {
            var input = new Vector<double>(gain.Rows);
            for (int i = 0; i < gain.Rows; i++)
            {
                double value = 0.0;
                for (int c = 0; c < state.Length; c++) value += gain[i, c] * state[c];
                input[i] = -value;
            }

            for (int i = 0; i < state.Length; i++)
            {
                for (int j = 0; j < state.Length; j++) total += state[i] * q[i, j] * state[j];
            }

            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < input.Length; j++) total += input[i] * r[i, j] * input[j];
            }

            state = Advance(a, b, state, input);
        }

        return total;
    }

    #endregion
}
