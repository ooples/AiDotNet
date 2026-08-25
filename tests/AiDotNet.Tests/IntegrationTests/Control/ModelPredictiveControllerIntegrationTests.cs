#nullable disable
using AiDotNet.Control;
using AiDotNet.Models.Options;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Integration tests for the linear model predictive controller.
/// </summary>
/// <remarks>
/// CRITICAL: The central check here does not involve any expected number at all. With the terminal
/// cost set to the infinite-horizon Riccati solution and nothing constrained, MPC and LQR solve the
/// same problem by completely different means — one a quadratic program over a finite horizon, the
/// other a fixed gain from an algebraic equation — so they must produce identical inputs. That
/// equality is a theorem, not a tolerance, and it exercises the condensed dynamics, the stacked cost
/// and the terminal cost all at once.
///
/// The constrained tests then check the property LQR cannot have: that the limits are respected.
/// If a test fails, FIX THE CONTROLLER — do not relax the assertion.
/// </remarks>
public class ModelPredictiveControllerIntegrationTests
{
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

    private static Matrix<decimal> M(decimal[,] values)
    {
        var matrix = new Matrix<decimal>(values.GetLength(0), values.GetLength(1));
        for (int r = 0; r < values.GetLength(0); r++)
        {
            for (int c = 0; c < values.GetLength(1); c++) matrix[r, c] = values[r, c];
        }

        return matrix;
    }

    private static Matrix<double> DoubleIntegrator() =>
        M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });

    private static Matrix<double> DoubleIntegratorInput() =>
        M(new[,] { { 0.5 }, { 1.0 } });

    #region Equivalence with LQR

    /// <summary>
    /// The defining consistency check: unconstrained MPC with a Riccati terminal cost must reproduce
    /// the LQR input, from any state.
    ///
    /// The equality is exact as mathematics; the agreement asserted here is to five decimals because
    /// one side is an iterative interior-point solve. An unbounded input is rewritten as a difference
    /// of two non-negative variables, which leaves the quadratic program with a direction along which
    /// the objective is exactly flat — both parts can grow together without changing anything. The
    /// optimum is therefore a face rather than a point, and while the difference that matters is
    /// pinned down, it is recovered to about the square root of the solver's tolerance rather than to
    /// the tolerance itself. Five decimals on a control input is many orders of magnitude below
    /// anything an actuator could act on.
    /// </summary>
    [Theory]
    [InlineData(3.0, -1.0)]
    [InlineData(-7.0, 2.5)]
    [InlineData(0.25, 0.0)]
    [InlineData(-0.5, -0.5)]
    public void Mpc_Unconstrained_ReproducesTheLqrInput(double position, double velocity)
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r);
        var controller = new ModelPredictiveController<double>(
            a, b, q, r, new ModelPredictiveControllerOptions<double> { Horizon = 12 });

        var state = V(position, velocity);

        var expected = regulator.ComputeControl(state);
        var actual = controller.ComputeControl(state);

        Assert.Equal(expected[0], actual[0], 5);
    }

    /// <summary>
    /// The active-set solver has no such degenerate direction — it solves an equality-constrained
    /// system on the working set directly rather than approaching an optimal face from inside — so
    /// on the same unconstrained problem it must reproduce LQR far more tightly. This documents the
    /// real difference between the two solvers rather than leaving the looser tolerance above looking
    /// like the best either can do.
    /// </summary>
    [Fact]
    public void Mpc_UnconstrainedWithActiveSet_ReproducesLqrToHighPrecision()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r);
        var controller = new ModelPredictiveController<double>(
            a, b, q, r,
            new ModelPredictiveControllerOptions<double> { Horizon = 12 },
            new ActiveSetQuadraticProgramSolver<double>());

        var state = V(3.0, -1.0);

        Assert.Equal(
            regulator.ComputeControl(state)[0], controller.ComputeControl(state)[0], 9);
    }

    /// <summary>
    /// The terminal cost is what makes that equivalence hold, so it must hold even at a horizon of
    /// one: with the Riccati solution pricing the state left at the end, a single step of lookahead
    /// already accounts for the entire infinite future.
    /// </summary>
    [Fact]
    public void Mpc_HorizonOfOne_StillReproducesLqr()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r);
        var controller = new ModelPredictiveController<double>(
            a, b, q, r, new ModelPredictiveControllerOptions<double> { Horizon = 1 });

        var state = V(4.0, -2.0);

        Assert.Equal(
            regulator.ComputeControl(state)[0], controller.ComputeControl(state)[0], 5);
    }

    /// <summary>
    /// Conversely, a zero terminal cost is a different problem and must give a different answer at a
    /// short horizon — this confirms the terminal cost is actually being applied rather than the
    /// equivalence above holding for some unrelated reason.
    /// </summary>
    [Fact]
    public void Mpc_ZeroTerminalCostAtShortHorizon_DiffersFromLqr()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r);
        var controller = new ModelPredictiveController<double>(
            a, b, q, r,
            new ModelPredictiveControllerOptions<double>
            {
                Horizon = 2,
                TerminalCost = new Matrix<double>(2, 2),
            });

        var state = V(4.0, -2.0);

        Assert.True(
            Math.Abs(regulator.ComputeControl(state)[0] - controller.ComputeControl(state)[0])
                > 1e-3,
            "With no terminal cost and a two-step horizon the controller is solving a genuinely " +
            "different problem, so matching LQR would mean the terminal cost is being ignored.");
    }

    /// <summary>
    /// The two quadratic program solvers must agree, since they solve the same program by unrelated
    /// methods.
    /// </summary>
    [Fact]
    public void Mpc_ActiveSetAndInteriorPoint_ProduceTheSameInput()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var options = new ModelPredictiveControllerOptions<double>
        {
            Horizon = 6,
            InputLowerBounds = V(-0.4),
            InputUpperBounds = V(0.4),
        };

        var interior = new ModelPredictiveController<double>(a, b, q, r, options);
        var activeSet = new ModelPredictiveController<double>(
            a, b, q, r, options, new ActiveSetQuadraticProgramSolver<double>());

        var state = V(5.0, 1.0);

        Assert.Equal(
            interior.ComputeControl(state)[0], activeSet.ComputeControl(state)[0], 5);
    }

    #endregion

    #region Constraints

    /// <summary>
    /// An input limit must be respected, and it must actually bind: from a state far from the origin
    /// the unconstrained controller would command far more than the actuator can deliver.
    /// </summary>
    [Fact]
    public void Mpc_InputLimit_IsRespectedAndBinds()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        const double Limit = 0.5;

        var unconstrained = new LinearQuadraticRegulator<double>(a, b, q, r);
        var controller = new ModelPredictiveController<double>(
            a, b, q, r,
            new ModelPredictiveControllerOptions<double>
            {
                Horizon = 15,
                InputLowerBounds = V(-Limit),
                InputUpperBounds = V(Limit),
            });

        var state = V(20.0, 0.0);

        Assert.True(
            Math.Abs(unconstrained.ComputeControl(state)[0]) > Limit,
            "This test is only meaningful if the unconstrained controller would violate the limit.");

        var input = controller.ComputeControl(state);

        Assert.True(
            input[0] >= -Limit - 1e-6 && input[0] <= Limit + 1e-6,
            $"The controller commanded {input[0]}, outside its own limit of ±{Limit}.");
    }

    /// <summary>
    /// Every step of the plan must respect the limit, not merely the first — the constraint is what
    /// the whole plan is built around, and a plan that violates it later is not a feasible plan.
    /// </summary>
    [Fact]
    public void Mpc_WholePlan_RespectsTheInputLimit()
    {
        const double Limit = 0.3;

        var controller = new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            new ModelPredictiveControllerOptions<double>
            {
                Horizon = 10,
                InputLowerBounds = V(-Limit),
                InputUpperBounds = V(Limit),
            });

        var plan = controller.ComputePlan(V(15.0, 2.0));

        Assert.Equal(10, plan.Rows);

        for (int k = 0; k < plan.Rows; k++)
        {
            Assert.True(
                plan[k, 0] >= -Limit - 1e-6 && plan[k, 0] <= Limit + 1e-6,
                $"Step {k} of the plan commands {plan[k, 0]}, outside the limit of ±{Limit}.");
        }
    }

    /// <summary>
    /// With a saturating actuator the controller must still bring the system to rest, taking longer
    /// than it would unconstrained but getting there. This is the whole point of MPC over a clipped
    /// regulator.
    /// </summary>
    [Fact]
    public void Mpc_WithSaturatingActuator_StillReachesTheOrigin()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();

        var controller = new ModelPredictiveController<double>(
            a, b, Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            new ModelPredictiveControllerOptions<double>
            {
                Horizon = 25,
                InputLowerBounds = V(-1.0),
                InputUpperBounds = V(1.0),
            });

        var state = V(10.0, 0.0);

        for (int step = 0; step < 300; step++)
        {
            var input = controller.ComputeControl(state);
            state = V(
                a[0, 0] * state[0] + a[0, 1] * state[1] + b[0, 0] * input[0],
                a[1, 0] * state[0] + a[1, 1] * state[1] + b[1, 0] * input[0]);
        }

        Assert.True(
            Math.Abs(state[0]) < 1e-3 && Math.Abs(state[1]) < 1e-3,
            $"The saturated controller did not settle: ({state[0]}, {state[1]}).");
    }

    /// <summary>
    /// A state limit must hold over the whole simulated run. Here the velocity is capped, which the
    /// controller can only honour by anticipating — it has to stop accelerating well before the cap,
    /// since braking is not instantaneous.
    /// </summary>
    [Fact]
    public void Mpc_StateLimit_IsRespectedThroughoutTheRun()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();

        const double VelocityLimit = 1.5;

        var controller = new ModelPredictiveController<double>(
            a, b, Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            new ModelPredictiveControllerOptions<double>
            {
                Horizon = 20,
                StateLowerBounds = V(double.NegativeInfinity, -VelocityLimit),
                StateUpperBounds = V(double.PositiveInfinity, VelocityLimit),
            });

        var state = V(-30.0, 0.0);

        for (int step = 0; step < 120; step++)
        {
            var input = controller.ComputeControl(state);
            state = V(
                a[0, 0] * state[0] + a[0, 1] * state[1] + b[0, 0] * input[0],
                a[1, 0] * state[0] + a[1, 1] * state[1] + b[1, 0] * input[0]);

            Assert.True(
                state[1] <= VelocityLimit + 1e-4 && state[1] >= -VelocityLimit - 1e-4,
                $"Velocity reached {state[1]} at step {step}, outside the limit of " +
                $"±{VelocityLimit}.");
        }

        Assert.True(
            Math.Abs(state[0]) < 1.0,
            $"The controller respected the speed limit but never arrived: position {state[0]}.");
    }

    /// <summary>
    /// A state constraint that is already violated and cannot be recovered from within the horizon
    /// must be reported, not papered over. A caller running real equipment needs to distinguish a
    /// plan from a guess.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Mpc_UnreachableStateConstraint_ReportsInfeasibility()
    {
        await Task.Yield();

        var controller = new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            new ModelPredictiveControllerOptions<double>
            {
                Horizon = 3,
                InputLowerBounds = V(-0.001),
                InputUpperBounds = V(0.001),
                StateLowerBounds = V(-1.0, double.NegativeInfinity),
                StateUpperBounds = V(1.0, double.PositiveInfinity),
            });

        // Position 500 moving away at 50 per step, with an actuator that can barely do anything:
        // no input sequence brings the position inside ±1 within three steps.
        var exception = Assert.Throws<InvalidOperationException>(
            () => controller.ComputeControl(V(500.0, 50.0)));

        Assert.Equal(LinearProgramStatus.Infeasible, controller.LastStatus);
        Assert.Contains("could not be solved", exception.Message, StringComparison.Ordinal);
    }

    #endregion

    #region Structure and validation

    [Fact]
    public void Mpc_TerminalCost_DefaultsToTheRiccatiSolution()
    {
        var a = DoubleIntegrator();
        var b = DoubleIntegratorInput();
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        var regulator = new LinearQuadraticRegulator<double>(a, b, q, r);
        var controller = new ModelPredictiveController<double>(a, b, q, r);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(regulator.CostToGoMatrix[i, j], controller.TerminalCost[i, j], 8);
            }
        }
    }

    [Fact]
    public void Mpc_SuccessfulSolve_ReportsOptimalStatus()
    {
        var controller = new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        controller.ComputeControl(V(1.0, 0.0));

        Assert.Equal(LinearProgramStatus.Optimal, controller.LastStatus);
    }

    [Fact(Timeout = 120000)]
    public async Task Mpc_NonPositiveHorizon_ThrowsAtAssignment()
    {
        await Task.Yield();

        var options = new ModelPredictiveControllerOptions<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() => options.Horizon = 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => options.Horizon = -1);
        Assert.Equal(10, options.Horizon);
    }

    [Fact]
    public void Mpc_InputBoundOfWrongLength_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            new ModelPredictiveControllerOptions<double>
            {
                InputUpperBounds = V(1.0, 2.0),
            }));
    }

    [Fact]
    public void Mpc_TerminalCostOfWrongSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1),
            new ModelPredictiveControllerOptions<double>
            {
                TerminalCost = Matrix<double>.CreateIdentity(3),
            }));
    }

    [Fact]
    public void Mpc_StateOfWrongLength_Throws()
    {
        var controller = new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        Assert.Throws<ArgumentException>(() => controller.ComputeControl(V(1.0)));
    }

    /// <summary>
    /// Inputs must be free to go negative. This verifies that the MPC preserves the quadratic
    /// program's null-as-unbounded contract rather than materializing a numeric zero lower bound.
    /// </summary>
    [Fact]
    public void Mpc_WithNoBoundsConfigured_StillCommandsNegativeInputs()
    {
        var controller = new ModelPredictiveController<double>(
            DoubleIntegrator(), DoubleIntegratorInput(),
            Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1));

        var input = controller.ComputeControl(V(5.0, 0.0));

        Assert.True(
            input[0] < -1e-6,
            $"A positive displacement needs a negative input to correct it, but the controller " +
            $"commanded {input[0]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Mpc_DecimalWithNoBounds_DoesNotRequireInfinity()
    {
        await Task.Yield();

        var controller = new ModelPredictiveController<decimal>(
            M(new decimal[,] { { 0.5m } }), M(new decimal[,] { { 1m } }),
            M(new decimal[,] { { 1m } }), M(new decimal[,] { { 1m } }),
            new ModelPredictiveControllerOptions<decimal> { Horizon = 3 });

        var input = controller.ComputeControl(Vector<decimal>.FromArray(new[] { 2m }));

        Assert.True(input[0] < 0m, $"Expected a corrective negative input, got {input[0]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task MpcOptions_CopyDeepCopiesMutableBoundsAndCost()
    {
        await Task.Yield();

        var original = new ModelPredictiveControllerOptions<double>
        {
            Seed = 9,
            Horizon = 4,
            InputLowerBounds = V(-2.0),
            InputUpperBounds = V(2.0),
            StateLowerBounds = V(-3.0, -4.0),
            StateUpperBounds = V(3.0, 4.0),
            TerminalCost = Matrix<double>.CreateIdentity(2),
        };

        var copy = new ModelPredictiveControllerOptions<double>(original);
        original.InputLowerBounds[0] = -99.0;
        original.TerminalCost[0, 0] = 99.0;

        Assert.Equal(9, copy.Seed);
        Assert.Equal(4, copy.Horizon);
        Assert.Equal(-2.0, copy.InputLowerBounds[0]);
        Assert.Equal(1.0, copy.TerminalCost[0, 0]);
    }

    #endregion
}
