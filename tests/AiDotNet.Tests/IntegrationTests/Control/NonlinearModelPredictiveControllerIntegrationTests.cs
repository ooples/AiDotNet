#nullable disable
using AiDotNet.Control;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Control;

/// <summary>
/// Integration tests for nonlinear model predictive control.
/// </summary>
/// <remarks>
/// CRITICAL: The anchoring test needs no expected number. Given a linear system, the linearization
/// the nonlinear controller takes is exact, so it must reproduce the linear controller's plan —
/// despite computing it by a completely different route (time-varying condensation in deviation
/// coordinates, iterated, versus a single time-invariant condensation solved once). Where the two
/// disagree, one of them is wrong.
///
/// The nonlinear tests then check the thing the linear controller cannot do at all: stabilizing a
/// system whose behaviour changes with where it is.
/// If a test fails, FIX THE CONTROLLER — do not relax the assertion.
/// </remarks>
public class NonlinearModelPredictiveControllerIntegrationTests
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

    #region Agreement with the linear controller

    /// <summary>
    /// On a linear system the linearization is exact, so the nonlinear controller must arrive at the
    /// same plan as the linear one.
    /// </summary>
    [Theory]
    [InlineData(3.0, -1.0)]
    [InlineData(-5.0, 2.0)]
    [InlineData(0.5, 0.5)]
    public void Nmpc_OnALinearSystem_MatchesTheLinearController(double position, double velocity)
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });
        var q = Matrix<double>.CreateIdentity(2);
        var r = Matrix<double>.CreateIdentity(1);

        // The linear controller must be told to use Q as its terminal cost, since that is what the
        // nonlinear one defaults to — there is no Riccati solution to fall back on when the dynamics
        // are arbitrary.
        var linear = new ModelPredictiveController<double>(
            a, b, q, r,
            new ModelPredictiveControllerOptions<double> { Horizon = 8, TerminalCost = q });

        var nonlinear = new NonlinearModelPredictiveController<double>(
            dynamics: (x, u) => V(
                a[0, 0] * x[0] + a[0, 1] * x[1] + b[0, 0] * u[0],
                a[1, 0] * x[0] + a[1, 1] * x[1] + b[1, 0] * u[0]),
            jacobians: (x, u) => (a, b),
            stateCost: q,
            inputCost: r,
            options: new NonlinearModelPredictiveControllerOptions<double>
            {
                Horizon = 8,
                WarmStart = false,
            });

        var state = V(position, velocity);

        Assert.Equal(
            linear.ComputeControl(state)[0], nonlinear.ComputeControl(state)[0], 4);
    }

    /// <summary>
    /// With exact linearization the correction is right the first time, so the iteration must
    /// converge almost immediately rather than grinding to its limit.
    /// </summary>
    [Fact]
    public void Nmpc_OnALinearSystem_ConvergesInVeryFewIterations()
    {
        var a = M(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
        var b = M(new[,] { { 0.5 }, { 1.0 } });

        var controller = new NonlinearModelPredictiveController<double>(
            dynamics: (x, u) => V(
                x[0] + x[1] + 0.5 * u[0],
                x[1] + u[0]),
            jacobians: (x, u) => (a, b),
            stateCost: Matrix<double>.CreateIdentity(2),
            inputCost: Matrix<double>.CreateIdentity(1),
            options: new NonlinearModelPredictiveControllerOptions<double>
            {
                Horizon = 6,
                SqpIterations = 10,
                WarmStart = false,
            });

        controller.ComputeControl(V(4.0, -1.0));

        Assert.True(
            controller.LastIterationCount <= 3,
            $"A linear system should converge in one or two corrections; it took " +
            $"{controller.LastIterationCount}.");
    }

    #endregion

    #region Genuinely nonlinear behaviour

    /// <summary>
    /// A pendulum, whose restoring force varies as the sine of the angle. Left alone it swings
    /// forever — the model has no damping — so any decay to rest is the controller's doing.
    /// </summary>
    private static NonlinearModelPredictiveController<double> BuildPendulumController(
        NonlinearModelPredictiveControllerOptions<double> options)
    {
        const double Dt = 0.05;

        return new NonlinearModelPredictiveController<double>(
            dynamics: (x, u) => V(
                x[0] + Dt * x[1],
                x[1] + Dt * (-Math.Sin(x[0]) + u[0])),
            jacobians: (x, u) => (
                M(new[,] { { 1.0, Dt }, { -Dt * Math.Cos(x[0]), 1.0 } }),
                M(new[,] { { 0.0 }, { Dt } })),
            stateCost: Matrix<double>.CreateIdentity(2),
            inputCost: M(new[,] { { 0.1 } }),
            options: options);
    }

    private static Vector<double> AdvancePendulum(Vector<double> state, double input)
    {
        const double Dt = 0.05;
        return V(
            state[0] + Dt * state[1],
            state[1] + Dt * (-Math.Sin(state[0]) + input));
    }

    /// <summary>
    /// The undamped pendulum must not settle on its own — this establishes that the next test is
    /// measuring the controller rather than the model quietly doing the work.
    /// </summary>
    [Fact]
    public void Pendulum_WithNoControl_KeepsSwinging()
    {
        var state = V(0.5, 0.0);
        double largestAngleLate = 0.0;

        for (int step = 0; step < 2000; step++)
        {
            state = AdvancePendulum(state, 0.0);
            if (step > 1500) largestAngleLate = Math.Max(largestAngleLate, Math.Abs(state[0]));
        }

        Assert.True(
            largestAngleLate > 0.3,
            $"The uncontrolled pendulum should still be swinging; its late-run amplitude was only " +
            $"{largestAngleLate}.");
    }

    /// <summary>
    /// With control it must come to rest.
    /// </summary>
    [Fact]
    public void Nmpc_Pendulum_IsDrivenToRest()
    {
        var controller = BuildPendulumController(
            new NonlinearModelPredictiveControllerOptions<double> { Horizon = 25 });

        var state = V(0.5, 0.0);

        for (int step = 0; step < 600; step++)
        {
            var input = controller.ComputeControl(state);
            state = AdvancePendulum(state, input[0]);
        }

        Assert.True(
            Math.Abs(state[0]) < 0.02 && Math.Abs(state[1]) < 0.02,
            $"The controller failed to settle the pendulum: angle {state[0]}, rate {state[1]}.");
    }

    /// <summary>
    /// From a large angle, where the sine is markedly not the angle, the controller must still work
    /// — this is where a single fixed linearization would go wrong and the resolving iteration earns
    /// its cost.
    /// </summary>
    [Fact]
    public void Nmpc_PendulumFromALargeAngle_StillSettles()
    {
        var controller = BuildPendulumController(
            new NonlinearModelPredictiveControllerOptions<double> { Horizon = 30 });

        var state = V(2.0, 0.0);

        for (int step = 0; step < 800; step++)
        {
            var input = controller.ComputeControl(state);
            state = AdvancePendulum(state, input[0]);
        }

        Assert.True(
            Math.Abs(state[0]) < 0.05 && Math.Abs(state[1]) < 0.05,
            $"The controller failed from a large angle: angle {state[0]}, rate {state[1]}.");
    }

    /// <summary>
    /// A torque limit must be respected at every step, and it must bind — the limit here is well
    /// below what the controller would otherwise use from this angle.
    /// </summary>
    [Fact]
    public void Nmpc_TorqueLimit_IsRespectedThroughout()
    {
        const double Limit = 0.2;

        var controller = BuildPendulumController(
            new NonlinearModelPredictiveControllerOptions<double>
            {
                Horizon = 25,
                InputLowerBounds = V(-Limit),
                InputUpperBounds = V(Limit),
            });

        var state = V(1.5, 0.0);
        bool everSaturated = false;

        for (int step = 0; step < 400; step++)
        {
            var input = controller.ComputeControl(state);

            Assert.True(
                input[0] >= -Limit - 1e-5 && input[0] <= Limit + 1e-5,
                $"Step {step} commanded {input[0]}, outside the limit of ±{Limit}.");

            if (Math.Abs(input[0]) > Limit - 1e-3) everSaturated = true;

            state = AdvancePendulum(state, input[0]);
        }

        Assert.True(
            everSaturated,
            "The limit never bound, so this test would pass even if bounds were ignored.");
    }

    /// <summary>
    /// Every step of the plan must respect the bound, not merely the first.
    /// </summary>
    [Fact]
    public void Nmpc_WholePlan_RespectsTheTorqueLimit()
    {
        const double Limit = 0.15;

        var controller = BuildPendulumController(
            new NonlinearModelPredictiveControllerOptions<double>
            {
                Horizon = 12,
                InputLowerBounds = V(-Limit),
                InputUpperBounds = V(Limit),
            });

        var plan = controller.ComputePlan(V(1.2, 0.3));

        Assert.Equal(12, plan.Rows);

        for (int k = 0; k < plan.Rows; k++)
        {
            Assert.True(
                plan[k, 0] >= -Limit - 1e-5 && plan[k, 0] <= Limit + 1e-5,
                $"Step {k} of the plan commands {plan[k, 0]}, outside ±{Limit}.");
        }
    }

    #endregion

    #region Warm starting

    /// <summary>
    /// Warm starting must actually help: after the first step, consecutive problems are nearly
    /// identical and the shifted previous plan should already be close enough to converge in fewer
    /// corrections than a cold start needs.
    /// </summary>
    [Fact]
    public void Nmpc_WarmStart_ConvergesFasterThanColdStart()
    {
        var warm = BuildPendulumController(
            new NonlinearModelPredictiveControllerOptions<double>
            {
                Horizon = 20,
                SqpIterations = 20,
                WarmStart = true,
            });

        var cold = BuildPendulumController(
            new NonlinearModelPredictiveControllerOptions<double>
            {
                Horizon = 20,
                SqpIterations = 20,
                WarmStart = false,
            });

        var state = V(1.0, 0.0);
        int warmTotal = 0;
        int coldTotal = 0;

        for (int step = 0; step < 30; step++)
        {
            warm.ComputeControl(state);
            cold.ComputeControl(state);

            warmTotal += warm.LastIterationCount;
            coldTotal += cold.LastIterationCount;

            state = AdvancePendulum(state, warm.ComputeControl(state)[0]);
        }

        Assert.True(
            warmTotal <= coldTotal,
            $"Warm starting used {warmTotal} corrections against a cold start's {coldTotal}; it " +
            "should never need more.");
    }

    #endregion

    #region Validation

    private static NonlinearModelPredictiveController<double> BuildTrivial(
        NonlinearModelPredictiveControllerOptions<double> options)
        => new(
            dynamics: (x, u) => V(x[0] + u[0]),
            jacobians: (x, u) => (Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1)),
            stateCost: Matrix<double>.CreateIdentity(1),
            inputCost: Matrix<double>.CreateIdentity(1),
            options: options);

    [Fact]
    public void Nmpc_NullDynamics_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new NonlinearModelPredictiveController<double>(
            null,
            (x, u) => (Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1)),
            Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1)));
    }

    [Fact]
    public void Nmpc_ZeroSqpIterations_Throws()
    {
        // Zero corrections would return the initial guess untouched, which is not a control law.
        Assert.Throws<ArgumentException>(() => BuildTrivial(
            new NonlinearModelPredictiveControllerOptions<double> { SqpIterations = 0 }));
    }

    [Fact]
    public void Nmpc_StepSizeAboveOne_Throws()
    {
        Assert.Throws<ArgumentException>(() => BuildTrivial(
            new NonlinearModelPredictiveControllerOptions<double> { StepSize = 1.5 }));
    }

    [Fact]
    public void Nmpc_NonPositiveHorizon_Throws()
    {
        Assert.Throws<ArgumentException>(() => BuildTrivial(
            new NonlinearModelPredictiveControllerOptions<double> { Horizon = 0 }));
    }

    [Fact]
    public void Nmpc_StateOfWrongLength_Throws()
    {
        var controller = BuildTrivial(new NonlinearModelPredictiveControllerOptions<double>());
        Assert.Throws<ArgumentException>(() => controller.ComputeControl(V(1.0, 2.0)));
    }

    [Fact]
    public void Nmpc_DynamicsReturningTheWrongShape_Throws()
    {
        var controller = new NonlinearModelPredictiveController<double>(
            dynamics: (x, u) => V(1.0, 2.0),
            jacobians: (x, u) => (Matrix<double>.CreateIdentity(1), Matrix<double>.CreateIdentity(1)),
            stateCost: Matrix<double>.CreateIdentity(1),
            inputCost: Matrix<double>.CreateIdentity(1));

        Assert.Throws<InvalidOperationException>(() => controller.ComputeControl(V(1.0)));
    }

    [Fact]
    public void Nmpc_JacobianOfTheWrongShape_Throws()
    {
        var controller = new NonlinearModelPredictiveController<double>(
            dynamics: (x, u) => V(x[0] + u[0]),
            jacobians: (x, u) => (Matrix<double>.CreateIdentity(2), Matrix<double>.CreateIdentity(1)),
            stateCost: Matrix<double>.CreateIdentity(1),
            inputCost: Matrix<double>.CreateIdentity(1));

        Assert.Throws<InvalidOperationException>(() => controller.ComputeControl(V(1.0)));
    }

    #endregion
}
