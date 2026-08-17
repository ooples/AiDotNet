using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the nonlinear model predictive controller.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NonlinearModelPredictiveControllerOptions<T> : ModelOptions
{
    /// <summary>Initializes the options with documented defaults.</summary>
    public NonlinearModelPredictiveControllerOptions()
    {
    }

    /// <summary>Initializes the options by copying another configuration.</summary>
    /// <param name="other">The configuration to copy.</param>
    /// <remarks>
    /// <para>
    /// The bound vectors and terminal cost are carried across by reference, as immutable problem
    /// data.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public NonlinearModelPredictiveControllerOptions(
        NonlinearModelPredictiveControllerOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Horizon = other.Horizon;
        SqpIterations = other.SqpIterations;
        StepSize = other.StepSize;
        Tolerance = other.Tolerance;
        InputLowerBounds = other.InputLowerBounds;
        InputUpperBounds = other.InputUpperBounds;
        TerminalCost = other.TerminalCost;
        WarmStart = other.WarmStart;
    }

    /// <summary>
    /// Gets or sets how many steps ahead the controller plans.
    /// </summary>
    /// <value>The prediction horizon, defaulting to 10.</value>
    public int Horizon { get; set; } = 10;

    /// <summary>
    /// Gets or sets how many times the linearization is refined per control step.
    /// </summary>
    /// <value>The number of sequential quadratic programming iterations, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// Each iteration re-linearizes about the trajectory the previous one produced and solves another
    /// quadratic program, so this is the knob that trades computation against how far the plan may
    /// stray from where the linearization is valid. Iterating to convergence gives the exact solution
    /// of the nonlinear problem; stopping after one gives the <i>real-time iteration</i> scheme of
    /// Diehl et al., which is what makes nonlinear MPC feasible at millisecond rates and is often
    /// enough because the next control step will re-solve anyway.
    /// </para>
    /// </remarks>
    public int SqpIterations { get; set; } = 5;

    /// <summary>
    /// Gets or sets the step size applied to each sequential quadratic programming update.
    /// </summary>
    /// <value>The damping factor, defaulting to 1.0 (a full step).</value>
    /// <remarks>
    /// <para>
    /// A full step is correct when the linearization is accurate over the whole update. On a
    /// strongly nonlinear system it can overshoot into a region where the model no longer resembles
    /// the plant, and the iteration oscillates instead of converging; damping below one trades
    /// convergence speed for reliability there.
    /// </para>
    /// </remarks>
    public double StepSize { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the change in the planned inputs below which the iteration stops early.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 1e-8.</value>
    public double Tolerance { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the lower bound on each control input, or <c>null</c> for unbounded below.
    /// </summary>
    public Vector<T>? InputLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bound on each control input, or <c>null</c> for unbounded above.
    /// </summary>
    public Vector<T>? InputUpperBounds { get; set; }

    /// <summary>
    /// Gets or sets the terminal cost matrix, or <c>null</c> to reuse the running state cost.
    /// </summary>
    /// <remarks>
    /// <para>
    /// There is no nonlinear equivalent of the Riccati solution to fall back on, so unlike the linear
    /// controller this cannot default to something principled. Reusing the running cost is the
    /// conventional choice; a caller who has linearized about their intended operating point can do
    /// better by passing that linearization's Riccati solution here.
    /// </para>
    /// </remarks>
    public Matrix<T>? TerminalCost { get; set; }

    /// <summary>
    /// Gets or sets whether to warm-start each control step from the previous step's plan.
    /// </summary>
    /// <value><c>true</c> by default.</value>
    /// <remarks>
    /// <para>
    /// Consecutive control steps solve nearly the same problem — the state has moved by one step and
    /// nothing else has changed — so the previous plan, shifted forward by one, is an excellent
    /// starting guess. This is the single largest speedup available in nonlinear MPC and costs
    /// nothing but remembering the last answer.
    /// </para>
    /// </remarks>
    public bool WarmStart { get; set; } = true;
}
