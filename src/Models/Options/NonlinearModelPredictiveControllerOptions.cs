using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the nonlinear model predictive controller.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The controller predicts several steps into the future, repeatedly
/// improves that plan, and can remember the previous plan to make the next decision faster. The
/// defaults provide a usable unconstrained controller.</para>
/// <para><b>Reference:</b> M. Diehl et al., "Real-time optimization and nonlinear model predictive
/// control of processes governed by differential-algebraic equations", <i>Journal of Process
/// Control</i> 12(4), 2002, pp. 577-585; J. B. Rawlings, D. Q. Mayne and M. M. Diehl,
/// <i>Model Predictive Control: Theory, Computation, and Design</i>, 2nd ed., 2017.</para>
/// </remarks>
public class NonlinearModelPredictiveControllerOptions<T> : ModelOptions
{
    private int _horizon = 10;
    private int _sqpIterations = 5;
    private double _stepSize = 1.0;
    private double _tolerance = 1e-8;

    /// <summary>Creates options with the documented defaults.</summary>
    public NonlinearModelPredictiveControllerOptions()
    {
    }

    /// <summary>Creates an independent copy of another nonlinear MPC configuration.</summary>
    /// <param name="other">The options to copy.</param>
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
        InputLowerBounds = other.InputLowerBounds?.Clone();
        InputUpperBounds = other.InputUpperBounds?.Clone();
        TerminalCost = other.TerminalCost?.Clone();
        WarmStart = other.WarmStart;
    }

    /// <summary>
    /// Gets or sets how many steps ahead the controller plans.
    /// </summary>
    /// <value>The prediction horizon, defaulting to 10.</value>
    /// <remarks><para><b>For Beginners:</b> Larger values see farther ahead but make every control
    /// decision more expensive.</para></remarks>
    public int Horizon
    {
        get => _horizon;
        set => _horizon = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "Horizon must be positive.");
    }

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
    /// <para><b>For Beginners:</b> This is how many times the nonlinear plan is corrected before
    /// the controller uses it. More corrections cost more time.</para>
    /// </remarks>
    public int SqpIterations
    {
        get => _sqpIterations;
        set => _sqpIterations = value > 0
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "SqpIterations must be positive.");
    }

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
    /// <para><b>For Beginners:</b> Keep the full step unless the plan oscillates; reduce this value
    /// to make each correction more cautious.</para>
    /// </remarks>
    public double StepSize
    {
        get => _stepSize;
        set => _stepSize = value > 0.0 && value <= 1.0 &&
            !double.IsNaN(value) && !double.IsInfinity(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "StepSize must be finite and in the interval (0, 1].");
    }

    /// <summary>
    /// Gets or sets the change in the planned inputs below which the iteration stops early.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 1e-8.</value>
    /// <remarks><para><b>For Beginners:</b> When a correction changes the plan by less than this,
    /// further work is too small to matter and the controller stops early.</para></remarks>
    public double Tolerance
    {
        get => _tolerance;
        set => _tolerance = value >= 0.0 && !double.IsNaN(value) && !double.IsInfinity(value)
            ? value
            : throw new ArgumentOutOfRangeException(
                nameof(value), value, "Tolerance must be finite and non-negative.");
    }

    /// <summary>
    /// Gets or sets the lower bound on each control input, or <c>null</c> for unbounded below.
    /// </summary>
    /// <value>One lower limit per input, or <c>null</c> for no lower limits.</value>
    /// <remarks><para><b>For Beginners:</b> Put each actuator's smallest allowed command here.
    /// Leave it <c>null</c> when there is no lower limit.</para></remarks>
    public Vector<T>? InputLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bound on each control input, or <c>null</c> for unbounded above.
    /// </summary>
    /// <value>One upper limit per input, or <c>null</c> for no upper limits.</value>
    /// <remarks><para><b>For Beginners:</b> Put each actuator's largest allowed command here.
    /// Leave it <c>null</c> when there is no upper limit.</para></remarks>
    public Vector<T>? InputUpperBounds { get; set; }

    /// <summary>
    /// Gets or sets the terminal cost matrix, or <c>null</c> to reuse the running state cost.
    /// </summary>
    /// <value>A state-by-state cost matrix, or <c>null</c> to reuse the running cost.</value>
    /// <remarks>
    /// <para>
    /// There is no nonlinear equivalent of the Riccati solution to fall back on, so unlike the linear
    /// controller this cannot default to something principled. Reusing the running cost is the
    /// conventional choice; a caller who has linearized about their intended operating point can do
    /// better by passing that linearization's Riccati solution here.
    /// </para>
    /// <para><b>For Beginners:</b> This prices the state left at the end of the visible planning
    /// window. The default is suitable for general use.</para>
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
    /// <para><b>For Beginners:</b> Keep this enabled. It reuses the previous answer as the next
    /// starting point and usually reduces the work needed.</para>
    /// </remarks>
    public bool WarmStart { get; set; } = true;
}
