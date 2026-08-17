using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for the linear model predictive controller.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Every bound here is optional and defaults to absent, which means unconstrained. An MPC with no
/// constraints at all is a legitimate configuration — and a useful one, because it must reproduce
/// the linear quadratic regulator exactly, which is the sharpest check that the rest of the setup is
/// right.
/// </para>
/// <para><b>For Beginners:</b> The horizon controls how far ahead the controller looks. The bound
/// vectors describe real actuator and safety limits, and the terminal cost describes what happens
/// after the visible horizon. The defaults provide unconstrained control with a principled
/// automatically computed terminal cost, so most users only need to choose a horizon.</para>
/// <para><b>Reference:</b> J. B. Rawlings, D. Q. Mayne and M. M. Diehl,
/// <i>Model Predictive Control: Theory, Computation, and Design</i>, 2nd ed., 2017; D. Q. Mayne et
/// al., "Constrained model predictive control: Stability and optimality", <i>Automatica</i> 36(6),
/// 2000, pp. 789-814.</para>
/// </remarks>
public class ModelPredictiveControllerOptions<T> : ModelOptions
{
    /// <summary>Creates options with the documented defaults.</summary>
    public ModelPredictiveControllerOptions()
    {
    }

    /// <summary>Creates an independent copy of another MPC configuration.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public ModelPredictiveControllerOptions(ModelPredictiveControllerOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Horizon = other.Horizon;
        InputLowerBounds = other.InputLowerBounds?.Clone();
        InputUpperBounds = other.InputUpperBounds?.Clone();
        StateLowerBounds = other.StateLowerBounds?.Clone();
        StateUpperBounds = other.StateUpperBounds?.Clone();
        TerminalCost = other.TerminalCost?.Clone();
    }

    /// <summary>
    /// Gets or sets how many steps ahead the controller plans.
    /// </summary>
    /// <value>The prediction horizon, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// The horizon is the main design choice in MPC and it trades two failure modes against each
    /// other. Too short and the controller is myopic: it cannot see a constraint coming in time to
    /// avoid it, and it can steer into a corner it cannot get out of. Too long and every step costs
    /// more to solve, with diminishing returns once the horizon exceeds the system's settling time.
    /// </para>
    /// <para>
    /// A terminal cost mitigates a short horizon substantially — it stands in for everything beyond
    /// the horizon — which is why the default terminal cost is the infinite-horizon Riccati solution
    /// rather than zero.
    /// </para>
    /// <para><b>For Beginners:</b> Increase this when the controller reacts too late to a future
    /// limit. Decrease it when each control step is too expensive to compute.</para>
    /// </remarks>
    public int Horizon { get; set; } = 10;

    /// <summary>
    /// Gets or sets the lower bound on each control input, or <c>null</c> for unbounded below.
    /// </summary>
    /// <value>One lower bound per input, or <c>null</c> for no lower input bounds.</value>
    /// <remarks>
    /// <para>
    /// Input constraints are the reason MPC exists. Every real actuator saturates, and a regulator
    /// that ignores that will command what it cannot deliver and then behave in ways its design never
    /// predicted. MPC is the standard answer because it accounts for the limit while planning rather
    /// than clipping the answer afterwards.
    /// </para>
    /// <para><b>For Beginners:</b> Put each actuator's smallest allowed command here. Leave this
    /// <c>null</c> when inputs have no lower limit.</para>
    /// </remarks>
    public Vector<T>? InputLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bound on each control input, or <c>null</c> for unbounded above.
    /// </summary>
    /// <value>One upper bound per input, or <c>null</c> for no upper input bounds.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Put each actuator's largest allowed command here. Leave this
    /// <c>null</c> when inputs have no upper limit.</para>
    /// </remarks>
    public Vector<T>? InputUpperBounds { get; set; }

    /// <summary>
    /// Gets or sets the lower bound on each state over the horizon, or <c>null</c> for unbounded.
    /// </summary>
    /// <value>
    /// One lower bound per state, or <c>null</c> for none. <see cref="INumericOperations{T}.MinValue"/>
    /// can mark an individual generic-numeric state as unbounded.
    /// </value>
    /// <remarks>
    /// <para>
    /// IEEE negative infinity, or <see cref="INumericOperations{T}.MinValue"/> for a type without
    /// infinity such as <see cref="decimal"/>, leaves that state unconstrained. This permits one
    /// state limit without inventing bounds for the others.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for safety limits such as a minimum temperature or
    /// position. A state limit can make a control step infeasible when recovery is impossible.</para>
    /// </remarks>
    public Vector<T>? StateLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bound on each state over the horizon, or <c>null</c> for unbounded.
    /// </summary>
    /// <value>
    /// One upper bound per state, or <c>null</c> for none. <see cref="INumericOperations{T}.MaxValue"/>
    /// can mark an individual generic-numeric state as unbounded.
    /// </value>
    /// <remarks>
    /// <para>
    /// IEEE positive infinity, or <see cref="INumericOperations{T}.MaxValue"/> for a type without
    /// infinity such as <see cref="decimal"/>, leaves that state unconstrained.
    /// </para>
    /// <para>
    /// Be aware that state constraints, unlike input constraints, can make the problem infeasible:
    /// there may be no input sequence that keeps a state within its limit from where the system
    /// currently is. A controller that must never fail should treat state limits as soft, or supply
    /// enough horizon to avoid painting itself into a corner.
    /// </para>
    /// <para><b>For Beginners:</b> Use this for safety limits such as a maximum temperature or
    /// position. Leave it <c>null</c> when no state has an upper limit.</para>
    /// </remarks>
    public Vector<T>? StateUpperBounds { get; set; }

    /// <summary>
    /// Gets or sets the terminal cost matrix, or <c>null</c> to use the infinite-horizon Riccati
    /// solution.
    /// </summary>
    /// <value>
    /// A state-by-state terminal cost matrix, or <c>null</c> to compute the principled default.
    /// </value>
    /// <remarks>
    /// <para>
    /// The terminal cost prices the state left at the end of the horizon, standing in for all the
    /// cost that would still be incurred beyond it. Using the Riccati solution makes that stand-in
    /// exact for the unconstrained problem, which has two consequences worth having: the controller
    /// then reproduces the linear quadratic regulator exactly when no constraint is active, and it
    /// inherits the classical nominal stability guarantee. Setting this to zero instead — a common
    /// shortcut — discards both and can destabilize a short-horizon controller.
    /// </para>
    /// <para><b>For Beginners:</b> Leave this <c>null</c> unless you have a domain-specific cost for
    /// the state at the end of the planning window.</para>
    /// </remarks>
    public Matrix<T>? TerminalCost { get; set; }
}
