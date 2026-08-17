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
/// </remarks>
public class ModelPredictiveControllerOptions<T> : ModelOptions
{
    /// <summary>Initializes the options with documented defaults.</summary>
    public ModelPredictiveControllerOptions()
    {
    }

    /// <summary>Initializes the options by copying another configuration.</summary>
    /// <param name="other">The configuration to copy.</param>
    /// <remarks>
    /// <para>
    /// The bound vectors and terminal cost are carried across by reference. They are treated as
    /// immutable problem data — a caller who mutates a bound vector after handing it over has changed
    /// the problem, and no copy here would make that safe.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public ModelPredictiveControllerOptions(ModelPredictiveControllerOptions<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        Horizon = other.Horizon;
        InputLowerBounds = other.InputLowerBounds;
        InputUpperBounds = other.InputUpperBounds;
        StateLowerBounds = other.StateLowerBounds;
        StateUpperBounds = other.StateUpperBounds;
        TerminalCost = other.TerminalCost;
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
    /// </remarks>
    public int Horizon { get; set; } = 10;

    /// <summary>
    /// Gets or sets the lower bound on each control input, or <c>null</c> for unbounded below.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Input constraints are the reason MPC exists. Every real actuator saturates, and a regulator
    /// that ignores that will command what it cannot deliver and then behave in ways its design never
    /// predicted. MPC is the standard answer because it accounts for the limit while planning rather
    /// than clipping the answer afterwards.
    /// </para>
    /// </remarks>
    public Vector<T>? InputLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bound on each control input, or <c>null</c> for unbounded above.
    /// </summary>
    public Vector<T>? InputUpperBounds { get; set; }

    /// <summary>
    /// Gets or sets the lower bound on each state over the horizon, or <c>null</c> for unbounded.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Entries set to negative infinity leave that state unconstrained, so a single limit can be
    /// imposed on one state without inventing bounds for the others.
    /// </para>
    /// </remarks>
    public Vector<T>? StateLowerBounds { get; set; }

    /// <summary>
    /// Gets or sets the upper bound on each state over the horizon, or <c>null</c> for unbounded.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Entries set to positive infinity leave that state unconstrained.
    /// </para>
    /// <para>
    /// Be aware that state constraints, unlike input constraints, can make the problem infeasible:
    /// there may be no input sequence that keeps a state within its limit from where the system
    /// currently is. A controller that must never fail should treat state limits as soft, or supply
    /// enough horizon to avoid painting itself into a corner.
    /// </para>
    /// </remarks>
    public Vector<T>? StateUpperBounds { get; set; }

    /// <summary>
    /// Gets or sets the terminal cost matrix, or <c>null</c> to use the infinite-horizon Riccati
    /// solution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The terminal cost prices the state left at the end of the horizon, standing in for all the
    /// cost that would still be incurred beyond it. Using the Riccati solution makes that stand-in
    /// exact for the unconstrained problem, which has two consequences worth having: the controller
    /// then reproduces the linear quadratic regulator exactly when no constraint is active, and it
    /// inherits the classical nominal stability guarantee. Setting this to zero instead — a common
    /// shortcut — discards both and can destabilize a short-horizon controller.
    /// </para>
    /// </remarks>
    public Matrix<T>? TerminalCost { get; set; }
}
