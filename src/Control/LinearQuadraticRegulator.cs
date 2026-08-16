using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// The infinite-horizon linear quadratic regulator: the optimal state-feedback controller for a
/// linear system with a quadratic cost.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Minimizes <c>Σ (xᵀQx + uᵀRu)</c> in discrete time, or <c>∫ (xᵀQx + uᵀRu) dt</c> in continuous
/// time, subject to the system's dynamics. The answer is remarkable and is the reason LQR is the
/// foundation of linear control: the optimal input is a constant linear function of the current
/// state, <c>u = −Kx</c>. No lookahead, no memory, no schedule — one matrix multiply, and it is
/// optimal over an infinite horizon.
/// </para>
/// <para>
/// Follows the standard development in R. E. Kalman, "Contributions to the Theory of Optimal
/// Control", <i>Boletín de la Sociedad Matemática Mexicana</i> 5, 1960, pp. 102-119. The gain comes
/// from the algebraic Riccati equation, solved by
/// <see cref="DiscreteAlgebraicRiccatiSolver{T}"/> or <see cref="ContinuousAlgebraicRiccatiSolver{T}"/>
/// according to the time domain.
/// </para>
/// <para>
/// <b>Choosing Q and R is the whole design.</b> They are not tuning knobs bolted onto an algorithm;
/// they <i>are</i> the specification. <c>Q</c> prices deviations of the state, <c>R</c> prices
/// control effort, and only their ratio matters — scaling both by the same factor leaves the gain
/// unchanged. Raising <c>Q</c> relative to <c>R</c> buys a faster, more aggressive response paid for
/// with larger inputs.
/// </para>
/// <para><b>For Beginners:</b> You describe how the system moves, say how much you dislike being
/// away from the target, and say how much you dislike using the controls. This works out the best
/// possible trade-off and hands you a matrix. From then on, controlling the system is multiplying
/// the current state by that matrix and negating it.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // A double integrator sampled at 1 Hz: position and velocity, force input.
/// var a = new Matrix&lt;double&gt;(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } });
/// var b = new Matrix&lt;double&gt;(new[,] { { 0.5 }, { 1.0 } });
/// var q = Matrix&lt;double&gt;.CreateIdentity(2);
/// var r = Matrix&lt;double&gt;.CreateIdentity(1);
///
/// var regulator = new LinearQuadraticRegulator&lt;double&gt;(a, b, q, r);
/// var input = regulator.ComputeControl(new Vector&lt;double&gt;(new[] { 5.0, 0.0 }));
/// </code>
/// </example>
public sealed class LinearQuadraticRegulator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Matrix<T> _stateMatrix;
    private readonly Matrix<T> _inputMatrix;

    /// <summary>
    /// Creates a linear quadratic regulator and solves for its gain.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>, <c>n</c>-by-<c>n</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>, <c>n</c>-by-<c>m</c>.</param>
    /// <param name="stateCost">
    /// The state cost <c>Q</c>, symmetric positive-semidefinite: what deviation costs.
    /// </param>
    /// <param name="inputCost">
    /// The input cost <c>R</c>, symmetric positive-definite: what control effort costs.
    /// </param>
    /// <param name="timeDomain">
    /// Whether the system is discrete or continuous. Defaults to
    /// <see cref="ControlTimeDomain.Discrete"/>, which is what a sampled control loop needs.
    /// </param>
    /// <param name="options">
    /// Riccati solver configuration, or <c>null</c> for the defaults.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when a required matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no stabilizing solution exists, which means no feedback whatever can hold this
    /// system's state bounded.
    /// </exception>
    public LinearQuadraticRegulator(
        Matrix<T> stateMatrix,
        Matrix<T> inputMatrix,
        Matrix<T> stateCost,
        Matrix<T> inputCost,
        ControlTimeDomain timeDomain = ControlTimeDomain.Discrete,
        AlgebraicRiccatiSolverOptions? options = null)
    {
        RiccatiValidation<T>.Validate(stateMatrix, inputMatrix, stateCost, inputCost);

        _stateMatrix = stateMatrix;
        _inputMatrix = inputMatrix;
        TimeDomain = timeDomain;

        var solverOptions = options ?? new AlgebraicRiccatiSolverOptions();

        if (timeDomain == ControlTimeDomain.Discrete)
        {
            RiccatiSolution = new DiscreteAlgebraicRiccatiSolver<T>(solverOptions)
                .Solve(stateMatrix, inputMatrix, stateCost, inputCost);

            Gain = DiscreteAlgebraicRiccatiSolver<T>.ComputeGain(
                RiccatiSolution.Solution, stateMatrix, inputMatrix, inputCost);
        }
        else
        {
            RiccatiSolution = new ContinuousAlgebraicRiccatiSolver<T>(solverOptions)
                .Solve(stateMatrix, inputMatrix, stateCost, inputCost);

            Gain = ContinuousAlgebraicRiccatiSolver<T>.ComputeGain(
                RiccatiSolution.Solution, inputMatrix, inputCost);
        }
    }

    /// <summary>Gets whether this regulator was designed for a discrete or continuous system.</summary>
    public ControlTimeDomain TimeDomain { get; }

    /// <summary>
    /// Gets the optimal feedback gain <c>K</c>. The optimal input is <c>u = −Kx</c>.
    /// </summary>
    public Matrix<T> Gain { get; }

    /// <summary>
    /// Gets the underlying Riccati solution, including its residual and whether it converged.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Exposed rather than hidden because a controller whose Riccati equation did not converge is a
    /// controller you should not deploy, and the caller is the only one positioned to decide what to
    /// do about it.
    /// </para>
    /// </remarks>
    public AlgebraicRiccatiSolution<T> RiccatiSolution { get; }

    /// <summary>
    /// Gets the cost-to-go matrix <c>P</c>, for which <c>xᵀPx</c> is the total remaining cost from
    /// state <c>x</c> under optimal control.
    /// </summary>
    public Matrix<T> CostToGoMatrix => RiccatiSolution.Solution;

    /// <summary>
    /// Computes the optimal control input <c>u = −Kx</c> for the current state.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>The optimal input.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the state's length does not match the system.
    /// </exception>
    public Vector<T> ComputeControl(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (state.Length != _stateMatrix.Rows)
        {
            throw new ArgumentException(
                $"The state must have {_stateMatrix.Rows} entries to match the system; it has " +
                $"{state.Length}.", nameof(state));
        }

        var product = ControlMath<T>.Multiply(Gain, state);

        var control = new Vector<T>(product.Length);
        for (int i = 0; i < product.Length; i++) control[i] = NumOps.Negate(product[i]);

        return control;
    }

    /// <summary>
    /// Returns <c>xᵀPx</c>, the total remaining cost of driving <paramref name="state"/> to the
    /// origin under optimal control.
    /// </summary>
    /// <param name="state">The state to evaluate.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the state's length does not match the system.
    /// </exception>
    public T CostToGo(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (state.Length != _stateMatrix.Rows)
        {
            throw new ArgumentException(
                $"The state must have {_stateMatrix.Rows} entries to match the system; it has " +
                $"{state.Length}.", nameof(state));
        }

        return state.DotProduct(ControlMath<T>.Multiply(CostToGoMatrix, state));
    }

    /// <summary>
    /// Returns the closed-loop state matrix <c>A − BK</c>, which governs how the controlled system
    /// evolves.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the object to inspect when asking whether the design actually works: the closed loop
    /// is stable exactly when this matrix's eigenvalues lie inside the unit circle (discrete) or in
    /// the left half-plane (continuous). LQR guarantees that whenever a stabilizing solution exists,
    /// so it is a check rather than a question — but it is the check worth running before trusting a
    /// controller with hardware.
    /// </para>
    /// </remarks>
    public Matrix<T> ClosedLoopMatrix =>
        ControlMath<T>.Subtract(_stateMatrix, ControlMath<T>.Multiply(_inputMatrix, Gain));
}
