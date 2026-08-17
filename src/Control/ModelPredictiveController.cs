using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Solvers.InteriorPoint;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Linear model predictive control: at every step, plan the whole future subject to the actual
/// limits, then apply only the first move.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Minimizes <c>Σ (xₖᵀQxₖ + uₖᵀRuₖ) + x_NᵀP_f x_N</c> over a finite horizon subject to the dynamics
/// and to bounds on the inputs and states, by solving one quadratic program per step. Follows the
/// standard condensed formulation; see J. B. Rawlings, D. Q. Mayne and M. M. Diehl, <i>Model
/// Predictive Control: Theory, Computation, and Design</i> (2nd ed., Nob Hill 2017), and the survey
/// D. Q. Mayne, J. B. Rawlings, C. V. Rao and P. O. M. Scokaert, "Constrained model predictive
/// control: Stability and optimality", <i>Automatica</i> 36(6), 2000, pp. 789-814.
/// </para>
/// <para>
/// <b>Why plan the whole horizon and then discard all but the first move.</b> It sounds wasteful, and
/// it is what makes MPC work. The plan is optimal for the model, and the model is wrong; by the next
/// step the true state is not where the plan said it would be. Re-planning from the state that
/// actually occurred turns an open-loop plan into feedback, and the discarded tail was never meant to
/// be executed — it exists so that the first move accounts for what comes after it.
/// </para>
/// <para>
/// <b>Where it beats LQR.</b> Nowhere, when nothing is constrained: with the terminal cost set to the
/// infinite-horizon Riccati solution the two produce identical inputs, and the regulator is far
/// cheaper. The moment an actuator saturates or a state must not be exceeded, they diverge — the
/// regulator commands what it cannot deliver and a clipped LQR is no longer optimal or even
/// necessarily stable, while MPC plans around the limit. That is the entire trade: MPC costs a
/// quadratic program per step and buys constraints.
/// </para>
/// <para>
/// <b>Condensing.</b> The states are eliminated rather than kept as variables, so the quadratic
/// program is over the inputs alone — <c>horizon × inputs</c> variables rather than
/// <c>horizon × (states + inputs)</c>. Everything that depends only on the model and the costs is
/// built once in the constructor; each step recomputes only the linear term, which is the part that
/// depends on the current state.
/// </para>
/// <para><b>For Beginners:</b> Think of driving. You look ahead as far as you can see, work out a
/// whole route that respects the lane edges and the speed limit, then turn the wheel by just the
/// amount that route calls for right now. A moment later you look again and re-plan from where you
/// actually are. You never execute the whole plan, but you could not choose the current move well
/// without having made one.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // A double integrator whose actuator saturates at ±0.5.
/// var options = new ModelPredictiveControllerOptions&lt;double&gt;
/// {
///     Horizon = 20,
///     InputLowerBounds = new Vector&lt;double&gt;(new[] { -0.5 }),
///     InputUpperBounds = new Vector&lt;double&gt;(new[] { 0.5 }),
/// };
///
/// var controller = new ModelPredictiveController&lt;double&gt;(a, b, q, r, options);
/// var input = controller.ComputeControl(currentState);
/// </code>
/// </example>
public sealed class ModelPredictiveController<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Matrix<T> _stateMatrix;
    private readonly Matrix<T> _inputMatrix;
    private readonly ModelPredictiveControllerOptions<T> _options;
    private readonly IQuadraticProgramSolver<T> _solver;

    private readonly int _stateCount;
    private readonly int _inputCount;
    private readonly int _horizon;

    // X = _prediction * x0 + _response * U, the condensed dynamics over the horizon.
    private readonly Matrix<T> _prediction;
    private readonly Matrix<T> _response;

    // Fixed parts of the quadratic program: H = 2(GammaᵀQ̄Gamma + R̄), and GammaᵀQ̄Phi for the
    // linear term. Neither depends on the current state, so both are built once.
    private readonly Matrix<T> _hessian;
    private readonly Matrix<T> _linearFactor;

    private readonly Vector<T>? _inputLower;
    private readonly Vector<T>? _inputUpper;

    /// <summary>
    /// Creates a model predictive controller and precomputes everything that does not depend on the
    /// current state.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>, <c>n</c>-by-<c>n</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>, <c>n</c>-by-<c>m</c>.</param>
    /// <param name="stateCost">The state cost <c>Q</c>, symmetric positive-semidefinite.</param>
    /// <param name="inputCost">The input cost <c>R</c>, symmetric positive-definite.</param>
    /// <param name="options">Horizon, constraints and terminal cost, or <c>null</c> for defaults.</param>
    /// <param name="solver">
    /// The quadratic program solver used each step. Defaults to
    /// <see cref="InteriorPointSolver{T}"/>, whose cost grows mildly with the horizon; an
    /// <see cref="ActiveSetQuadraticProgramSolver{T}"/> is often faster for a short horizon with few
    /// active constraints.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when a required matrix is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the dimensions are inconsistent, the horizon is not positive, or a bound vector's
    /// length does not match.
    /// </exception>
    public ModelPredictiveController(
        Matrix<T> stateMatrix,
        Matrix<T> inputMatrix,
        Matrix<T> stateCost,
        Matrix<T> inputCost,
        ModelPredictiveControllerOptions<T>? options = null,
        IQuadraticProgramSolver<T>? solver = null)
    {
        RiccatiValidation<T>.Validate(stateMatrix, inputMatrix, stateCost, inputCost);

        _options = options is null
            ? new ModelPredictiveControllerOptions<T>()
            : new ModelPredictiveControllerOptions<T>(options);
        _solver = solver ?? new InteriorPointSolver<T>();

        if (_options.Horizon <= 0)
        {
            throw new ArgumentException(
                "The horizon must be at least one step; a controller that plans nothing has nothing " +
                "to apply.", nameof(options));
        }

        _stateMatrix = stateMatrix;
        _inputMatrix = inputMatrix;
        _stateCount = stateMatrix.Rows;
        _inputCount = inputMatrix.Columns;
        _horizon = _options.Horizon;

        ValidateBounds(_options.InputLowerBounds, _inputCount, "InputLowerBounds");
        ValidateBounds(_options.InputUpperBounds, _inputCount, "InputUpperBounds");
        ValidateBounds(_options.StateLowerBounds, _stateCount, "StateLowerBounds");
        ValidateBounds(_options.StateUpperBounds, _stateCount, "StateUpperBounds");

        _inputLower = _options.InputLowerBounds;
        _inputUpper = _options.InputUpperBounds;

        // The terminal cost stands in for everything past the horizon. The Riccati solution makes
        // that stand-in exact for the unconstrained problem, which is what makes this controller
        // reproduce LQR when no constraint binds.
        TerminalCost = _options.TerminalCost
            ?? new DiscreteAlgebraicRiccatiSolver<T>()
                .Solve(stateMatrix, inputMatrix, stateCost, inputCost).Solution;

        if (TerminalCost.Rows != _stateCount || TerminalCost.Columns != _stateCount)
        {
            throw new ArgumentException(
                $"The terminal cost must be {_stateCount}-by-{_stateCount}; it is " +
                $"{TerminalCost.Rows}-by-{TerminalCost.Columns}.", nameof(options));
        }

        (_prediction, _response) = BuildCondensedDynamics();
        (_hessian, _linearFactor) = BuildQuadraticForm(stateCost, inputCost);
    }

    /// <summary>Gets the prediction horizon.</summary>
    public int Horizon => _horizon;

    /// <summary>Gets the terminal cost matrix in use.</summary>
    public Matrix<T> TerminalCost { get; }

    /// <summary>
    /// Gets the status of the most recent solve.
    /// </summary>
    /// <remarks>
    /// <para>
    /// State constraints can make a step genuinely infeasible — there may be no input sequence that
    /// keeps a state inside its limit from where the system now is. When that happens the status says
    /// so rather than the controller silently returning something arbitrary, because a caller running
    /// real equipment needs to know the difference between a plan and a guess.
    /// </para>
    /// </remarks>
    public LinearProgramStatus LastStatus { get; private set; } = LinearProgramStatus.Optimal;

    /// <summary>
    /// Computes the input to apply now, by planning the whole horizon and keeping the first move.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>The first input of the optimal plan.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the state's length does not match the system.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the step is infeasible, which means no input sequence satisfies the state
    /// constraints from here.
    /// </exception>
    public Vector<T> ComputeControl(Vector<T> state)
    {
        var plan = ComputePlan(state);

        var first = new Vector<T>(_inputCount);
        for (int i = 0; i < _inputCount; i++) first[i] = plan[0, i];

        return first;
    }

    /// <summary>
    /// Computes the whole planned input sequence over the horizon.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>
    /// A <c>horizon</c>-by-<c>inputs</c> matrix whose row <c>k</c> is the input planned for step
    /// <c>k</c>.
    /// </returns>
    /// <remarks>
    /// <para>
    /// Only the first row is meant to be applied. The rest is exposed because it is genuinely useful
    /// to look at — a plan that saturates for ten steps and then reverses hard tells you something
    /// about your horizon or your cost weights that the applied input alone never would.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the state's length does not match the system.
    /// </exception>
    /// <exception cref="InvalidOperationException">Thrown when the step is infeasible.</exception>
    public Matrix<T> ComputePlan(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (state.Length != _stateCount)
        {
            throw new ArgumentException(
                $"The state must have {_stateCount} entries to match the system; it has " +
                $"{state.Length}.", nameof(state));
        }

        var program = BuildProgram(state);
        var solution = _solver.Solve(program);

        LastStatus = solution.Status;

        if (solution.Solution is null)
        {
            throw new InvalidOperationException(
                $"The predictive control step could not be solved (status {solution.Status}). With " +
                "state constraints this usually means no input sequence keeps the state inside its " +
                "limits from here — either the horizon is too short to steer clear in time, or the " +
                "system has already left the region it can recover from.");
        }

        var plan = new Matrix<T>(_horizon, _inputCount);
        for (int k = 0; k < _horizon; k++)
        {
            for (int i = 0; i < _inputCount; i++)
            {
                plan[k, i] = solution.Solution[k * _inputCount + i];
            }
        }

        return plan;
    }

    /// <summary>
    /// Builds the condensed dynamics <c>X = Φ·x₀ + Γ·U</c> over the horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Row block <c>k</c> of <c>Φ</c> is <c>A^(k+1)</c>, and block <c>(k, j)</c> of <c>Γ</c> is
    /// <c>A^(k-j)·B</c> for <c>j ≤ k</c> and zero above the diagonal — the system's impulse response,
    /// arranged so that a future input cannot affect a past state.
    /// </para>
    /// </remarks>
    private (Matrix<T> Prediction, Matrix<T> Response) BuildCondensedDynamics()
    {
        int rows = _horizon * _stateCount;

        var prediction = new Matrix<T>(rows, _stateCount);
        var response = new Matrix<T>(rows, _horizon * _inputCount);

        // powers[k] = A^k, accumulated rather than recomputed.
        var powers = new Matrix<T>[_horizon + 1];
        powers[0] = Matrix<T>.CreateIdentity(_stateCount);
        for (int k = 1; k <= _horizon; k++)
        {
            powers[k] = ControlMath<T>.Multiply(powers[k - 1], _stateMatrix);
        }

        for (int k = 0; k < _horizon; k++)
        {
            var power = powers[k + 1];
            for (int r = 0; r < _stateCount; r++)
            {
                for (int c = 0; c < _stateCount; c++)
                {
                    prediction[k * _stateCount + r, c] = power[r, c];
                }
            }

            for (int j = 0; j <= k; j++)
            {
                var block = ControlMath<T>.Multiply(powers[k - j], _inputMatrix);
                for (int r = 0; r < _stateCount; r++)
                {
                    for (int c = 0; c < _inputCount; c++)
                    {
                        response[k * _stateCount + r, j * _inputCount + c] = block[r, c];
                    }
                }
            }
        }

        return (prediction, response);
    }

    /// <summary>
    /// Builds the state-independent parts of the quadratic program.
    /// </summary>
    private (Matrix<T> Hessian, Matrix<T> LinearFactor) BuildQuadraticForm(
        Matrix<T> stateCost, Matrix<T> inputCost)
    {
        int stateRows = _horizon * _stateCount;
        // Q̄ = blockdiag(Q, ..., Q, P_f): the terminal cost replaces Q in the final block.
        var stackedStateCost = new Matrix<T>(stateRows, stateRows);
        for (int k = 0; k < _horizon; k++)
        {
            var block = k == _horizon - 1 ? TerminalCost : stateCost;
            for (int r = 0; r < _stateCount; r++)
            {
                for (int c = 0; c < _stateCount; c++)
                {
                    stackedStateCost[k * _stateCount + r, k * _stateCount + c] = block[r, c];
                }
            }
        }

        var responseTransposed = ControlMath<T>.Transpose(_response);
        var weightedResponse = ControlMath<T>.Multiply(responseTransposed, stackedStateCost);

        var hessian = ControlMath<T>.Multiply(weightedResponse, _response);

        // R̄ = blockdiag(R, ..., R) added along the diagonal.
        for (int k = 0; k < _horizon; k++)
        {
            for (int r = 0; r < _inputCount; r++)
            {
                for (int c = 0; c < _inputCount; c++)
                {
                    int row = k * _inputCount + r;
                    int column = k * _inputCount + c;
                    hessian[row, column] = NumOps.Add(hessian[row, column], inputCost[r, c]);
                }
            }
        }

        // The quadratic program is stated as 0.5*UᵀHU + fᵀU, so the doubling here cancels the half.
        hessian = ControlMath<T>.Symmetrize(ControlMath<T>.Scale(hessian, 2.0));

        var linearFactor = ControlMath<T>.Scale(
            ControlMath<T>.Multiply(weightedResponse, _prediction), 2.0);

        return (hessian, linearFactor);
    }

    /// <summary>
    /// Assembles this step's quadratic program from the current state.
    /// </summary>
    private QuadraticProgram<T> BuildProgram(Vector<T> state)
    {
        int variableCount = _horizon * _inputCount;

        var linear = ControlMath<T>.Multiply(_linearFactor, state);

        // A null quadratic-program bound means unbounded. Preserve that representation instead of
        // manufacturing IEEE infinities, because generic numeric types such as decimal cannot
        // represent them.
        Vector<T>? lowerBounds = null;
        Vector<T>? upperBounds = null;

        if (_inputLower is not null)
        {
            lowerBounds = new Vector<T>(variableCount);
            for (int k = 0; k < _horizon; k++)
            {
                for (int i = 0; i < _inputCount; i++)
                {
                    lowerBounds[k * _inputCount + i] = _inputLower[i];
                }
            }
        }

        if (_inputUpper is not null)
        {
            upperBounds = new Vector<T>(variableCount);
            for (int k = 0; k < _horizon; k++)
            {
                for (int i = 0; i < _inputCount; i++)
                {
                    upperBounds[k * _inputCount + i] = _inputUpper[i];
                }
            }
        }

        var (inequalityMatrix, inequalityBounds) = BuildStateConstraints(state);

        return new QuadraticProgram<T>(
            quadratic: _hessian,
            linear: linear,
            inequalityMatrix: inequalityMatrix,
            inequalityBounds: inequalityBounds,
            lowerBounds: lowerBounds,
            upperBounds: upperBounds);
    }

    /// <summary>
    /// Turns the state bounds into inequality rows over the input sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Since <c>X = Φx₀ + ΓU</c>, an upper bound becomes <c>ΓU ≤ x_max − Φx₀</c> and a lower bound
    /// becomes <c>−ΓU ≤ Φx₀ − x_min</c>. Only entries with a finite bound produce a row, so
    /// constraining one state does not require inventing limits for the others.
    /// </para>
    /// </remarks>
    private (Matrix<T>? Matrix, Vector<T>? Bounds) BuildStateConstraints(Vector<T> state)
    {
        var lower = _options.StateLowerBounds;
        var upper = _options.StateUpperBounds;

        if (lower is null && upper is null) return (null, null);

        var predicted = ControlMath<T>.Multiply(_prediction, state);

        var rows = new List<Vector<T>>();
        var bounds = new List<T>();
        int columnCount = _horizon * _inputCount;

        for (int k = 0; k < _horizon; k++)
        {
            for (int i = 0; i < _stateCount; i++)
            {
                int row = k * _stateCount + i;

                if (upper is not null && IsFinite(upper[i]))
                {
                    var coefficients = new Vector<T>(columnCount);
                    for (int c = 0; c < columnCount; c++) coefficients[c] = _response[row, c];

                    rows.Add(coefficients);
                    bounds.Add(NumOps.Subtract(upper[i], predicted[row]));
                }

                if (lower is not null && IsFinite(lower[i]))
                {
                    var coefficients = new Vector<T>(columnCount);
                    for (int c = 0; c < columnCount; c++)
                    {
                        coefficients[c] = NumOps.Negate(_response[row, c]);
                    }

                    rows.Add(coefficients);
                    bounds.Add(NumOps.Subtract(predicted[row], lower[i]));
                }
            }
        }

        if (rows.Count == 0) return (null, null);

        var matrix = new Matrix<T>(rows.Count, columnCount);
        var vector = new Vector<T>(rows.Count);

        for (int r = 0; r < rows.Count; r++)
        {
            for (int c = 0; c < columnCount; c++) matrix[r, c] = rows[r][c];
            vector[r] = bounds[r];
        }

        return (matrix, vector);
    }

    private static bool IsFinite(T value)
    {
        if (NumOps.Equals(value, NumOps.MinValue) || NumOps.Equals(value, NumOps.MaxValue))
        {
            return false;
        }

        double asDouble = NumOps.ToDouble(value);
        return !double.IsInfinity(asDouble) && !double.IsNaN(asDouble);
    }

    private static void ValidateBounds(Vector<T>? bounds, int expected, string name)
    {
        if (bounds is null) return;
        if (bounds.Length != expected)
        {
            throw new ArgumentException(
                $"{name} must have {expected} entries; it has {bounds.Length}.", name);
        }
    }
}
