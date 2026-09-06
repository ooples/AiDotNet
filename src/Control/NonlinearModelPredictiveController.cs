using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Solvers.InteriorPoint;
using AiDotNet.Solvers.LinearProgramming;
using AiDotNet.Solvers.QuadraticProgramming;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Nonlinear model predictive control by sequential quadratic programming: repeatedly linearize the
/// plan you have, improve it with a quadratic program, and relinearize about the result.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Handles systems of the form <c>x[k+1] = f(x[k], u[k])</c> where <c>f</c> is any differentiable
/// function. Follows the sequential quadratic programming approach to nonlinear predictive control;
/// see M. Diehl, H. G. Bock, J. P. Schlöder, R. Findeisen, Z. Nagy and F. Allgöwer, "Real-time
/// optimization and nonlinear model predictive control of processes governed by differential-
/// algebraic equations", <i>Journal of Process Control</i> 12(4), 2002, pp. 577-585, and Chapter 8 of
/// Rawlings, Mayne and Diehl.
/// </para>
/// <para>
/// <b>How it works.</b> A nonlinear predictive control step is not a quadratic program and cannot be
/// solved as one. But a <i>correction</i> to an existing plan very nearly is: simulate the current
/// plan through the true nonlinear dynamics, linearize about the trajectory that produces, and the
/// question "how should I adjust this plan" becomes exactly the linear problem the previous class
/// solves. Apply the correction, simulate again, repeat. Each pass costs one quadratic program and
/// the sequence converges to the nonlinear optimum.
/// </para>
/// <para>
/// <b>Why the linearization must be time-varying.</b> The trajectory visits a different point of the
/// state space at every step, and the system's behaviour there is different — that is what nonlinear
/// means. So there is a distinct <c>Aₖ</c> and <c>Bₖ</c> for each step of the horizon, and the
/// condensed dynamics accumulate products of them rather than powers of a single matrix. This is the
/// one structural difference from <see cref="ModelPredictiveController{T}"/>, and it is why that
/// class cannot simply be called in a loop.
/// </para>
/// <para>
/// <b>Warm starting matters more here than anywhere else.</b> Consecutive control steps solve nearly
/// the same problem, so the previous plan shifted forward by one step is an excellent guess — often
/// good enough that a single sequential iteration suffices. That observation is the basis of the
/// real-time iteration scheme, and it is what makes nonlinear predictive control possible at
/// millisecond rates on real hardware.
/// </para>
/// <para><b>For Beginners:</b> Straight-line thinking works fine over short distances even on a
/// curved road. This plans as if the system were linear, checks what the real system would actually
/// have done, corrects for the difference, and repeats until the plan stops changing — then applies
/// only its first move, and does the whole thing again next step.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // A pendulum: state is (angle, rate), input is torque. The costs are weighting MATRICES, not
/// // scalars — identity here means every state and input is penalised equally.
/// var stateCost = Matrix&lt;double&gt;.CreateIdentity(2);
/// var inputCost = Matrix&lt;double&gt;.CreateIdentity(1);
/// var currentState = new Vector&lt;double&gt;(new double[] { 0.1, 0.0 });
///
/// var controller = new NonlinearModelPredictiveController&lt;double&gt;(
///     dynamics: (x, u) =&gt; x,                       // your discrete-time step, x_{k+1} = f(x_k, u_k)
///     jacobians: (x, u) =&gt; (Matrix&lt;double&gt;.CreateIdentity(2), Matrix&lt;double&gt;.CreateIdentity(2)),
///     stateCost: stateCost, inputCost: inputCost,
///     options: new NonlinearModelPredictiveControllerOptions&lt;double&gt; { Horizon = 20 });
///
/// var torque = controller.ComputeControl(currentState);
/// </code>
/// </example>
public sealed class NonlinearModelPredictiveController<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>, Vector<T>> _dynamics;
    private readonly Func<Vector<T>, Vector<T>, (Matrix<T> StateJacobian, Matrix<T> InputJacobian)>
        _jacobians;

    private readonly Matrix<T> _stateCost;
    private readonly Matrix<T> _inputCost;
    private readonly Matrix<T> _terminalCost;
    private readonly NonlinearModelPredictiveControllerOptions<T> _options;
    private readonly IQuadraticProgramSolver<T> _solver;

    private readonly int _stateCount;
    private readonly int _inputCount;
    private readonly int _horizon;

    private Vector<T>[]? _previousPlan;

    /// <summary>
    /// Creates a nonlinear model predictive controller.
    /// </summary>
    /// <param name="dynamics">
    /// The one-step dynamics <c>f(x, u)</c> returning the next state.
    /// </param>
    /// <param name="jacobians">
    /// The partial derivatives of <c>f</c> at a point: <c>∂f/∂x</c> and <c>∂f/∂u</c>. Supplied
    /// together because both are needed at the same point on every linearization, and computing them
    /// in one pass lets a caller share the expensive intermediate work.
    /// </param>
    /// <param name="stateCost">The state cost <c>Q</c>.</param>
    /// <param name="inputCost">The input cost <c>R</c>, which must be positive definite.</param>
    /// <param name="options">Horizon, iteration count and bounds, or <c>null</c> for defaults.</param>
    /// <param name="solver">
    /// The quadratic program solver used for each correction, defaulting to interior point.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when a required argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shapes or settings are inconsistent.</exception>
    public NonlinearModelPredictiveController(
        Func<Vector<T>, Vector<T>, Vector<T>> dynamics,
        Func<Vector<T>, Vector<T>, (Matrix<T> StateJacobian, Matrix<T> InputJacobian)> jacobians,
        Matrix<T> stateCost,
        Matrix<T> inputCost,
        NonlinearModelPredictiveControllerOptions<T>? options = null,
        IQuadraticProgramSolver<T>? solver = null)
    {
        _dynamics = dynamics ?? throw new ArgumentNullException(nameof(dynamics));
        _jacobians = jacobians ?? throw new ArgumentNullException(nameof(jacobians));
        _stateCost = stateCost ?? throw new ArgumentNullException(nameof(stateCost));
        _inputCost = inputCost ?? throw new ArgumentNullException(nameof(inputCost));

        _options = options is null
            ? new NonlinearModelPredictiveControllerOptions<T>()
            : new NonlinearModelPredictiveControllerOptions<T>(options);
        _solver = solver ?? new InteriorPointSolver<T>();

        if (_stateCost.Rows != _stateCost.Columns)
        {
            throw new ArgumentException("The state cost Q must be square.", nameof(stateCost));
        }

        if (_inputCost.Rows != _inputCost.Columns)
        {
            throw new ArgumentException("The input cost R must be square.", nameof(inputCost));
        }

        if (_options.Horizon <= 0)
        {
            throw new ArgumentException(
                "The horizon must be at least one step.", nameof(options));
        }

        if (_options.SqpIterations <= 0)
        {
            throw new ArgumentException(
                "At least one sequential quadratic programming iteration is required; zero would " +
                "return the initial guess unchanged.", nameof(options));
        }

        if (_options.StepSize <= 0.0 || _options.StepSize > 1.0)
        {
            throw new ArgumentException(
                "The step size must lie in (0, 1]. Above one the update overshoots the correction " +
                "the linearization actually justifies.", nameof(options));
        }

        _stateCount = _stateCost.Rows;
        _inputCount = _inputCost.Rows;
        _horizon = _options.Horizon;

        _terminalCost = _options.TerminalCost ?? _stateCost;

        if (_terminalCost.Rows != _stateCount || _terminalCost.Columns != _stateCount)
        {
            throw new ArgumentException(
                $"The terminal cost must be {_stateCount}-by-{_stateCount}; it is " +
                $"{_terminalCost.Rows}-by-{_terminalCost.Columns}.", nameof(options));
        }

        if (_options.InputLowerBounds is not null &&
            _options.InputLowerBounds.Length != _inputCount)
        {
            throw new ArgumentException(
                $"InputLowerBounds must have {_inputCount} entries.", nameof(options));
        }

        if (_options.InputUpperBounds is not null &&
            _options.InputUpperBounds.Length != _inputCount)
        {
            throw new ArgumentException(
                $"InputUpperBounds must have {_inputCount} entries.", nameof(options));
        }
    }

    /// <summary>Gets the prediction horizon.</summary>
    public int Horizon => _horizon;

    /// <summary>
    /// Gets the number of sequential iterations the most recent step actually used.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Fewer than the configured maximum means the plan stopped changing and the iteration converged.
    /// Consistently hitting the maximum means it did not, which on a strongly nonlinear system is a
    /// reason to reduce the step size rather than to raise the iteration count.
    /// </para>
    /// </remarks>
    public int LastIterationCount { get; private set; }

    /// <summary>
    /// Computes the input to apply now.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>The first input of the refined plan.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the state's length does not match the cost matrices.
    /// </exception>
    /// <exception cref="InvalidOperationException">Thrown when a correction step cannot be solved.</exception>
    public Vector<T> ComputeControl(Vector<T> state)
    {
        var plan = ComputePlan(state);

        var first = new Vector<T>(_inputCount);
        for (int i = 0; i < _inputCount; i++) first[i] = plan[0, i];

        return first;
    }

    /// <summary>
    /// Computes the whole refined input sequence over the horizon.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <returns>
    /// A <c>horizon</c>-by-<c>inputs</c> matrix whose row <c>k</c> is the input planned for step
    /// <c>k</c>.
    /// </returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the state's length does not match the cost matrices.
    /// </exception>
    /// <exception cref="InvalidOperationException">Thrown when a correction step cannot be solved.</exception>
    public Matrix<T> ComputePlan(Vector<T> state)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (state.Length != _stateCount)
        {
            throw new ArgumentException(
                $"The state must have {_stateCount} entries to match the cost matrices; it has " +
                $"{state.Length}.", nameof(state));
        }

        var inputs = BuildInitialPlan();

        int iteration;
        for (iteration = 1; iteration <= _options.SqpIterations; iteration++)
        {
            // Roll the current plan through the TRUE dynamics, so the linearization is taken about
            // where the system would actually go rather than where a previous linear model predicted.
            var trajectory = Simulate(state, inputs);

            var correction = SolveCorrection(inputs, trajectory);

            double change = ApplyCorrection(inputs, correction);

            if (change <= _options.Tolerance) break;
        }

        LastIterationCount = Math.Min(iteration, _options.SqpIterations);

        if (_options.WarmStart) _previousPlan = inputs;

        var plan = new Matrix<T>(_horizon, _inputCount);
        for (int k = 0; k < _horizon; k++)
        {
            for (int i = 0; i < _inputCount; i++) plan[k, i] = inputs[k][i];
        }

        return plan;
    }

    /// <summary>
    /// Produces the plan the iteration starts from.
    /// </summary>
    /// <remarks>
    /// <para>
    /// With warm starting, the previous step's plan shifted forward by one — the move that was
    /// applied is dropped, and the last entry is repeated to fill the gap. Without it, all zeros.
    /// </para>
    /// </remarks>
    private Vector<T>[] BuildInitialPlan()
    {
        var inputs = new Vector<T>[_horizon];

        if (_options.WarmStart && _previousPlan is not null)
        {
            for (int k = 0; k < _horizon; k++)
            {
                int source = Math.Min(k + 1, _horizon - 1);
                var copy = new Vector<T>(_inputCount);
                for (int i = 0; i < _inputCount; i++) copy[i] = _previousPlan[source][i];
                inputs[k] = copy;
            }

            return inputs;
        }

        for (int k = 0; k < _horizon; k++) inputs[k] = new Vector<T>(_inputCount);
        return inputs;
    }

    /// <summary>
    /// Rolls an input sequence through the nonlinear dynamics, returning the states it visits.
    /// </summary>
    private Vector<T>[] Simulate(Vector<T> state, Vector<T>[] inputs)
    {
        var trajectory = new Vector<T>[_horizon + 1];
        trajectory[0] = state;

        for (int k = 0; k < _horizon; k++)
        {
            var next = _dynamics(trajectory[k], inputs[k]);

            if (next is null || next.Length != _stateCount)
            {
                throw new InvalidOperationException(
                    $"The dynamics function must return a vector of {_stateCount} entries; it " +
                    $"returned {(next is null ? "null" : next.Length.ToString())} at step {k}.");
            }

            trajectory[k + 1] = next;
        }

        return trajectory;
    }

    /// <summary>
    /// Builds and solves the quadratic program for a correction to the current plan.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In deviation coordinates the dynamics are <c>δx[k+1] = Aₖ·δx[k] + Bₖ·δu[k]</c> with
    /// <c>δx[0] = 0</c>, since the current state is known exactly and no correction can change it.
    /// Condensing gives <c>δX = Γ·δU</c>, and the cost
    /// <c>Σ (x̄ₖ + δxₖ)ᵀQ(x̄ₖ + δxₖ) + Σ (ūₖ + δuₖ)ᵀR(ūₖ + δuₖ)</c> expands to a quadratic program in
    /// <c>δU</c> whose linear term carries the nominal trajectory — which is what makes the
    /// correction point toward the origin rather than merely toward the nominal plan.
    /// </para>
    /// </remarks>
    private Vector<T> SolveCorrection(Vector<T>[] inputs, Vector<T>[] trajectory)
    {
        var stateJacobians = new Matrix<T>[_horizon];
        var inputJacobians = new Matrix<T>[_horizon];

        for (int k = 0; k < _horizon; k++)
        {
            var (stateJacobian, inputJacobian) = _jacobians(trajectory[k], inputs[k]);

            if (stateJacobian is null || inputJacobian is null)
            {
                throw new InvalidOperationException(
                    $"The Jacobian function returned null at step {k}.");
            }

            if (stateJacobian.Rows != _stateCount || stateJacobian.Columns != _stateCount)
            {
                throw new InvalidOperationException(
                    $"The state Jacobian must be {_stateCount}-by-{_stateCount}; at step {k} it " +
                    $"was {stateJacobian.Rows}-by-{stateJacobian.Columns}.");
            }

            if (inputJacobian.Rows != _stateCount || inputJacobian.Columns != _inputCount)
            {
                throw new InvalidOperationException(
                    $"The input Jacobian must be {_stateCount}-by-{_inputCount}; at step {k} it " +
                    $"was {inputJacobian.Rows}-by-{inputJacobian.Columns}.");
            }

            stateJacobians[k] = stateJacobian;
            inputJacobians[k] = inputJacobian;
        }

        var response = BuildTimeVaryingResponse(stateJacobians, inputJacobians);

        int variableCount = _horizon * _inputCount;
        int stateRows = _horizon * _stateCount;

        // Q̄ over the predicted states, with the terminal cost in the final block.
        var stackedStateCost = new Matrix<T>(stateRows, stateRows);
        for (int k = 0; k < _horizon; k++)
        {
            var block = k == _horizon - 1 ? _terminalCost : _stateCost;
            for (int r = 0; r < _stateCount; r++)
            {
                for (int c = 0; c < _stateCount; c++)
                {
                    stackedStateCost[k * _stateCount + r, k * _stateCount + c] = block[r, c];
                }
            }
        }

        var responseTransposed = ControlMath<T>.Transpose(response);
        var weightedResponse = ControlMath<T>.Multiply(responseTransposed, stackedStateCost);

        var hessian = ControlMath<T>.Multiply(weightedResponse, response);
        for (int k = 0; k < _horizon; k++)
        {
            for (int r = 0; r < _inputCount; r++)
            {
                for (int c = 0; c < _inputCount; c++)
                {
                    int row = k * _inputCount + r;
                    int column = k * _inputCount + c;
                    hessian[row, column] = NumOps.Add(hessian[row, column], _inputCost[r, c]);
                }
            }
        }

        hessian = ControlMath<T>.Symmetrize(ControlMath<T>.Scale(hessian, 2.0));

        // Linear term: 2*(Gamma' Q̄ X̄ + R̄ Ū), the pull toward the origin from where the nominal
        // plan currently sits.
        var stackedStates = new Vector<T>(stateRows);
        for (int k = 0; k < _horizon; k++)
        {
            for (int i = 0; i < _stateCount; i++)
            {
                stackedStates[k * _stateCount + i] = trajectory[k + 1][i];
            }
        }

        var linear = ControlMath<T>.Scale(
            ControlMath<T>.Multiply(weightedResponse, stackedStates), 2.0);

        for (int k = 0; k < _horizon; k++)
        {
            for (int r = 0; r < _inputCount; r++)
            {
                T accumulator = NumOps.Zero;
                for (int c = 0; c < _inputCount; c++)
                {
                    accumulator = NumOps.Add(
                        accumulator, NumOps.Multiply(_inputCost[r, c], inputs[k][c]));
                }

                int index = k * _inputCount + r;
                linear[index] = NumOps.Add(
                    linear[index], NumOps.Multiply(NumOps.FromDouble(2.0), accumulator));
            }
        }

        // Bounds move with the nominal plan: the correction must land the total input in range.
        Vector<T>? lowerBounds = _options.InputLowerBounds is null
            ? null
            : new Vector<T>(variableCount);
        Vector<T>? upperBounds = _options.InputUpperBounds is null
            ? null
            : new Vector<T>(variableCount);

        for (int k = 0; k < _horizon; k++)
        {
            for (int i = 0; i < _inputCount; i++)
            {
                int index = k * _inputCount + i;

                if (lowerBounds is not null)
                {
                    lowerBounds[index] = NumOps.Subtract(
                        _options.InputLowerBounds![i], inputs[k][i]);
                }

                if (upperBounds is not null)
                {
                    upperBounds[index] = NumOps.Subtract(
                        _options.InputUpperBounds![i], inputs[k][i]);
                }
            }
        }

        var program = new QuadraticProgram<T>(
            quadratic: hessian,
            linear: linear,
            lowerBounds: lowerBounds,
            upperBounds: upperBounds);

        var solution = _solver.Solve(program);

        if (solution.Solution is null)
        {
            throw new InvalidOperationException(
                $"A nonlinear predictive control correction could not be solved (status " +
                $"{solution.Status}). Check that R is positive definite and that the input bounds " +
                "are not contradictory.");
        }

        return solution.Solution;
    }

    /// <summary>
    /// Builds the condensed response <c>Γ</c> for a time-varying linearization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Block <c>(k, j)</c> is <c>A_{k}·A_{k-1}···A_{j+1}·B_j</c> for <c>j ≤ k</c> — how the input at
    /// step <c>j</c> reaches the state at step <c>k+1</c> after being carried through every
    /// intervening linearization. The accumulation runs backwards from each row so that each product
    /// is formed once rather than rebuilt per block.
    /// </para>
    /// </remarks>
    private Matrix<T> BuildTimeVaryingResponse(
        Matrix<T>[] stateJacobians, Matrix<T>[] inputJacobians)
    {
        int rows = _horizon * _stateCount;
        var response = new Matrix<T>(rows, _horizon * _inputCount);

        for (int k = 0; k < _horizon; k++)
        {
            // carry = product of A over (j, k]; starts at identity for j == k.
            var carry = Matrix<T>.CreateIdentity(_stateCount);

            for (int j = k; j >= 0; j--)
            {
                var block = ControlMath<T>.Multiply(carry, inputJacobians[j]);

                for (int r = 0; r < _stateCount; r++)
                {
                    for (int c = 0; c < _inputCount; c++)
                    {
                        response[k * _stateCount + r, j * _inputCount + c] = block[r, c];
                    }
                }

                if (j > 0) carry = ControlMath<T>.Multiply(carry, stateJacobians[j]);
            }
        }

        return response;
    }

    /// <summary>
    /// Applies a damped correction to the plan and reports how far it moved.
    /// </summary>
    private double ApplyCorrection(Vector<T>[] inputs, Vector<T> correction)
    {
        T step = NumOps.FromDouble(_options.StepSize);
        double movement = 0.0;

        for (int k = 0; k < _horizon; k++)
        {
            for (int i = 0; i < _inputCount; i++)
            {
                T delta = NumOps.Multiply(step, correction[k * _inputCount + i]);
                inputs[k][i] = NumOps.Add(inputs[k][i], delta);

                double magnitude = NumOps.ToDouble(delta);
                movement += magnitude * magnitude;
            }
        }

        return Math.Sqrt(movement);
    }
}
