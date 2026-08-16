using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// The recursive Kalman filter: the optimal state estimator for a linear system with Gaussian noise.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Implements the filter of R. E. Kalman, "A New Approach to Linear Filtering and Prediction
/// Problems", <i>Transactions of the ASME — Journal of Basic Engineering</i> 82(D), 1960, pp. 35-45,
/// for the system <c>x[k+1] = A·x[k] + B·u[k] + w[k]</c>, <c>y[k] = C·x[k] + v[k]</c>, with process
/// noise covariance <c>Q</c> and measurement noise covariance <c>R</c>.
/// </para>
/// <para>
/// <b>Why an estimator is needed at all.</b> A regulator such as
/// <see cref="LinearQuadraticRegulator{T}"/> assumes the whole state is known. Almost never is it:
/// sensors measure some combination of the state, corrupted by noise, and often fewer numbers than
/// there are states. The filter reconstructs the rest. It is the exact mirror image of the
/// regulator — the equation it solves is the regulator's Riccati equation with the matrices
/// transposed — which is the duality Kalman pointed out and the reason
/// <see cref="SteadyStateGain"/> can be computed by the same solver.
/// </para>
/// <para>
/// <b>Predict and update.</b> Each step has two halves. <see cref="Predict()"/> pushes the estimate
/// forward through the dynamics, which always increases uncertainty — the model is not perfect.
/// <see cref="Update"/> folds in a measurement, which always decreases it. The Kalman gain sets the
/// balance, and it is not a tuning parameter: it is derived, and it is the value that minimizes the
/// estimate's variance given <c>Q</c> and <c>R</c>.
/// </para>
/// <para>
/// <b>The covariance update is Joseph's form,</b> <c>(I−KC)P(I−KC)ᵀ + KRKᵀ</c>, rather than the
/// shorter <c>P − KCP</c>. The two are algebraically equal, but only at the exactly optimal gain,
/// and only in exact arithmetic. The short form is a difference of two positive matrices, so
/// rounding can make it asymmetric or even indefinite, and once a covariance goes indefinite the
/// filter diverges and never recovers. Joseph's form is a sum of two positive terms and stays
/// symmetric positive-semidefinite whatever the arithmetic does — at the cost of one extra matrix
/// product per step, which is the cheapest insurance in filtering.
/// </para>
/// <para><b>For Beginners:</b> You have a model of how something moves and some noisy measurements
/// of it. Neither is trustworthy alone. This blends them, weighting each by how much it deserves to
/// be trusted, and keeps track of how uncertain the blend is. It is what lets a GPS give a smooth
/// position from jumpy satellite fixes.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Track position and velocity from noisy position measurements alone.
/// var filter = new KalmanFilter&lt;double&gt;(
///     transition: new Matrix&lt;double&gt;(new[,] { { 1.0, 1.0 }, { 0.0, 1.0 } }),
///     observation: new Matrix&lt;double&gt;(new[,] { { 1.0, 0.0 } }),
///     processNoise: new Matrix&lt;double&gt;(new[,] { { 0.01, 0.0 }, { 0.0, 0.01 } }),
///     measurementNoise: new Matrix&lt;double&gt;(new[,] { { 1.0 } }));
///
/// filter.Initialize(new Vector&lt;double&gt;(new[] { 0.0, 0.0 }), Matrix&lt;double&gt;.CreateIdentity(2));
///
/// filter.Predict();
/// filter.Update(new Vector&lt;double&gt;(new[] { 1.02 }));
/// // filter.State now holds the best estimate of both position and velocity.
/// </code>
/// </example>
public sealed class KalmanFilter<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Matrix<T> _transition;
    private readonly Matrix<T> _observation;
    private readonly Matrix<T> _processNoise;
    private readonly Matrix<T> _measurementNoise;
    private readonly Matrix<T>? _control;

    /// <summary>
    /// Creates a Kalman filter.
    /// </summary>
    /// <param name="transition">The state transition matrix <c>A</c>, <c>n</c>-by-<c>n</c>.</param>
    /// <param name="observation">The observation matrix <c>C</c>, <c>p</c>-by-<c>n</c>.</param>
    /// <param name="processNoise">
    /// The process noise covariance <c>Q</c>, <c>n</c>-by-<c>n</c> — how much the model is wrong by.
    /// </param>
    /// <param name="measurementNoise">
    /// The measurement noise covariance <c>R</c>, <c>p</c>-by-<c>p</c> — how much the sensors are
    /// wrong by.
    /// </param>
    /// <param name="control">
    /// The control matrix <c>B</c>, <c>n</c>-by-<c>m</c>, or <c>null</c> when the system has no
    /// known input. Supplying it lets <see cref="Predict(Vector{T})"/> account for commands you
    /// issued, which is what distinguishes an estimator inside a control loop from one merely
    /// watching.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when a required matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    public KalmanFilter(
        Matrix<T> transition,
        Matrix<T> observation,
        Matrix<T> processNoise,
        Matrix<T> measurementNoise,
        Matrix<T>? control = null)
    {
        if (transition is null) throw new ArgumentNullException(nameof(transition));
        if (observation is null) throw new ArgumentNullException(nameof(observation));
        if (processNoise is null) throw new ArgumentNullException(nameof(processNoise));
        if (measurementNoise is null) throw new ArgumentNullException(nameof(measurementNoise));

        if (transition.Rows != transition.Columns)
        {
            throw new ArgumentException(
                $"The transition matrix A must be square; it is {transition.Rows}-by-" +
                $"{transition.Columns}.", nameof(transition));
        }

        int stateCount = transition.Rows;
        if (stateCount == 0)
        {
            throw new ArgumentException(
                "The system must have at least one state.", nameof(transition));
        }

        if (observation.Columns != stateCount)
        {
            throw new ArgumentException(
                $"The observation matrix C must have one column per state: expected {stateCount} " +
                $"columns, but it has {observation.Columns}.", nameof(observation));
        }

        int measurementCount = observation.Rows;
        if (measurementCount == 0)
        {
            throw new ArgumentException(
                "The system must produce at least one measurement; with none there is nothing to " +
                "filter.", nameof(observation));
        }

        if (processNoise.Rows != stateCount || processNoise.Columns != stateCount)
        {
            throw new ArgumentException(
                $"The process noise covariance Q must be {stateCount}-by-{stateCount}; it is " +
                $"{processNoise.Rows}-by-{processNoise.Columns}.", nameof(processNoise));
        }

        if (measurementNoise.Rows != measurementCount ||
            measurementNoise.Columns != measurementCount)
        {
            throw new ArgumentException(
                $"The measurement noise covariance R must be {measurementCount}-by-" +
                $"{measurementCount} to match the rows of C; it is {measurementNoise.Rows}-by-" +
                $"{measurementNoise.Columns}.", nameof(measurementNoise));
        }

        if (control is not null && control.Rows != stateCount)
        {
            throw new ArgumentException(
                $"The control matrix B must have one row per state: expected {stateCount} rows, " +
                $"but it has {control.Rows}.", nameof(control));
        }

        _transition = transition;
        _observation = observation;
        _processNoise = processNoise;
        _measurementNoise = measurementNoise;
        _control = control;

        StateCount = stateCount;
        MeasurementCount = measurementCount;

        State = new Vector<T>(stateCount);
        Covariance = Matrix<T>.CreateIdentity(stateCount);
        Gain = new Matrix<T>(stateCount, measurementCount);
        Innovation = new Vector<T>(measurementCount);
    }

    /// <summary>Gets the number of states.</summary>
    public int StateCount { get; }

    /// <summary>Gets the number of measurements per step.</summary>
    public int MeasurementCount { get; }

    /// <summary>Gets the current state estimate.</summary>
    public Vector<T> State { get; private set; }

    /// <summary>
    /// Gets the current estimate covariance, whose diagonal holds the variance of each state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the filter's own account of how much it should be trusted, and it is available before
    /// any measurement arrives — which is what makes a Kalman filter usable for deciding whether to
    /// act on an estimate at all, not merely for producing one.
    /// </para>
    /// </remarks>
    public Matrix<T> Covariance { get; private set; }

    /// <summary>
    /// Gets the Kalman gain from the most recent <see cref="Update"/>.
    /// </summary>
    public Matrix<T> Gain { get; private set; }

    /// <summary>
    /// Gets the innovation from the most recent <see cref="Update"/> — the difference between the
    /// measurement received and the one expected.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The innovation sequence is the standard diagnostic for a filter: when the model and the noise
    /// covariances are right it is white noise with zero mean. Innovations that are consistently
    /// biased or correlated over time mean the model is wrong, and no amount of tuning <c>Q</c> and
    /// <c>R</c> will fix that.
    /// </para>
    /// </remarks>
    public Vector<T> Innovation { get; private set; }

    /// <summary>
    /// Sets the initial state estimate and its covariance.
    /// </summary>
    /// <param name="state">The initial state estimate.</param>
    /// <param name="covariance">
    /// The initial covariance — how uncertain that estimate is. A large multiple of the identity
    /// says "no idea", which lets the first few measurements dominate.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when an argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions do not match the system.</exception>
    public void Initialize(Vector<T> state, Matrix<T> covariance)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (covariance is null) throw new ArgumentNullException(nameof(covariance));

        if (state.Length != StateCount)
        {
            throw new ArgumentException(
                $"The initial state must have {StateCount} entries; it has {state.Length}.",
                nameof(state));
        }

        if (covariance.Rows != StateCount || covariance.Columns != StateCount)
        {
            throw new ArgumentException(
                $"The initial covariance must be {StateCount}-by-{StateCount}; it is " +
                $"{covariance.Rows}-by-{covariance.Columns}.", nameof(covariance));
        }

        State = state;
        Covariance = covariance;
    }

    /// <summary>
    /// Advances the estimate through the dynamics with no control input.
    /// </summary>
    public void Predict()
    {
        State = ControlMath<T>.Multiply(_transition, State);
        PropagateCovariance();
    }

    /// <summary>
    /// Advances the estimate through the dynamics, accounting for a known control input.
    /// </summary>
    /// <param name="input">The input applied over this step.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the filter was built without a control matrix.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the input's length does not match the control matrix.
    /// </exception>
    public void Predict(Vector<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        if (_control is null)
        {
            throw new InvalidOperationException(
                "This filter was created without a control matrix B, so it cannot account for an " +
                "input. Supply B to the constructor, or call the parameterless Predict.");
        }

        if (input.Length != _control.Columns)
        {
            throw new ArgumentException(
                $"The input must have {_control.Columns} entries to match B; it has {input.Length}.",
                nameof(input));
        }

        State = ControlMath<T>.Add(
            ControlMath<T>.Multiply(_transition, State),
            ControlMath<T>.Multiply(_control, input));

        PropagateCovariance();
    }

    /// <summary>
    /// Folds a measurement into the estimate.
    /// </summary>
    /// <param name="measurement">The measurement received this step.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="measurement"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the measurement's length does not match the system.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the innovation covariance is singular, which means a measurement direction
    /// carries no noise and no uncertainty at all.
    /// </exception>
    public void Update(Vector<T> measurement)
    {
        if (measurement is null) throw new ArgumentNullException(nameof(measurement));
        if (measurement.Length != MeasurementCount)
        {
            throw new ArgumentException(
                $"The measurement must have {MeasurementCount} entries; it has " +
                $"{measurement.Length}.", nameof(measurement));
        }

        var observationTransposed = ControlMath<T>.Transpose(_observation);

        // S = C P Cᵀ + R, the covariance of the innovation.
        var innovationCovariance = ControlMath<T>.Add(
            ControlMath<T>.Multiply(
                ControlMath<T>.Multiply(_observation, Covariance), observationTransposed),
            _measurementNoise);

        var innovationInverse = ControlMath<T>.TryInvert(innovationCovariance)
            ?? throw new InvalidOperationException(
                "The innovation covariance C·P·Cᵀ + R is singular. That means some direction of " +
                "the measurement is claimed to be known exactly — check that R is positive " +
                "definite.");

        // K = P Cᵀ S⁻¹
        Gain = ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(Covariance, observationTransposed), innovationInverse);

        Innovation = ControlMath<T>.Subtract(
            measurement, ControlMath<T>.Multiply(_observation, State));

        State = ControlMath<T>.Add(State, ControlMath<T>.Multiply(Gain, Innovation));

        // Joseph form: (I − KC) P (I − KC)ᵀ + K R Kᵀ.
        var factor = ControlMath<T>.Subtract(
            Matrix<T>.CreateIdentity(StateCount),
            ControlMath<T>.Multiply(Gain, _observation));

        var propagated = ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(factor, Covariance), ControlMath<T>.Transpose(factor));

        var noiseTerm = ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(Gain, _measurementNoise), ControlMath<T>.Transpose(Gain));

        Covariance = ControlMath<T>.Symmetrize(ControlMath<T>.Add(propagated, noiseTerm));
    }

    /// <summary>
    /// Computes the steady-state Kalman gain, the value the recursive gain converges to.
    /// </summary>
    /// <param name="transition">The state transition matrix <c>A</c>.</param>
    /// <param name="observation">The observation matrix <c>C</c>.</param>
    /// <param name="processNoise">The process noise covariance <c>Q</c>.</param>
    /// <param name="measurementNoise">The measurement noise covariance <c>R</c>.</param>
    /// <param name="options">Riccati solver configuration, or <c>null</c> for the defaults.</param>
    /// <returns>The steady-state predictor gain <c>A·P·Cᵀ(C·P·Cᵀ + R)⁻¹</c>.</returns>
    /// <remarks>
    /// <para>
    /// For a time-invariant system the recursive gain settles to a constant within a few steps, and
    /// from then on the filter is doing identical arithmetic every step to recompute a number that
    /// has stopped changing. Using the steady-state gain directly removes the covariance recursion
    /// entirely, which is what makes a Kalman filter cheap enough for a fast embedded loop.
    /// </para>
    /// <para>
    /// It is obtained by <i>duality</i>: the filter's Riccati equation is the regulator's with
    /// <c>A</c> replaced by <c>Aᵀ</c> and <c>B</c> by <c>Cᵀ</c>, so the same solver produces both.
    /// This is Kalman's own observation, and it is why estimation and control are one theory rather
    /// than two.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    public static Matrix<T> SteadyStateGain(
        Matrix<T> transition,
        Matrix<T> observation,
        Matrix<T> processNoise,
        Matrix<T> measurementNoise,
        AlgebraicRiccatiSolverOptions? options = null)
    {
        if (transition is null) throw new ArgumentNullException(nameof(transition));
        if (observation is null) throw new ArgumentNullException(nameof(observation));

        var transitionTransposed = ControlMath<T>.Transpose(transition);
        var observationTransposed = ControlMath<T>.Transpose(observation);

        var solution = new DiscreteAlgebraicRiccatiSolver<T>(
                options ?? new AlgebraicRiccatiSolverOptions())
            .Solve(transitionTransposed, observationTransposed, processNoise, measurementNoise);

        var covariance = solution.Solution;

        var innovationCovariance = ControlMath<T>.Add(
            ControlMath<T>.Multiply(
                ControlMath<T>.Multiply(observation, covariance), observationTransposed),
            measurementNoise);

        var innovationInverse = ControlMath<T>.TryInvert(innovationCovariance)
            ?? throw new InvalidOperationException(
                "The steady-state innovation covariance is singular; check that R is positive " +
                "definite.");

        return ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(
                ControlMath<T>.Multiply(transition, covariance), observationTransposed),
            innovationInverse);
    }

    /// <summary>
    /// Propagates the covariance through the dynamics: <c>P ← A P Aᵀ + Q</c>.
    /// </summary>
    private void PropagateCovariance()
    {
        Covariance = ControlMath<T>.Symmetrize(
            ControlMath<T>.Add(
                ControlMath<T>.Multiply(
                    ControlMath<T>.Multiply(_transition, Covariance),
                    ControlMath<T>.Transpose(_transition)),
                _processNoise));
    }

}
