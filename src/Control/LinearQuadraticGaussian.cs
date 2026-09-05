using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// The linear quadratic Gaussian controller: an optimal regulator driven by an optimal estimate of a
/// state it cannot measure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Combines <see cref="LinearQuadraticRegulator{T}"/> with <see cref="KalmanFilter{T}"/> for the
/// system <c>x[k+1] = A·x[k] + B·u[k] + w[k]</c>, <c>y[k] = C·x[k] + v[k]</c>, minimizing the
/// expected quadratic cost when neither the state nor the noise can be observed.
/// </para>
/// <para>
/// <b>The separation principle</b> is what makes this legitimate, and it is not obvious. The
/// controller must choose inputs without knowing the state, so one might expect the estimator and
/// the regulator to have to be designed together — the regulator compensating for the estimator's
/// errors, the estimator anticipating what the regulator will do. They do not. For a linear system
/// with Gaussian noise and a quadratic cost, the optimal solution is exactly: estimate the state as
/// well as possible, ignoring the control problem entirely; then apply the control law you would
/// have used had the estimate been the truth. Each half is designed as if the other were perfect,
/// and the combination is still optimal.
/// </para>
/// <para>
/// This is a genuinely surprising result — it fails for nonlinear systems, for non-Gaussian noise,
/// and for non-quadratic costs — and it is the reason linear control theory is as clean as it is.
/// See H. Kwakernaak and R. Sivan, <i>Linear Optimal Control Systems</i> (Wiley 1972), Chapter 5,
/// and W. M. Wonham, "On the Separation Theorem of Stochastic Control", <i>SIAM Journal on
/// Control</i> 6(2), 1968, pp. 312-326.
/// </para>
/// <para>
/// <b>A caution worth stating.</b> LQR alone has famously good stability margins. LQG does not
/// inherit them: J. C. Doyle, "Guaranteed Margins for LQG Regulators", <i>IEEE Transactions on
/// Automatic Control</i> 23(4), 1978, pp. 756-757, is a one-page paper whose entire content is that
/// an LQG loop can have arbitrarily small margins. Optimal for the model you wrote down is not the
/// same as robust to the ways that model is wrong.
/// </para>
/// <para><b>For Beginners:</b> You want to control something but can only see part of it, through
/// noisy sensors. This runs a best-guess reconstruction of the full situation, then controls as if
/// that guess were fact. Remarkably, that is provably the best you can do — for this class of
/// problem.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var q = 2;
/// var r = 2;
/// var controller = new LinearQuadraticGaussian&lt;double&gt;(
///     stateMatrix: a, inputMatrix: b, observationMatrix: c,
///     stateCost: q, inputCost: r,
///     processNoise: processCovariance, measurementNoise: sensorCovariance);
///
/// controller.Initialize(initialGuess, Matrix&lt;double&gt;.CreateIdentity(2));
///
/// // Each control step: measure, then act.
/// var input = controller.Step(measurement);
/// </code>
/// </example>
public sealed class LinearQuadraticGaussian<T>
{
    private readonly LinearQuadraticRegulator<T> _regulator;
    private readonly KalmanFilter<T> _filter;

    /// <summary>
    /// Creates a linear quadratic Gaussian controller, designing both halves.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>.</param>
    /// <param name="observationMatrix">The observation matrix <c>C</c>.</param>
    /// <param name="stateCost">The state cost <c>Q</c> for the regulator.</param>
    /// <param name="inputCost">The input cost <c>R</c> for the regulator.</param>
    /// <param name="processNoise">The process noise covariance for the estimator.</param>
    /// <param name="measurementNoise">The measurement noise covariance for the estimator.</param>
    /// <param name="options">Riccati solver configuration, or <c>null</c> for the defaults.</param>
    /// <remarks>
    /// <para>
    /// Note the four cost and covariance matrices are distinct and are not interchangeable, even
    /// where they have the same shape. <c>Q</c> and <c>R</c> say what you want; the noise
    /// covariances say what the world does. Passing one where the other belongs produces a
    /// controller that runs and is wrong.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    public LinearQuadraticGaussian(
        Matrix<T> stateMatrix,
        Matrix<T> inputMatrix,
        Matrix<T> observationMatrix,
        Matrix<T> stateCost,
        Matrix<T> inputCost,
        Matrix<T> processNoise,
        Matrix<T> measurementNoise,
        AlgebraicRiccatiSolverOptions? options = null)
    {
        _regulator = new LinearQuadraticRegulator<T>(
            stateMatrix, inputMatrix, stateCost, inputCost, ControlTimeDomain.Discrete, options);

        _filter = new KalmanFilter<T>(
            stateMatrix, observationMatrix, processNoise, measurementNoise, inputMatrix);
    }

    /// <summary>Gets the regulator half, including its gain and cost-to-go matrix.</summary>
    public LinearQuadraticRegulator<T> Regulator => _regulator;

    /// <summary>Gets the estimator half, including its current state and covariance.</summary>
    public KalmanFilter<T> Estimator => _filter;

    /// <summary>Gets the current state estimate.</summary>
    public Vector<T> EstimatedState => _filter.State;

    /// <summary>
    /// Sets the initial state estimate and its covariance.
    /// </summary>
    /// <param name="state">The initial state estimate.</param>
    /// <param name="covariance">How uncertain that estimate is.</param>
    public void Initialize(Vector<T> state, Matrix<T> covariance)
        => _filter.Initialize(state, covariance);

    /// <summary>
    /// Runs one control step: folds in a measurement, then returns the input to apply.
    /// </summary>
    /// <param name="measurement">The measurement received this step.</param>
    /// <returns>The control input to apply, <c>u = −K·x̂</c>.</returns>
    /// <remarks>
    /// <para>
    /// The order matters. The measurement is incorporated first so that the input is computed from
    /// the freshest estimate available; predicting first would act on an estimate one step stale.
    /// The prediction for the next step is then made using the input just returned, which is what
    /// the control matrix is for — an estimator that ignored its own commands would attribute their
    /// effects to noise.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="measurement"/> is null.</exception>
    public Vector<T> Step(Vector<T> measurement)
    {
        _filter.Update(measurement);

        var input = _regulator.ComputeControl(_filter.State);

        _filter.Predict(input);

        return input;
    }
}
