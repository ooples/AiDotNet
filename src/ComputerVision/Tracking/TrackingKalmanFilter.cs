using AiDotNet.Helpers;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Tracking;

/// <summary>
/// The constant-velocity Kalman filter that DeepSORT and ByteTrack run in the
/// <c>(x, y, a, h, vx, vy, va, vh)</c> image space: box centre, aspect ratio (width / height),
/// height and their velocities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>This follows <c>deep_sort/kalman_filter.py</c> (Wojke et al. 2017), which ByteTrack
/// (Zhang et al. 2022) reuses unchanged: process and measurement noise scale with the box height,
/// with position weight 1/20 and velocity weight 1/160. Means are <c>[8, 1]</c> columns and
/// covariances <c>[8, 8]</c>; every step is a whole-matrix engine operation.</para>
/// </remarks>
internal sealed class XyahKalmanFilter<T>
{
    /// <summary>0.95 quantile of the chi-square distribution with 4 degrees of freedom
    /// (<c>chi2inv95[4]</c> in deep_sort), the Mahalanobis gate for a full xyah measurement.</summary>
    internal const double GatingThreshold4Dof = 9.4877;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly Tensor<T> _motion;
    private readonly Tensor<T> _motionT;
    private readonly Tensor<T> _update;
    private readonly Tensor<T> _updateT;
    private readonly Tensor<T> _initCoef;
    private readonly Tensor<T> _initOffset;
    private readonly Tensor<T> _predictCoef;
    private readonly Tensor<T> _predictOffset;
    private readonly Tensor<T> _projectCoef;
    private readonly Tensor<T> _projectOffset;

    /// <summary>Creates the filter with deep_sort's noise weights.</summary>
    public XyahKalmanFilter(double stdWeightPosition = 1.0 / 20, double stdWeightVelocity = 1.0 / 160)
    {
        T zero = NumOps.Zero;
        T wp = NumOps.FromDouble(stdWeightPosition);
        T wv = NumOps.FromDouble(stdWeightVelocity);
        T twoWp = NumOps.FromDouble(2 * stdWeightPosition);
        T tenWv = NumOps.FromDouble(10 * stdWeightVelocity);
        T aspectPos = NumOps.FromDouble(1e-2);
        T aspectVel = NumOps.FromDouble(1e-5);
        T aspectMeasure = NumOps.FromDouble(1e-1);

        // F = [[I, dt*I], [0, I]] with dt = 1; H = [I, 0].
        var eye8 = _engine.TensorEye<T>(8);
        var shift = _engine.TensorSetSlice(new Tensor<T>(new[] { 8, 8 }), _engine.TensorEye<T>(4), new[] { 0, 4 });
        _motion = _engine.TensorAdd(eye8, shift);
        _motionT = _engine.TensorTranspose(_motion);
        _update = _engine.TensorSlice(eye8, new[] { 0, 0 }, new[] { 4, 8 });
        _updateT = _engine.TensorTranspose(_update);

        // Standard deviations are coef * h + offset, as in initiate / predict / project.
        _initCoef = new Tensor<T>(new[] { twoWp, twoWp, zero, twoWp, tenWv, tenWv, zero, tenWv }, new[] { 8 });
        _initOffset = new Tensor<T>(new[] { zero, zero, aspectPos, zero, zero, zero, aspectVel, zero }, new[] { 8 });
        _predictCoef = new Tensor<T>(new[] { wp, wp, zero, wp, wv, wv, zero, wv }, new[] { 8 });
        _predictOffset = new Tensor<T>(new[] { zero, zero, aspectPos, zero, zero, zero, aspectVel, zero }, new[] { 8 });
        _projectCoef = new Tensor<T>(new[] { wp, wp, zero, wp }, new[] { 4 });
        _projectOffset = new Tensor<T>(new[] { zero, zero, aspectMeasure, zero }, new[] { 4 });
    }

    /// <summary>Creates a track from an unassociated xyah measurement <c>[4]</c>.</summary>
    public (Tensor<T> Mean, Tensor<T> Covariance) Initiate(Tensor<T> measurement)
    {
        var column = _engine.Reshape(measurement, new[] { 4, 1 });
        var mean = _engine.TensorConcatenate(new[] { column, new Tensor<T>(new[] { 4, 1 }) }, 0);
        var covariance = NoiseCovariance(_initCoef, _initOffset, measurement[3]);
        return (mean, covariance);
    }

    /// <summary>Runs the prediction step.</summary>
    public (Tensor<T> Mean, Tensor<T> Covariance) Predict(Tensor<T> mean, Tensor<T> covariance)
    {
        var processNoise = NoiseCovariance(_predictCoef, _predictOffset, mean[3, 0]);
        var newMean = _engine.TensorMatMul(_motion, mean);
        var newCovariance = _engine.TensorAdd(
            _engine.TensorMatMul(_engine.TensorMatMul(_motion, covariance), _motionT), processNoise);
        return (newMean, newCovariance);
    }

    /// <summary>Projects the state distribution to measurement space.</summary>
    public (Tensor<T> Mean, Tensor<T> Covariance) Project(Tensor<T> mean, Tensor<T> covariance)
    {
        var innovationNoise = NoiseCovariance(_projectCoef, _projectOffset, mean[3, 0]);
        var projectedMean = _engine.TensorMatMul(_update, mean);
        var projectedCovariance = _engine.TensorAdd(
            _engine.TensorMatMul(_engine.TensorMatMul(_update, covariance), _updateT), innovationNoise);
        return (projectedMean, projectedCovariance);
    }

    /// <summary>Runs the correction step with an xyah measurement <c>[4]</c>.</summary>
    public (Tensor<T> Mean, Tensor<T> Covariance) Update(Tensor<T> mean, Tensor<T> covariance, Tensor<T> measurement)
    {
        var (projectedMean, projectedCovariance) = Project(mean, covariance);
        var factor = LinalgDispatch<T>.CholeskyEx(projectedCovariance).L;

        // kalman_gain = cho_solve(chol, (P H^T)^T)^T
        var gainT = LinalgDispatch<T>.CholeskySolve(_engine.TensorMatMul(_update, covariance), factor);
        var gain = _engine.TensorTranspose(gainT);
        var innovation = _engine.TensorSubtract(_engine.Reshape(measurement, new[] { 4, 1 }), projectedMean);

        var newMean = _engine.TensorAdd(mean, _engine.TensorMatMul(gain, innovation));
        var newCovariance = _engine.TensorSubtract(
            covariance, _engine.TensorMatMul(_engine.TensorMatMul(gain, projectedCovariance), gainT));
        return (newMean, newCovariance);
    }

    /// <summary>
    /// Squared Mahalanobis distance between the state distribution and each xyah measurement row of
    /// <paramref name="measurements"/> (<c>[N, 4]</c>); returns <c>[N]</c>.
    /// </summary>
    public Tensor<T> GatingDistance(Tensor<T> mean, Tensor<T> covariance, Tensor<T> measurements)
    {
        int count = measurements.Shape[0];
        var (projectedMean, projectedCovariance) = Project(mean, covariance);
        var factor = LinalgDispatch<T>.CholeskyEx(projectedCovariance).L;
        var deltaT = _engine.TensorSubtract(
            _engine.TensorTranspose(measurements), _engine.TensorBroadcastTo(projectedMean, new[] { 4, count }));
        var z = LinalgDispatch<T>.SolveTriangular(factor, deltaT, upper: false);
        return _engine.ReduceSum(_engine.TensorSquare(z), new[] { 0 }, keepDims: false);
    }

    private Tensor<T> NoiseCovariance(Tensor<T> coef, Tensor<T> offset, T height)
    {
        var std = _engine.TensorAdd(_engine.TensorMultiplyScalar(coef, height), offset);
        return _engine.TensorDiag(_engine.TensorSquare(std));
    }
}
