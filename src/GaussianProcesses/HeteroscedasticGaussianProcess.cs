namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Gaussian Process with input-dependent (heteroscedastic) noise levels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard Gaussian Processes assume the same level of noise
/// everywhere in your data. But real-world data often has varying uncertainty:
/// - Sensor readings may be more accurate in certain conditions
/// - Financial data has varying volatility
/// - Experimental measurements may be more precise in some ranges
///
/// The Heteroscedastic GP models this by learning a separate GP for the noise variance,
/// allowing it to capture input-dependent uncertainty.
///
/// This uses the "Most Likely Heteroscedastic GP" (MLHGP) approach:
/// 1. A primary GP models the mean function f(x)
/// 2. A secondary GP models the log noise variance g(x) = log(σ²(x))
/// 3. The two GPs are jointly optimized
///
/// The noise at any point x is: σ²(x) = exp(g(x))
/// This exponential ensures the variance is always positive.
/// </para>
/// <para>
/// When to use:
/// - Data with varying noise levels
/// - Sensor fusion with different accuracy sensors
/// - Financial modeling with time-varying volatility
/// - Any regression where uncertainty varies with input
/// </para>
/// </remarks>
public class HeteroscedasticGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The kernel function for the mean GP.
    /// </summary>
    private readonly IKernelFunction<T> _meanKernel;

    /// <summary>
    /// The kernel function for the noise GP.
    /// </summary>
    private readonly IKernelFunction<T> _noiseKernel;

    /// <summary>
    /// Training input data.
    /// </summary>
    private Matrix<T>? _X;

    /// <summary>
    /// Training target values.
    /// </summary>
    private Vector<T>? _y;

    /// <summary>
    /// Cholesky decomposition of mean GP covariance matrix.
    /// </summary>
    private Matrix<T>? _L;

    /// <summary>
    /// Alpha coefficients for mean predictions.
    /// </summary>
    private Vector<T>? _alpha;

    /// <summary>
    /// Learned noise variances at training points.
    /// </summary>
    private Vector<T>? _noiseVariances;

    /// <summary>
    /// Noise GP latent values (log noise variance).
    /// </summary>
    private Vector<T>? _noiseLatentValues;

    /// <summary>
    /// Prior mean for noise GP (log scale).
    /// </summary>
    private readonly double _noisePriorMean;

    /// <summary>
    /// Prior variance for noise GP.
    /// </summary>
    private readonly double _noisePriorVariance;

    /// <summary>
    /// Small regularization constant.
    /// </summary>
    private readonly double _jitter;

    /// <summary>
    /// Number of EM iterations for joint optimization.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance for EM algorithm.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// Whether the model has been trained.
    /// </summary>
    private bool _isTrained;

    /// <summary>
    /// Initializes a new heteroscedastic Gaussian Process.
    /// </summary>
    /// <param name="meanKernel">Kernel for the mean function GP.</param>
    /// <param name="noiseKernel">Kernel for the noise variance GP.</param>
    /// <param name="noisePriorMean">Prior mean for log noise (default: log(0.1) ≈ -2.3).</param>
    /// <param name="noisePriorVariance">Prior variance for noise GP (default: 1.0).</param>
    /// <param name="maxIterations">Maximum EM iterations (default: 50).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-4).</param>
    /// <param name="jitter">Numerical stability constant (default: 1e-6).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a heteroscedastic GP with separate kernels for
    /// mean and noise modeling.
    ///
    /// Tips for kernel selection:
    /// - Mean kernel: Choose based on expected smoothness of your function
    ///   (RBF for smooth, Matern for less smooth)
    /// - Noise kernel: Often use RBF with longer length scale
    ///   (noise typically varies smoothly across input space)
    ///
    /// The noisePriorMean controls the expected noise level:
    /// - log(0.01) ≈ -4.6 for low noise
    /// - log(0.1) ≈ -2.3 for moderate noise (default)
    /// - log(1.0) = 0 for high noise
    /// </para>
    /// </remarks>
    public HeteroscedasticGaussianProcess(
        IKernelFunction<T> meanKernel,
        IKernelFunction<T>? noiseKernel = null,
        double noisePriorMean = -2.302585, // log(0.1)
        double noisePriorVariance = 1.0,
        int maxIterations = 50,
        double tolerance = 1e-4,
        double jitter = 1e-6)
    {
        if (meanKernel is null) throw new ArgumentNullException(nameof(meanKernel));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));
        if (tolerance <= 0)
            throw new ArgumentException("Tolerance must be positive.", nameof(tolerance));
        if (jitter <= 0)
            throw new ArgumentException("Jitter must be positive.", nameof(jitter));

        _meanKernel = meanKernel;
        _noiseKernel = noiseKernel ?? meanKernel;
        _noisePriorMean = noisePriorMean;
        _noisePriorVariance = noisePriorVariance;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _jitter = jitter;
        _numOps = MathHelper.GetNumericOperations<T>();
        _isTrained = false;
    }

    /// <summary>
    /// Gets the kernel function used for the mean GP.
    /// </summary>
    public IKernelFunction<T> MeanKernel => _meanKernel;

    /// <summary>
    /// Gets the kernel function used for the noise GP.
    /// </summary>
    public IKernelFunction<T> NoiseKernel => _noiseKernel;

    /// <summary>
    /// Gets the learned noise variances at training points.
    /// </summary>
    /// <returns>Array of noise variances, or null if not trained.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After training, this shows how much noise the model
    /// thinks exists at each training point. High values indicate more uncertainty
    /// in the data at that location.
    /// </para>
    /// </remarks>
    public double[]? GetNoiseVariances()
    {
        if (_noiseVariances is null) return null;

        var result = new double[_noiseVariances.Length];
        for (int i = 0; i < _noiseVariances.Length; i++)
        {
            result[i] = _numOps.ToDouble(_noiseVariances[i]);
        }
        return result;
    }

    /// <summary>
    /// Trains the heteroscedastic GP on the given data.
    /// </summary>
    /// <param name="X">Training inputs as rows.</param>
    /// <param name="y">Training targets.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This fits both the mean GP and noise GP to your data
    /// using an iterative algorithm called Expectation-Maximization (EM).
    ///
    /// The process alternates between:
    /// 1. E-step: Estimate noise levels given current mean predictions
    /// 2. M-step: Update mean GP given current noise estimates
    ///
    /// This continues until convergence or max iterations reached.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of rows in X must match length of y.");

        _X = X;
        _y = y;
        int n = X.Rows;

        // Initialize noise latent values to prior mean
        _noiseLatentValues = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            _noiseLatentValues[i] = _numOps.FromDouble(_noisePriorMean);
        }

        // Initialize noise variances
        _noiseVariances = new Vector<T>(n);
        UpdateNoiseVariances();

        // EM-style optimization
        double prevLoss = double.MaxValue;
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // M-step: Update mean GP with current noise estimates
            FitMeanGP();

            // E-step: Update noise estimates given current mean GP
            UpdateNoiseLatentValues();
            UpdateNoiseVariances();

            // Check convergence
            double loss = ComputeNegativeLogLikelihood();
            if (Math.Abs(prevLoss - loss) < _tolerance)
            {
                break;
            }
            prevLoss = loss;
        }

        _isTrained = true;
    }

    /// <summary>
    /// Updates the noise variance values from latent values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The noise GP models log(σ²), so we apply exp()
    /// to get the actual noise variances. The exponential ensures variances
    /// are always positive.
    /// </para>
    /// </remarks>
    private void UpdateNoiseVariances()
    {
        if (_noiseLatentValues is null || _noiseVariances is null) return;

        for (int i = 0; i < _noiseLatentValues.Length; i++)
        {
            double latent = _numOps.ToDouble(_noiseLatentValues[i]);
            _noiseVariances[i] = _numOps.FromDouble(Math.Exp(latent));
        }
    }

    /// <summary>
    /// Fits the mean GP with current noise estimates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the standard GP fitting procedure, but using
    /// the heteroscedastic noise variances instead of a constant noise.
    /// Each data point gets its own noise variance on the diagonal.
    /// </para>
    /// </remarks>
    private void FitMeanGP()
    {
        if (_X is null || _y is null || _noiseVariances is null) return;

        int n = _X.Rows;

        // Compute kernel matrix
        var K = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(_X, i);
            for (int j = i; j < n; j++)
            {
                var xj = GetRow(_X, j);
                T kval = _meanKernel.Calculate(xi, xj);

                if (i == j)
                {
                    // Add heteroscedastic noise + jitter to diagonal
                    double kd = _numOps.ToDouble(kval) +
                                _numOps.ToDouble(_noiseVariances[i]) +
                                _jitter;
                    K[i, j] = _numOps.FromDouble(kd);
                }
                else
                {
                    K[i, j] = kval;
                    K[j, i] = kval;
                }
            }
        }

        // Cholesky decomposition
        _L = CholeskyDecomposition(K);

        // Solve L * z = y for z, then L^T * alpha = z for alpha
        var z = SolveLowerTriangular(_L, _y);
        _alpha = SolveUpperTriangular(Transpose(_L), z);
    }

    /// <summary>
    /// Updates noise latent values using the current mean GP.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This updates our estimate of the noise at each point
    /// based on the squared residuals (how far the mean prediction is from the data).
    /// Large residuals suggest higher noise.
    /// </para>
    /// </remarks>
    private void UpdateNoiseLatentValues()
    {
        if (_X is null || _y is null || _alpha is null || _noiseLatentValues is null) return;

        int n = _X.Rows;

        // Compute residuals
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(_X, i);
            double pred = 0;
            for (int j = 0; j < n; j++)
            {
                var xj = GetRow(_X, j);
                double kval = _numOps.ToDouble(_meanKernel.Calculate(xi, xj));
                pred += kval * _numOps.ToDouble(_alpha[j]);
            }
            residuals[i] = _numOps.ToDouble(_y[i]) - pred;
        }

        // Update noise latent values using squared residuals with GP smoothing
        var K_noise = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(_X, i);
            for (int j = i; j < n; j++)
            {
                var xj = GetRow(_X, j);
                T kval = _noiseKernel.Calculate(xi, xj);

                if (i == j)
                {
                    double kd = _numOps.ToDouble(kval) + _noisePriorVariance + _jitter;
                    K_noise[i, j] = _numOps.FromDouble(kd);
                }
                else
                {
                    K_noise[i, j] = kval;
                    K_noise[j, i] = kval;
                }
            }
        }

        // Target for noise GP: log of squared residuals
        var logResidualsSq = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double r2 = residuals[i] * residuals[i];
            // Use prior mean as floor to prevent -inf
            double logR2 = Math.Max(Math.Log(r2 + 1e-10), _noisePriorMean - 3);
            logResidualsSq[i] = _numOps.FromDouble(logR2);
        }

        // Solve noise GP
        var L_noise = CholeskyDecomposition(K_noise);
        var z_noise = SolveLowerTriangular(L_noise, logResidualsSq);
        var alpha_noise = SolveUpperTriangular(Transpose(L_noise), z_noise);

        // Update noise latent values
        for (int i = 0; i < n; i++)
        {
            double newLatent = 0;
            var xi = GetRow(_X, i);
            for (int j = 0; j < n; j++)
            {
                var xj = GetRow(_X, j);
                double kval = _numOps.ToDouble(_noiseKernel.Calculate(xi, xj));
                newLatent += kval * _numOps.ToDouble(alpha_noise[j]);
            }
            // Blend with prior
            newLatent = 0.7 * newLatent + 0.3 * _noisePriorMean;
            _noiseLatentValues[i] = _numOps.FromDouble(newLatent);
        }
    }

    /// <summary>
    /// Computes the negative log marginal likelihood.
    /// </summary>
    /// <returns>The negative log likelihood value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This measures how well the model fits the data.
    /// Lower values mean better fit. It's used to monitor convergence during training.
    /// </para>
    /// </remarks>
    private double ComputeNegativeLogLikelihood()
    {
        if (_L is null || _alpha is null || _y is null) return double.MaxValue;

        int n = _y.Length;

        // Data fit term: 0.5 * y^T * K^{-1} * y = 0.5 * y^T * alpha
        double dataFit = 0;
        for (int i = 0; i < n; i++)
        {
            dataFit += _numOps.ToDouble(_y[i]) * _numOps.ToDouble(_alpha[i]);
        }
        dataFit *= 0.5;

        // Complexity penalty: 0.5 * log|K| = sum(log(L_ii))
        double logDet = 0;
        for (int i = 0; i < n; i++)
        {
            logDet += Math.Log(Math.Max(_numOps.ToDouble(_L[i, i]), 1e-10));
        }

        // Constant term
        double constant = 0.5 * n * Math.Log(2 * Math.PI);

        return dataFit + logDet + constant;
    }

    /// <summary>
    /// Makes predictions with uncertainty estimates.
    /// </summary>
    /// <param name="XNew">New input points as rows.</param>
    /// <returns>Tuple of (mean predictions, predictive variances, noise variances at new points).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts both the function value and two types of uncertainty:
    /// 1. Predictive variance: Uncertainty from not knowing the function (epistemic)
    /// 2. Noise variance: Expected observation noise at the point (aleatoric)
    ///
    /// Total uncertainty = predictive variance + noise variance
    /// </para>
    /// </remarks>
    public (Vector<T> Mean, Vector<T> PredictiveVariance, Vector<T> NoiseVariance) Predict(Matrix<T> XNew)
    {
        if (!_isTrained)
            throw new InvalidOperationException("Model must be trained before prediction.");
        if (_X is null || _alpha is null || _L is null || _noiseLatentValues is null)
            throw new InvalidOperationException("Model not properly initialized.");

        int n = _X.Rows;
        int nNew = XNew.Rows;

        var mean = new Vector<T>(nNew);
        var predictiveVar = new Vector<T>(nNew);
        var noiseVar = new Vector<T>(nNew);

        // For noise GP prediction
        var K_noise = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(_X, i);
            for (int j = i; j < n; j++)
            {
                var xj = GetRow(_X, j);
                T kval = _noiseKernel.Calculate(xi, xj);
                if (i == j)
                {
                    double kd = _numOps.ToDouble(kval) + _noisePriorVariance + _jitter;
                    K_noise[i, j] = _numOps.FromDouble(kd);
                }
                else
                {
                    K_noise[i, j] = kval;
                    K_noise[j, i] = kval;
                }
            }
        }
        var L_noise = CholeskyDecomposition(K_noise);
        var alpha_noise = SolveUpperTriangular(Transpose(L_noise),
            SolveLowerTriangular(L_noise, _noiseLatentValues));

        for (int i = 0; i < nNew; i++)
        {
            var xNew = GetRow(XNew, i);

            // Compute k_* (covariance between new point and training points)
            var kStar = new Vector<T>(n);
            var kStarNoise = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                var xj = GetRow(_X, j);
                kStar[j] = _meanKernel.Calculate(xNew, xj);
                kStarNoise[j] = _noiseKernel.Calculate(xNew, xj);
            }

            // Mean prediction: k_*^T * alpha
            double predMean = 0;
            for (int j = 0; j < n; j++)
            {
                predMean += _numOps.ToDouble(kStar[j]) * _numOps.ToDouble(_alpha[j]);
            }
            mean[i] = _numOps.FromDouble(predMean);

            // Predictive variance: k_** - k_*^T * K^{-1} * k_*
            double kStarStar = _numOps.ToDouble(_meanKernel.Calculate(xNew, xNew));
            var v = SolveLowerTriangular(_L, kStar);
            double vTv = 0;
            for (int j = 0; j < n; j++)
            {
                vTv += _numOps.ToDouble(v[j]) * _numOps.ToDouble(v[j]);
            }
            predictiveVar[i] = _numOps.FromDouble(Math.Max(kStarStar - vTv, 1e-10));

            // Noise variance prediction
            double predNoiseLatent = 0;
            for (int j = 0; j < n; j++)
            {
                predNoiseLatent += _numOps.ToDouble(kStarNoise[j]) * _numOps.ToDouble(alpha_noise[j]);
            }
            noiseVar[i] = _numOps.FromDouble(Math.Exp(predNoiseLatent));
        }

        return (mean, predictiveVar, noiseVar);
    }

    /// <summary>
    /// Predicts the mean and variance for a single input point.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>Tuple of (mean, variance) including noise variance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts for a single point. The variance returned
    /// is the total uncertainty (predictive + noise variance).
    /// </para>
    /// </remarks>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        // Convert single point to matrix
        var XNew = new Matrix<T>(1, x.Length);
        for (int j = 0; j < x.Length; j++)
        {
            XNew[0, j] = x[j];
        }

        var (means, predVars, noiseVars) = Predict(XNew);
        double totalVar = _numOps.ToDouble(predVars[0]) + _numOps.ToDouble(noiseVars[0]);
        return (means[0], _numOps.FromDouble(totalVar));
    }

    /// <summary>
    /// Updates the kernel function used for the mean GP.
    /// </summary>
    /// <param name="kernel">The new kernel function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This changes the mean kernel and invalidates the trained model.
    /// You'll need to retrain after updating the kernel.
    /// </para>
    /// </remarks>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        // Note: Since _meanKernel is readonly, we throw to indicate the model needs recreation
        throw new NotSupportedException(
            "HeteroscedasticGaussianProcess does not support kernel updates after construction. " +
            "Create a new instance with the desired kernel.");
    }

    /// <summary>
    /// Gets the total predictive uncertainty (epistemic + aleatoric).
    /// </summary>
    /// <param name="XNew">New input points.</param>
    /// <returns>Total uncertainty at each point.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This combines both sources of uncertainty into
    /// a single value, useful for applications like confidence intervals.
    /// </para>
    /// </remarks>
    public Vector<T> GetTotalUncertainty(Matrix<T> XNew)
    {
        var (_, predVar, noiseVar) = Predict(XNew);

        var total = new Vector<T>(predVar.Length);
        for (int i = 0; i < predVar.Length; i++)
        {
            double pv = _numOps.ToDouble(predVar[i]);
            double nv = _numOps.ToDouble(noiseVar[i]);
            total[i] = _numOps.FromDouble(pv + nv);
        }
        return total;
    }

    #region Linear Algebra Helpers

    /// <summary>
    /// Extracts a row from a matrix as a vector.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[row, j];
        }
        return result;
    }

    /// <summary>
    /// Performs Cholesky decomposition of a symmetric positive-definite matrix.
    /// </summary>
    private Matrix<T> CholeskyDecomposition(Matrix<T> A)
    {
        int n = A.Rows;
        var L = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += _numOps.ToDouble(L[i, k]) * _numOps.ToDouble(L[j, k]);
                }

                if (i == j)
                {
                    double val = _numOps.ToDouble(A[i, i]) - sum;
                    L[i, j] = _numOps.FromDouble(Math.Sqrt(Math.Max(val, 1e-10)));
                }
                else
                {
                    double lii = _numOps.ToDouble(L[j, j]);
                    L[i, j] = _numOps.FromDouble((_numOps.ToDouble(A[i, j]) - sum) / lii);
                }
            }
        }
        return L;
    }

    /// <summary>
    /// Solves Lx = b for x where L is lower triangular.
    /// </summary>
    private Vector<T> SolveLowerTriangular(Matrix<T> L, Vector<T> b)
    {
        int n = b.Length;
        var x = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < i; j++)
            {
                sum += _numOps.ToDouble(L[i, j]) * _numOps.ToDouble(x[j]);
            }
            x[i] = _numOps.FromDouble((_numOps.ToDouble(b[i]) - sum) / _numOps.ToDouble(L[i, i]));
        }
        return x;
    }

    /// <summary>
    /// Solves Ux = b for x where U is upper triangular.
    /// </summary>
    private Vector<T> SolveUpperTriangular(Matrix<T> U, Vector<T> b)
    {
        int n = b.Length;
        var x = new Vector<T>(n);

        for (int i = n - 1; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < n; j++)
            {
                sum += _numOps.ToDouble(U[i, j]) * _numOps.ToDouble(x[j]);
            }
            x[i] = _numOps.FromDouble((_numOps.ToDouble(b[i]) - sum) / _numOps.ToDouble(U[i, i]));
        }
        return x;
    }

    /// <summary>
    /// Transposes a matrix.
    /// </summary>
    private Matrix<T> Transpose(Matrix<T> A)
    {
        var result = new Matrix<T>(A.Columns, A.Rows);
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = 0; j < A.Columns; j++)
            {
                result[j, i] = A[i, j];
            }
        }
        return result;
    }

    #endregion
}
