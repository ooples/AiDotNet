namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Implements a Gaussian Process with Student-t likelihood for robust regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard Gaussian Processes assume Gaussian (normal) noise,
/// which is sensitive to outliers. A single outlier can drastically affect predictions.
///
/// The Student-t GP uses a heavy-tailed Student-t distribution instead:
/// - More tolerant of outliers (they have less influence)
/// - The "degrees of freedom" parameter (ν) controls robustness:
///   - ν = 1: Cauchy distribution (very robust, heavy tails)
///   - ν = 4-5: Good balance of robustness and efficiency
///   - ν → ∞: Approaches Gaussian (standard GP)
///
/// This uses Expectation Propagation (EP) for approximate inference since
/// the Student-t likelihood makes exact inference intractable.
/// </para>
/// <para>
/// When to use:
/// - Data with potential outliers
/// - Sensor data with occasional erroneous readings
/// - Financial data with market anomalies
/// - Any regression where robustness to bad data points is important
/// </para>
/// </remarks>
public class StudentTGaussianProcess<T> : IGaussianProcess<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The kernel function.
    /// </summary>
    private readonly IKernelFunction<T> _kernel;

    /// <summary>
    /// Degrees of freedom for Student-t distribution.
    /// </summary>
    private readonly double _nu;

    /// <summary>
    /// Scale parameter for Student-t noise.
    /// </summary>
    private double _scale;

    /// <summary>
    /// Training input data.
    /// </summary>
    private Matrix<T>? _X;

    /// <summary>
    /// Training target values.
    /// </summary>
    private Vector<T>? _y;

    /// <summary>
    /// Prior covariance matrix.
    /// </summary>
    private Matrix<T>? _K;

    /// <summary>
    /// Approximate posterior mean.
    /// </summary>
    private Vector<T>? _posteriorMean;

    /// <summary>
    /// Approximate posterior covariance.
    /// </summary>
    private Matrix<T>? _posteriorCov;

    /// <summary>
    /// Site natural parameters (precision).
    /// </summary>
    private Vector<T>? _sitePrecisions;

    /// <summary>
    /// Site natural parameters (precision * mean).
    /// </summary>
    private Vector<T>? _siteNaturalMeans;

    /// <summary>
    /// Outlier weights (downweights outliers).
    /// </summary>
    private Vector<T>? _weights;

    /// <summary>
    /// Small regularization constant.
    /// </summary>
    private readonly double _jitter;

    /// <summary>
    /// Maximum number of EP iterations.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance.
    /// </summary>
    private readonly double _tolerance;

    /// <summary>
    /// Damping factor for EP updates.
    /// </summary>
    private readonly double _damping;

    /// <summary>
    /// Whether the model has been trained.
    /// </summary>
    private bool _isTrained;

    /// <summary>
    /// Initializes a new Student-t Gaussian Process.
    /// </summary>
    /// <param name="kernel">The kernel function.</param>
    /// <param name="nu">Degrees of freedom (default: 4.0). Lower = more robust.</param>
    /// <param name="scale">Initial noise scale (default: 0.1).</param>
    /// <param name="maxIterations">Maximum EP iterations (default: 100).</param>
    /// <param name="tolerance">Convergence tolerance (default: 1e-5).</param>
    /// <param name="damping">Damping for EP updates (default: 0.5).</param>
    /// <param name="jitter">Numerical stability constant (default: 1e-6).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a robust GP with Student-t likelihood.
    ///
    /// Choosing degrees of freedom (ν):
    /// - ν = 1: Cauchy distribution - extremely robust, use for severe outliers
    /// - ν = 3-5: Good for moderate outlier robustness (recommended starting point)
    /// - ν = 10: Mild robustness
    /// - ν > 30: Essentially Gaussian
    ///
    /// The scale parameter is like the standard deviation in regular GP noise.
    /// </para>
    /// </remarks>
    public StudentTGaussianProcess(
        IKernelFunction<T> kernel,
        double nu = 4.0,
        double scale = 0.1,
        int maxIterations = 100,
        double tolerance = 1e-5,
        double damping = 0.5,
        double jitter = 1e-6)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (nu <= 0)
            throw new ArgumentException("Degrees of freedom must be positive.", nameof(nu));
        if (scale <= 0)
            throw new ArgumentException("Scale must be positive.", nameof(scale));
        if (damping <= 0 || damping > 1)
            throw new ArgumentException("Damping must be in (0, 1].", nameof(damping));

        _kernel = kernel;
        _nu = nu;
        _scale = scale;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _damping = damping;
        _jitter = jitter;
        _numOps = MathHelper.GetNumericOperations<T>();
        _isTrained = false;
    }

    /// <summary>
    /// Gets the kernel function.
    /// </summary>
    public IKernelFunction<T> Kernel => _kernel;

    /// <summary>
    /// Gets the degrees of freedom.
    /// </summary>
    public double DegreesOfFreedom => _nu;

    /// <summary>
    /// Gets the noise scale parameter.
    /// </summary>
    public double Scale => _scale;

    /// <summary>
    /// Gets the outlier weights after training.
    /// </summary>
    /// <returns>Array of weights in [0,1], where lower values indicate outliers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After training, this shows how much each training point
    /// is trusted. Values close to 1 are normal points; values close to 0 are likely outliers.
    /// This can help identify problematic data points in your dataset.
    /// </para>
    /// </remarks>
    public double[]? GetOutlierWeights()
    {
        if (_weights is null) return null;

        var result = new double[_weights.Length];
        for (int i = 0; i < _weights.Length; i++)
        {
            result[i] = _numOps.ToDouble(_weights[i]);
        }
        return result;
    }

    /// <summary>
    /// Identifies outliers in the training data.
    /// </summary>
    /// <param name="threshold">Weight threshold below which points are considered outliers (default: 0.5).</param>
    /// <returns>Indices of detected outliers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the indices of training points that the model
    /// considers outliers based on their weights. You can use these indices to:
    /// - Inspect the suspicious data points
    /// - Remove them and retrain
    /// - Investigate why they might be erroneous
    /// </para>
    /// </remarks>
    public int[] GetOutlierIndices(double threshold = 0.5)
    {
        if (_weights is null)
            throw new InvalidOperationException("Model must be trained before identifying outliers.");

        var outliers = new List<int>();
        for (int i = 0; i < _weights.Length; i++)
        {
            if (_numOps.ToDouble(_weights[i]) < threshold)
            {
                outliers.Add(i);
            }
        }
        return outliers.ToArray();
    }

    /// <summary>
    /// Trains the Student-t GP using Expectation Propagation.
    /// </summary>
    /// <param name="X">Training inputs as rows.</param>
    /// <param name="y">Training targets.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This fits the GP to your data using an iterative algorithm
    /// called Expectation Propagation (EP). EP approximates the non-Gaussian posterior
    /// (caused by the Student-t likelihood) with a Gaussian.
    ///
    /// During training, the algorithm automatically identifies and downweights outliers,
    /// so they have less influence on the predictions.
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

        // Compute prior covariance matrix
        _K = ComputeKernelMatrix(X, X);

        // Initialize EP site parameters
        _sitePrecisions = new Vector<T>(n);
        _siteNaturalMeans = new Vector<T>(n);
        _weights = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            _sitePrecisions[i] = _numOps.FromDouble(1e-6);
            _siteNaturalMeans[i] = _numOps.FromDouble(0);
            _weights[i] = _numOps.FromDouble(1.0);
        }

        // Initialize posterior to prior
        _posteriorMean = new Vector<T>(n);
        _posteriorCov = CopyMatrix(_K);

        // EP iterations
        double prevLogZ = double.NegativeInfinity;
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            // Update each site
            for (int i = 0; i < n; i++)
            {
                double change = UpdateSite(i);
                maxChange = Math.Max(maxChange, change);
            }

            // Update posterior
            UpdatePosterior();

            // Check convergence
            double logZ = ComputeLogMarginalLikelihood();
            if (Math.Abs(logZ - prevLogZ) < _tolerance)
            {
                break;
            }
            prevLogZ = logZ;

            if (maxChange < _tolerance)
            {
                break;
            }
        }

        _isTrained = true;
    }

    /// <summary>
    /// Updates a single EP site.
    /// </summary>
    /// <param name="i">Index of the site to update.</param>
    /// <returns>The magnitude of the change.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> EP works by iteratively refining local approximations
    /// at each data point (called "sites"). This method updates one site by:
    /// 1. Computing what the posterior would look like without this site
    /// 2. Including the true Student-t likelihood at this point
    /// 3. Projecting back to a Gaussian approximation
    /// </para>
    /// </remarks>
    private double UpdateSite(int i)
    {
        if (_posteriorMean is null || _posteriorCov is null ||
            _sitePrecisions is null || _siteNaturalMeans is null || _y is null || _weights is null)
            return 0;

        // Cavity distribution (posterior without site i)
        double postPrecision = 1.0 / Math.Max(_numOps.ToDouble(_posteriorCov[i, i]), 1e-10);
        double sitePrecision = _numOps.ToDouble(_sitePrecisions[i]);
        double cavityPrecision = postPrecision - sitePrecision;

        if (cavityPrecision <= 0)
        {
            cavityPrecision = 1e-6;
        }

        double postMean = _numOps.ToDouble(_posteriorMean[i]);
        double siteNaturalMean = _numOps.ToDouble(_siteNaturalMeans[i]);
        double cavityMean = (postPrecision * postMean - siteNaturalMean) / cavityPrecision;

        // Compute tilted distribution moments via numerical integration
        double yi = _numOps.ToDouble(_y[i]);
        var (tiltedMean, tiltedVar, logZ) = ComputeTiltedMoments(
            yi, cavityMean, 1.0 / cavityPrecision);

        // Compute new site parameters
        double tiltedPrecision = 1.0 / Math.Max(tiltedVar, 1e-10);
        double newSitePrecision = tiltedPrecision - cavityPrecision;
        double newSiteNaturalMean = tiltedPrecision * tiltedMean - cavityPrecision * cavityMean;

        // Ensure positive site precision
        if (newSitePrecision < 1e-10)
        {
            newSitePrecision = 1e-10;
        }

        // Update weights (outlier detection)
        double residual = Math.Abs(yi - tiltedMean);
        double scaledResidual = residual / (_scale * Math.Sqrt(tiltedVar + _scale * _scale));
        double weight = (_nu + 1) / (_nu + scaledResidual * scaledResidual);
        weight = Math.Min(1.0, Math.Max(0.01, weight));

        // Damped update
        double oldPrecision = sitePrecision;
        double oldNaturalMean = siteNaturalMean;
        double oldWeight = _numOps.ToDouble(_weights[i]);

        double dampedPrecision = _damping * newSitePrecision + (1 - _damping) * oldPrecision;
        double dampedNaturalMean = _damping * newSiteNaturalMean + (1 - _damping) * oldNaturalMean;
        double dampedWeight = _damping * weight + (1 - _damping) * oldWeight;

        _sitePrecisions[i] = _numOps.FromDouble(dampedPrecision);
        _siteNaturalMeans[i] = _numOps.FromDouble(dampedNaturalMean);
        _weights[i] = _numOps.FromDouble(dampedWeight);

        return Math.Abs(dampedPrecision - oldPrecision) +
               Math.Abs(dampedNaturalMean - oldNaturalMean);
    }

    /// <summary>
    /// Computes moments of the tilted distribution via numerical integration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The tilted distribution combines the cavity (prior without this point)
    /// with the Student-t likelihood. We compute its mean and variance using numerical integration.
    /// </para>
    /// </remarks>
    private (double Mean, double Variance, double LogZ) ComputeTiltedMoments(
        double y, double cavityMean, double cavityVar)
    {
        // Use Gauss-Hermite quadrature
        int numPoints = 20;
        double[] nodes = new double[numPoints];
        double[] weights = new double[numPoints];
        GaussHermiteQuadrature(numPoints, nodes, weights);

        double cavityStd = Math.Sqrt(cavityVar);
        double z = 0, m1 = 0, m2 = 0;

        for (int j = 0; j < numPoints; j++)
        {
            double f = cavityMean + cavityStd * Math.Sqrt(2) * nodes[j];
            double logLik = StudentTLogPdf(y - f, _nu, _scale);
            double w = weights[j] * Math.Exp(logLik);

            z += w;
            m1 += w * f;
            m2 += w * f * f;
        }

        z /= Math.Sqrt(Math.PI);
        m1 /= Math.Sqrt(Math.PI);
        m2 /= Math.Sqrt(Math.PI);

        if (z < 1e-100)
        {
            return (cavityMean, cavityVar, -100);
        }

        double mean = m1 / z;
        double variance = m2 / z - mean * mean;
        variance = Math.Max(variance, 1e-10);

        return (mean, variance, Math.Log(z));
    }

    /// <summary>
    /// Computes the log PDF of the Student-t distribution.
    /// </summary>
    private double StudentTLogPdf(double x, double nu, double sigma)
    {
        double z = x / sigma;
        double logNorm = LogGamma((nu + 1) / 2) - LogGamma(nu / 2) -
                         0.5 * Math.Log(nu * Math.PI) - Math.Log(sigma);
        double logKernel = -((nu + 1) / 2) * Math.Log(1 + z * z / nu);
        return logNorm + logKernel;
    }

    /// <summary>
    /// Updates the posterior distribution from site parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After updating all sites, we compute the new posterior
    /// by combining the prior (kernel matrix) with all the site approximations.
    /// </para>
    /// </remarks>
    private void UpdatePosterior()
    {
        if (_K is null || _sitePrecisions is null || _siteNaturalMeans is null ||
            _posteriorMean is null || _posteriorCov is null) return;

        int n = _K.Rows;

        // Posterior precision: K^{-1} + diag(site_precisions)
        // Posterior covariance: (K^{-1} + diag(site_precisions))^{-1}

        // Build K + diag(1/site_precisions) for numerical stability
        var Sigma = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Sigma[i, j] = _K[i, j];
            }
            double sp = _numOps.ToDouble(_sitePrecisions[i]);
            if (sp > 1e-10)
            {
                double diag = _numOps.ToDouble(Sigma[i, i]) + 1.0 / sp;
                Sigma[i, i] = _numOps.FromDouble(diag);
            }
        }

        // Compute posterior covariance via Cholesky
        var L = CholeskyDecomposition(Sigma);
        _posteriorCov = ComputeInverseFromCholesky(L);

        // Compute posterior mean
        for (int i = 0; i < n; i++)
        {
            double mean = 0;
            for (int j = 0; j < n; j++)
            {
                double snm = _numOps.ToDouble(_siteNaturalMeans[j]);
                double sp = _numOps.ToDouble(_sitePrecisions[j]);
                if (sp > 1e-10)
                {
                    mean += _numOps.ToDouble(_posteriorCov[i, j]) * snm;
                }
            }
            _posteriorMean[i] = _numOps.FromDouble(mean);
        }
    }

    /// <summary>
    /// Computes the log marginal likelihood approximation.
    /// </summary>
    private double ComputeLogMarginalLikelihood()
    {
        // Simplified approximation for convergence monitoring
        if (_posteriorMean is null || _y is null || _posteriorCov is null) return 0;

        double logZ = 0;
        int n = _y.Length;
        for (int i = 0; i < n; i++)
        {
            double yi = _numOps.ToDouble(_y[i]);
            double mi = _numOps.ToDouble(_posteriorMean[i]);
            double vi = _numOps.ToDouble(_posteriorCov[i, i]);
            logZ += StudentTLogPdf(yi - mi, _nu, Math.Sqrt(vi + _scale * _scale));
        }
        return logZ;
    }

    /// <summary>
    /// Predicts the mean and variance for a single input point.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>Tuple of (mean, variance).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts for a single point with uncertainty.
    /// The prediction is robust to outliers in the training data.
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

        var (means, variances) = PredictBatch(XNew);
        return (means[0], variances[0]);
    }

    /// <summary>
    /// Updates the kernel function.
    /// </summary>
    /// <param name="kernel">The new kernel function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This changes the kernel and invalidates the trained model.
    /// You'll need to retrain after updating the kernel.
    /// </para>
    /// </remarks>
    public void UpdateKernel(IKernelFunction<T> kernel)
    {
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        // Note: Since _kernel is readonly, we throw to indicate the model needs recreation
        throw new NotSupportedException(
            "StudentTGaussianProcess does not support kernel updates after construction. " +
            "Create a new instance with the desired kernel.");
    }

    /// <summary>
    /// Makes predictions with uncertainty estimates for multiple points.
    /// </summary>
    /// <param name="XNew">New input points as rows.</param>
    /// <returns>Tuple of (mean predictions, predictive variances).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts the function value at new points.
    /// The predictions are based on a posterior that has automatically downweighted
    /// outliers, so they should be robust to bad training points.
    /// </para>
    /// </remarks>
    public (Vector<T> Mean, Vector<T> Variance) PredictBatch(Matrix<T> XNew)
    {
        if (!_isTrained)
            throw new InvalidOperationException("Model must be trained before prediction.");
        if (_X is null || _K is null || _posteriorMean is null || _posteriorCov is null)
            throw new InvalidOperationException("Model not properly initialized.");

        int n = _X.Rows;
        int nNew = XNew.Rows;

        // Compute cross-covariance
        var KStar = ComputeKernelMatrix(XNew, _X);

        // Compute K^{-1} * posterior_mean (using site parameters)
        var alpha = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            // Simplified: use posterior mean directly
            alpha[i] = _posteriorMean[i];
        }

        // Solve for weights
        var Kregularized = CopyMatrix(_K);
        for (int i = 0; i < n; i++)
        {
            double diag = _numOps.ToDouble(Kregularized[i, i]) + _jitter;
            Kregularized[i, i] = _numOps.FromDouble(diag);
        }
        var Lk = CholeskyDecomposition(Kregularized);
        var alphaWeights = SolveLowerTriangular(Lk, alpha);
        alphaWeights = SolveUpperTriangular(Transpose(Lk), alphaWeights);

        var mean = new Vector<T>(nNew);
        var variance = new Vector<T>(nNew);

        for (int i = 0; i < nNew; i++)
        {
            // Mean prediction
            double predMean = 0;
            for (int j = 0; j < n; j++)
            {
                predMean += _numOps.ToDouble(KStar[i, j]) * _numOps.ToDouble(alphaWeights[j]);
            }
            mean[i] = _numOps.FromDouble(predMean);

            // Variance prediction
            var xNew = GetRow(XNew, i);
            double kStarStar = _numOps.ToDouble(_kernel.Calculate(xNew, xNew));

            var v = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                v[j] = KStar[i, j];
            }
            var vSolved = SolveLowerTriangular(Lk, v);

            double vTv = 0;
            for (int j = 0; j < n; j++)
            {
                vTv += _numOps.ToDouble(vSolved[j]) * _numOps.ToDouble(vSolved[j]);
            }

            variance[i] = _numOps.FromDouble(Math.Max(kStarStar - vTv + _scale * _scale, 1e-10));
        }

        return (mean, variance);
    }

    #region Helper Methods

    /// <summary>
    /// Computes the kernel matrix between two sets of points.
    /// </summary>
    private Matrix<T> ComputeKernelMatrix(Matrix<T> X1, Matrix<T> X2)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);
        for (int i = 0; i < X1.Rows; i++)
        {
            var xi = GetRow(X1, i);
            for (int j = 0; j < X2.Rows; j++)
            {
                var xj = GetRow(X2, j);
                K[i, j] = _kernel.Calculate(xi, xj);
            }
        }
        return K;
    }

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
    /// Creates a copy of a matrix.
    /// </summary>
    private Matrix<T> CopyMatrix(Matrix<T> A)
    {
        var result = new Matrix<T>(A.Rows, A.Columns);
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = 0; j < A.Columns; j++)
            {
                result[i, j] = A[i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Performs Cholesky decomposition.
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
                    double ljj = _numOps.ToDouble(L[j, j]);
                    L[i, j] = _numOps.FromDouble((_numOps.ToDouble(A[i, j]) - sum) / ljj);
                }
            }
        }
        return L;
    }

    /// <summary>
    /// Computes matrix inverse from Cholesky factor.
    /// </summary>
    private Matrix<T> ComputeInverseFromCholesky(Matrix<T> L)
    {
        int n = L.Rows;
        var Inv = new Matrix<T>(n, n);

        // Solve L * L^T * Inv = I column by column
        for (int j = 0; j < n; j++)
        {
            var e = new Vector<T>(n);
            e[j] = _numOps.FromDouble(1.0);

            var z = SolveLowerTriangular(L, e);
            var col = SolveUpperTriangular(Transpose(L), z);

            for (int i = 0; i < n; i++)
            {
                Inv[i, j] = col[i];
            }
        }
        return Inv;
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

    /// <summary>
    /// Computes log gamma function.
    /// </summary>
    private static double LogGamma(double x)
    {
        // Stirling's approximation for large x
        if (x > 10)
        {
            return (x - 0.5) * Math.Log(x) - x + 0.5 * Math.Log(2 * Math.PI)
                   + 1.0 / (12 * x) - 1.0 / (360 * x * x * x);
        }

        // Lanczos approximation for smaller values
        double[] g = { 1.000000000000000174663, 5716.400188274341379136,
            -14815.30426768413909044, 14291.49277657478554025,
            -6348.160217641458813289, 1301.608286058321874105,
            -108.1767053514369634679, 2.605696505611755827729 };

        if (x < 0.5)
        {
            return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1 - x);
        }

        x -= 1;
        double sum = g[0];
        for (int i = 1; i < 8; i++)
        {
            sum += g[i] / (x + i);
        }

        double t = x + 7.5;
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }

    /// <summary>
    /// Gauss-Hermite quadrature nodes and weights.
    /// </summary>
    private static void GaussHermiteQuadrature(int n, double[] nodes, double[] weights)
    {
        // Precomputed values for n=20
        if (n == 20)
        {
            double[] preNodes = {
                -5.387480890011, -4.603682449550, -3.944764040115, -3.347854567384,
                -2.788806058428, -2.254974002089, -1.738537712116, -1.234076215395,
                -0.737473728545, -0.245340708301, 0.245340708301, 0.737473728545,
                1.234076215395, 1.738537712116, 2.254974002089, 2.788806058428,
                3.347854567384, 3.944764040115, 4.603682449550, 5.387480890011
            };
            double[] preWeights = {
                2.229393645534e-13, 4.399340992273e-10, 1.086069370769e-7, 7.802556478532e-6,
                2.283386360163e-4, 3.243773342238e-3, 2.481052088746e-2, 1.090172060200e-1,
                2.866755053628e-1, 4.622436696006e-1, 4.622436696006e-1, 2.866755053628e-1,
                1.090172060200e-1, 2.481052088746e-2, 3.243773342238e-3, 2.283386360163e-4,
                7.802556478532e-6, 1.086069370769e-7, 4.399340992273e-10, 2.229393645534e-13
            };
            Array.Copy(preNodes, nodes, n);
            Array.Copy(preWeights, weights, n);
            return;
        }

        // Fallback: simple approximation
        for (int i = 0; i < n; i++)
        {
            nodes[i] = -4 + 8.0 * i / (n - 1);
            weights[i] = Math.Exp(-nodes[i] * nodes[i]) / n * 8;
        }
    }

    #endregion
}
