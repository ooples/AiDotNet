namespace AiDotNet.GaussianProcesses;

/// <summary>
/// Gaussian Process with Markov Chain Monte Carlo inference for full Bayesian treatment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard GP makes point estimates of hyperparameters (kernel lengthscale,
/// output variance, noise variance). MCMC provides a fully Bayesian treatment by sampling from
/// the posterior distribution of hyperparameters, giving better uncertainty quantification.
///
/// Instead of finding a single "best" lengthscale, MCMC explores many plausible lengthscales
/// and averages predictions across all of them. This is more robust when:
/// - You have limited data
/// - The hyperparameters are uncertain
/// - You need accurate uncertainty estimates
///
/// The implementation uses Slice Sampling, which automatically adapts to the target distribution
/// without requiring careful tuning of step sizes.
/// </para>
/// </remarks>
public class GPWithMCMC<T> : IGaussianProcess<T>
{
    /// <summary>
    /// The base kernel function.
    /// </summary>
    private readonly IKernelFunction<T> _kernel;

    /// <summary>
    /// Training input data.
    /// </summary>
    private Matrix<T>? _X;

    /// <summary>
    /// Training output data.
    /// </summary>
    private Vector<T>? _y;

    /// <summary>
    /// MCMC samples of hyperparameters [lengthscale, outputVariance, noiseVariance].
    /// </summary>
    private List<double[]>? _samples;

    /// <summary>
    /// Precomputed Cholesky factors for each sample (for efficiency).
    /// </summary>
    private List<Matrix<T>>? _choleskyFactors;

    /// <summary>
    /// Precomputed alpha vectors for each sample.
    /// </summary>
    private List<Vector<T>>? _alphaVectors;

    /// <summary>
    /// Number of MCMC samples to use.
    /// </summary>
    private readonly int _numSamples;

    /// <summary>
    /// Number of burn-in samples to discard.
    /// </summary>
    private readonly int _burnIn;

    /// <summary>
    /// Thinning factor (keep every nth sample).
    /// </summary>
    private readonly int _thinning;

    /// <summary>
    /// Prior mean for log-lengthscale.
    /// </summary>
    private readonly double _logLengthscalePriorMean;

    /// <summary>
    /// Prior std for log-lengthscale.
    /// </summary>
    private readonly double _logLengthscalePriorStd;

    /// <summary>
    /// Prior mean for log-variance.
    /// </summary>
    private readonly double _logVariancePriorMean;

    /// <summary>
    /// Prior std for log-variance.
    /// </summary>
    private readonly double _logVariancePriorStd;

    /// <summary>
    /// Random generator for MCMC.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether the model has been trained.
    /// </summary>
    private bool _isTrained;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a GP with MCMC inference.
    /// </summary>
    /// <param name="kernel">The base kernel function.</param>
    /// <param name="numSamples">Number of MCMC samples to collect (default 500).</param>
    /// <param name="burnIn">Number of burn-in samples to discard (default 200).</param>
    /// <param name="thinning">Thinning factor - keep every nth sample (default 2).</param>
    /// <param name="logLengthscalePriorMean">Prior mean for log-lengthscale (default 0).</param>
    /// <param name="logLengthscalePriorStd">Prior std for log-lengthscale (default 1).</param>
    /// <param name="logVariancePriorMean">Prior mean for log-variance (default 0).</param>
    /// <param name="logVariancePriorStd">Prior std for log-variance (default 1).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a GP that will use MCMC to sample hyperparameters.
    ///
    /// Parameter guidance:
    /// - numSamples: More samples = better uncertainty, but slower. 500-1000 is typical.
    /// - burnIn: Samples to discard before chain "converges". 100-500 typical.
    /// - thinning: Reduces correlation between samples. 2-5 typical.
    ///
    /// The priors are on the LOG of the parameters:
    /// - log-lengthscale prior: N(0, 1) means lengthscale ~ LogNormal, with mode around 1
    /// - log-variance prior: N(0, 1) means variance ~ LogNormal, with mode around 1
    ///
    /// Adjust priors if you have domain knowledge about typical scales.
    /// </para>
    /// </remarks>
    public GPWithMCMC(
        IKernelFunction<T> kernel,
        int numSamples = 500,
        int burnIn = 200,
        int thinning = 2,
        double logLengthscalePriorMean = 0.0,
        double logLengthscalePriorStd = 1.0,
        double logVariancePriorMean = 0.0,
        double logVariancePriorStd = 1.0,
        int? seed = null)
    {
        _kernel = kernel ?? throw new ArgumentNullException(nameof(kernel));
        _numSamples = numSamples;
        _burnIn = burnIn;
        _thinning = thinning;
        _logLengthscalePriorMean = logLengthscalePriorMean;
        _logLengthscalePriorStd = logLengthscalePriorStd;
        _logVariancePriorMean = logVariancePriorMean;
        _logVariancePriorStd = logVariancePriorStd;

        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        _numOps = MathHelper.GetNumericOperations<T>();
        _isTrained = false;
    }

    /// <summary>
    /// Gets the kernel function.
    /// </summary>
    public IKernelFunction<T> Kernel => _kernel;

    /// <summary>
    /// Gets whether the GP is trained.
    /// </summary>
    public bool IsTrained => _isTrained;

    /// <summary>
    /// Gets the number of stored MCMC samples.
    /// </summary>
    public int NumStoredSamples => _samples?.Count ?? 0;

    /// <summary>
    /// Gets the MCMC samples [lengthscale, outputVariance, noiseVariance].
    /// </summary>
    /// <returns>Copy of the samples array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the MCMC samples for analysis.
    /// Each sample is [lengthscale, outputVariance, noiseVariance].
    ///
    /// You can use these to:
    /// - Check convergence (samples should be stable, not trending)
    /// - Compute posterior statistics (mean, std of each hyperparameter)
    /// - Diagnose mixing (autocorrelation should be low)
    /// </para>
    /// </remarks>
    public List<double[]> GetSamples()
    {
        if (_samples is null)
            throw new InvalidOperationException("Model not trained.");
        return _samples.Select(s => (double[])s.Clone()).ToList();
    }

    /// <summary>
    /// Fits the GP to training data using MCMC sampling.
    /// </summary>
    /// <param name="X">Training inputs (n × d).</param>
    /// <param name="y">Training outputs (n).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs the MCMC sampler to explore hyperparameter space.
    ///
    /// The algorithm:
    /// 1. Initialize hyperparameters
    /// 2. For each MCMC iteration:
    ///    a. Propose new hyperparameters (via slice sampling)
    ///    b. Compute log-posterior (likelihood + prior)
    ///    c. Accept/reject based on Metropolis criterion
    /// 3. Discard burn-in, thin remaining samples
    /// 4. Precompute Cholesky factors for each kept sample
    ///
    /// Slice sampling automatically adapts step sizes, making it robust.
    /// </para>
    /// </remarks>
    public void Fit(Matrix<T> X, Vector<T> y)
    {
        _X = X ?? throw new ArgumentNullException(nameof(X));
        _y = y ?? throw new ArgumentNullException(nameof(y));

        int n = X.Rows;
        if (y.Length != n)
            throw new ArgumentException("X and y must have same number of samples.");

        // Initialize hyperparameters in log-space
        double[] logParams = new double[3];
        logParams[0] = _logLengthscalePriorMean; // log-lengthscale
        logParams[1] = _logVariancePriorMean;    // log-outputVariance
        logParams[2] = Math.Log(0.01);           // log-noiseVariance (start small)

        // Run MCMC
        var allSamples = new List<double[]>();
        int totalIterations = _burnIn + _numSamples * _thinning;

        double currentLogPost = ComputeLogPosterior(logParams);

        for (int iter = 0; iter < totalIterations; iter++)
        {
            // Slice sample each parameter
            for (int p = 0; p < 3; p++)
            {
                (logParams[p], currentLogPost) = SliceSample(
                    logParams, p, currentLogPost);
            }

            // After burn-in, collect samples with thinning
            if (iter >= _burnIn && (iter - _burnIn) % _thinning == 0)
            {
                // Convert from log-space to actual values
                allSamples.Add(new double[]
                {
                    Math.Exp(logParams[0]),  // lengthscale
                    Math.Exp(logParams[1]),  // outputVariance
                    Math.Exp(logParams[2])   // noiseVariance
                });
            }
        }

        _samples = allSamples;

        // Precompute Cholesky factors and alpha vectors for efficiency
        PrecomputeForSamples();

        _isTrained = true;
    }

    /// <summary>
    /// Predicts at a single test point with full posterior averaging.
    /// </summary>
    /// <param name="x">Test input vector.</param>
    /// <returns>Tuple of (mean, variance) averaging over all MCMC samples.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Makes a prediction by averaging over all sampled hyperparameters.
    ///
    /// For each MCMC sample:
    /// 1. Compute GP prediction with those hyperparameters
    /// 2. Get mean and variance
    ///
    /// Final prediction:
    /// - Mean = average of means across samples
    /// - Variance = average of variances + variance of means (law of total variance)
    ///
    /// This properly accounts for uncertainty in both the function AND hyperparameters.
    /// </para>
    /// </remarks>
    public (T mean, T variance) Predict(Vector<T> x)
    {
        if (!_isTrained || _X is null || _y is null || _samples is null ||
            _choleskyFactors is null || _alphaVectors is null)
            throw new InvalidOperationException("Model must be trained first.");

        int numSamples = _samples.Count;
        double[] means = new double[numSamples];
        double[] variances = new double[numSamples];

        for (int s = 0; s < numSamples; s++)
        {
            var sample = _samples[s];
            double lengthscale = sample[0];
            double outputVar = sample[1];
            double noiseVar = sample[2];

            // Compute k(x, X) with this sample's parameters
            var kstar = ComputeCrossCovariance(x, lengthscale, outputVar);

            // Mean: k* · alpha
            double mean = 0;
            for (int i = 0; i < _X.Rows; i++)
            {
                mean += _numOps.ToDouble(kstar[i]) * _numOps.ToDouble(_alphaVectors[s][i]);
            }
            means[s] = mean;

            // Variance: k(x,x) - k*' · L^{-1} · L^{-T} · k*
            double kxx = outputVar + noiseVar;

            // Solve L · v = k*
            var v = SolveLowerTriangular(_choleskyFactors[s], kstar);
            double dotV = 0;
            for (int i = 0; i < v.Length; i++)
            {
                double vi = _numOps.ToDouble(v[i]);
                dotV += vi * vi;
            }
            variances[s] = Math.Max(kxx - dotV, 1e-10);
        }

        // Law of total variance: E[Var] + Var[E]
        double meanOfMeans = means.Average();
        double meanOfVariances = variances.Average();
        double varianceOfMeans = means.Select(m => (m - meanOfMeans) * (m - meanOfMeans)).Average();

        double totalVariance = meanOfVariances + varianceOfMeans;

        return (
            _numOps.FromDouble(meanOfMeans),
            _numOps.FromDouble(totalVariance)
        );
    }

    /// <summary>
    /// Predicts at multiple test points.
    /// </summary>
    /// <param name="Xtest">Test inputs (m × d).</param>
    /// <returns>Tuple of (means, variances) vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Batch prediction for efficiency.
    /// Applies Predict() to each row of Xtest.
    /// </para>
    /// </remarks>
    public (Vector<T> Means, Vector<T> Variances) PredictBatch(Matrix<T> Xtest)
    {
        if (!_isTrained)
            throw new InvalidOperationException("Model must be trained first.");

        int m = Xtest.Rows;
        var means = new Vector<T>(m);
        var variances = new Vector<T>(m);

        for (int i = 0; i < m; i++)
        {
            var xi = GetRow(Xtest, i);
            var (mean, variance) = Predict(xi);
            means[i] = mean;
            variances[i] = variance;
        }

        return (means, variances);
    }

    /// <summary>
    /// Updates the kernel (not supported for MCMC GP).
    /// </summary>
    /// <param name="newKernel">New kernel function.</param>
    /// <exception cref="NotSupportedException">Always thrown as kernel is fixed after initialization.</exception>
    public void UpdateKernel(IKernelFunction<T> newKernel)
    {
        throw new NotSupportedException(
            "GPWithMCMC does not support kernel updates after initialization. " +
            "Create a new instance with the desired kernel.");
    }

    /// <summary>
    /// Computes posterior statistics for hyperparameters.
    /// </summary>
    /// <returns>Dictionary with mean and std for each hyperparameter.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns summary statistics of the posterior.
    ///
    /// Useful for:
    /// - Reporting uncertainty in hyperparameters
    /// - Comparing with point estimates (should be similar if data is informative)
    /// - Checking if priors are dominating (means close to prior means = limited data)
    /// </para>
    /// </remarks>
    public Dictionary<string, (double Mean, double Std)> GetPosteriorStatistics()
    {
        if (_samples is null || _samples.Count == 0)
            throw new InvalidOperationException("Model not trained.");

        var lengthscales = _samples.Select(s => s[0]).ToArray();
        var outputVars = _samples.Select(s => s[1]).ToArray();
        var noiseVars = _samples.Select(s => s[2]).ToArray();

        return new Dictionary<string, (double, double)>
        {
            ["lengthscale"] = (lengthscales.Average(), StdDev(lengthscales)),
            ["outputVariance"] = (outputVars.Average(), StdDev(outputVars)),
            ["noiseVariance"] = (noiseVars.Average(), StdDev(noiseVars))
        };
    }

    #region Private Methods

    /// <summary>
    /// Computes log posterior = log likelihood + log prior.
    /// </summary>
    private double ComputeLogPosterior(double[] logParams)
    {
        if (_X is null || _y is null)
            return double.NegativeInfinity;

        double lengthscale = Math.Exp(logParams[0]);
        double outputVar = Math.Exp(logParams[1]);
        double noiseVar = Math.Exp(logParams[2]);

        int n = _X.Rows;

        // Build kernel matrix
        var K = BuildKernelMatrix(lengthscale, outputVar, noiseVar);

        // Cholesky decomposition
        Matrix<T> L;
        try
        {
            L = CholeskyDecomposition(K);
        }
        catch
        {
            return double.NegativeInfinity; // Invalid parameters
        }

        // Solve L · alpha = y for alpha (then L' · beta = alpha for beta, but we use K^-1·y directly)
        var alpha = SolveLowerTriangular(L, _y);
        var beta = SolveUpperTriangular(L, alpha);

        // Log likelihood: -0.5 · (y' · K^-1 · y + log|K| + n·log(2π))
        double yKinvY = 0;
        for (int i = 0; i < n; i++)
        {
            yKinvY += _numOps.ToDouble(_y[i]) * _numOps.ToDouble(beta[i]);
        }

        double logDetK = 0;
        for (int i = 0; i < n; i++)
        {
            logDetK += Math.Log(_numOps.ToDouble(L[i, i]));
        }
        logDetK *= 2; // log|K| = 2·sum(log(diag(L)))

        double logLik = -0.5 * (yKinvY + logDetK + n * Math.Log(2 * Math.PI));

        // Log prior (normal priors on log-parameters)
        double logPrior = 0;
        logPrior += LogNormalPdf(logParams[0], _logLengthscalePriorMean, _logLengthscalePriorStd);
        logPrior += LogNormalPdf(logParams[1], _logVariancePriorMean, _logVariancePriorStd);
        logPrior += LogNormalPdf(logParams[2], _logVariancePriorMean, _logVariancePriorStd);

        return logLik + logPrior;
    }

    /// <summary>
    /// Log of normal PDF.
    /// </summary>
    private static double LogNormalPdf(double x, double mean, double std)
    {
        double z = (x - mean) / std;
        return -0.5 * z * z - Math.Log(std) - 0.5 * Math.Log(2 * Math.PI);
    }

    /// <summary>
    /// Slice sampling for one parameter.
    /// </summary>
    private (double, double) SliceSample(double[] logParams, int paramIdx, double currentLogPost)
    {
        double x0 = logParams[paramIdx];
        double w = 1.0; // Initial width
        int maxSteps = 100;

        // Draw slice level
        double logY = currentLogPost + Math.Log(_random.NextDouble());

        // Find interval [L, R] containing slice
        double L = x0 - w * _random.NextDouble();
        double R = L + w;

        // Expand left
        logParams[paramIdx] = L;
        int leftSteps = 0;
        while (ComputeLogPosterior(logParams) > logY && leftSteps < maxSteps)
        {
            L -= w;
            logParams[paramIdx] = L;
            leftSteps++;
        }

        // Expand right
        logParams[paramIdx] = R;
        int rightSteps = 0;
        while (ComputeLogPosterior(logParams) > logY && rightSteps < maxSteps)
        {
            R += w;
            logParams[paramIdx] = R;
            rightSteps++;
        }

        // Shrink to find sample
        double xNew = x0;
        double newLogPost = currentLogPost;
        for (int iter = 0; iter < maxSteps; iter++)
        {
            xNew = L + _random.NextDouble() * (R - L);
            logParams[paramIdx] = xNew;
            newLogPost = ComputeLogPosterior(logParams);

            if (newLogPost > logY)
            {
                break; // Found valid sample
            }

            // Shrink interval
            if (xNew < x0)
                L = xNew;
            else
                R = xNew;
        }

        logParams[paramIdx] = xNew;
        return (xNew, newLogPost);
    }

    /// <summary>
    /// Precomputes Cholesky factors and alpha vectors for all samples.
    /// </summary>
    private void PrecomputeForSamples()
    {
        if (_X is null || _y is null || _samples is null)
            return;

        _choleskyFactors = new List<Matrix<T>>();
        _alphaVectors = new List<Vector<T>>();

        foreach (var sample in _samples)
        {
            double lengthscale = sample[0];
            double outputVar = sample[1];
            double noiseVar = sample[2];

            var K = BuildKernelMatrix(lengthscale, outputVar, noiseVar);
            var L = CholeskyDecomposition(K);
            var alpha = SolveLowerTriangular(L, _y);
            var beta = SolveUpperTriangular(L, alpha);

            _choleskyFactors.Add(L);
            _alphaVectors.Add(beta);
        }
    }

    /// <summary>
    /// Builds kernel matrix with given hyperparameters.
    /// </summary>
    private Matrix<T> BuildKernelMatrix(double lengthscale, double outputVar, double noiseVar)
    {
        int n = _X!.Rows;
        var K = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(_X, i);
            for (int j = i; j < n; j++)
            {
                var xj = GetRow(_X, j);

                // RBF kernel: outputVar * exp(-0.5 * ||xi - xj||^2 / lengthscale^2)
                double sqDist = 0;
                for (int d = 0; d < xi.Length; d++)
                {
                    double diff = _numOps.ToDouble(xi[d]) - _numOps.ToDouble(xj[d]);
                    sqDist += diff * diff;
                }

                double kval = outputVar * Math.Exp(-0.5 * sqDist / (lengthscale * lengthscale));

                if (i == j)
                    kval += noiseVar;

                K[i, j] = _numOps.FromDouble(kval);
                K[j, i] = _numOps.FromDouble(kval);
            }
        }

        return K;
    }

    /// <summary>
    /// Computes cross-covariance k(x, X).
    /// </summary>
    private Vector<T> ComputeCrossCovariance(Vector<T> x, double lengthscale, double outputVar)
    {
        int n = _X!.Rows;
        var kstar = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            var xi = GetRow(_X, i);
            double sqDist = 0;
            for (int d = 0; d < x.Length; d++)
            {
                double diff = _numOps.ToDouble(x[d]) - _numOps.ToDouble(xi[d]);
                sqDist += diff * diff;
            }
            double kval = outputVar * Math.Exp(-0.5 * sqDist / (lengthscale * lengthscale));
            kstar[i] = _numOps.FromDouble(kval);
        }

        return kstar;
    }

    /// <summary>
    /// Cholesky decomposition of a matrix.
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
                    double diag = _numOps.ToDouble(A[i, i]) - sum;
                    if (diag <= 0)
                        throw new InvalidOperationException("Matrix is not positive definite.");
                    L[i, j] = _numOps.FromDouble(Math.Sqrt(diag));
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
    /// Solves Lx = b for lower triangular L.
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
    /// Solves L'x = b for lower triangular L (i.e., solves upper triangular system).
    /// </summary>
    private Vector<T> SolveUpperTriangular(Matrix<T> L, Vector<T> b)
    {
        int n = b.Length;
        var x = new Vector<T>(n);

        for (int i = n - 1; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < n; j++)
            {
                sum += _numOps.ToDouble(L[j, i]) * _numOps.ToDouble(x[j]);
            }
            x[i] = _numOps.FromDouble((_numOps.ToDouble(b[i]) - sum) / _numOps.ToDouble(L[i, i]));
        }

        return x;
    }

    /// <summary>
    /// Extracts a row from a matrix.
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
    /// Computes standard deviation.
    /// </summary>
    private static double StdDev(double[] values)
    {
        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / values.Length);
    }

    #endregion
}
