using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("MCMC Methods for Gaussian Process Models", "https://doi.org/10.1007/978-3-540-28650-9_6", Year = 2003, Authors = "Mark N. Gibbs")]
public partial class GPWithMCMC<T> : GaussianProcessBase<T>
{
    /// <summary>
    /// The base kernel function.
    /// </summary>
    private readonly IKernelFunction<T> _kernel;

    /// <summary>
    /// Training input data.
    /// </summary>
    [Buffer]
    private Matrix<T>? _X;
    private Matrix<T> FittedX => _X ?? throw new InvalidOperationException("GP not fitted. Call Fit() first.");

    /// <summary>
    /// Training output data, standardized as <c>(y - mean(y)) / std(y)</c>. Both the
    /// mean (for centering) and the standard deviation (for unit-variance scaling) are
    /// removed so the zero-mean unit-variance GP prior is well-calibrated regardless of
    /// the original target scale; <see cref="Predict"/> reverses both transforms before
    /// returning predictions.
    /// </summary>
    [Buffer]
    private Vector<T>? _y;
    private Vector<T> FittedY => _y ?? throw new InvalidOperationException("GP not fitted. Call Fit() first.");

    /// <summary>
    /// Mean of the original training targets, used to de-center predictions.
    /// Standard GP practice: centering targets makes the zero-mean prior assumption
    /// valid regardless of target scale, ensuring scaling/translation equivariance.
    /// </summary>
    private T _yMean;

    /// <summary>
    /// Standard deviation of the original training targets for output variance scaling.
    /// </summary>
    private T _yStd;

    /// <summary>
    /// MCMC samples of hyperparameters [lengthscale, outputVariance, noiseVariance].
    /// </summary>
    private List<double[]>? _samples;

    /// <summary>
    /// Precomputed Cholesky factors for each sample (for efficiency).
    /// </summary>
    [Buffer]
    private List<Matrix<T>>? _choleskyFactors;

    /// <summary>
    /// Precomputed alpha vectors for each sample.
    /// </summary>
    [Buffer]
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
    /// The invariant base kernel matrix (raw <c>k(xi, xj)</c>, before the MCMC-sampled outputVar
    /// scaling and noise) plus the centered targets, cached as <see cref="double"/> once in
    /// <see cref="Fit"/>. The training data never changes across MCMC iterations — only outputVar and
    /// noiseVar do — so caching these lets <see cref="ComputeLogPosterior"/> (evaluated ~10^6 times by
    /// slice sampling) skip rebuilding the kernel from the generic <see cref="IKernelFunction{T}"/> and
    /// converting every element through <see cref="INumericOperations{T}"/> on each call.
    /// <para>
    /// Cache-validity invariant: these caches never go stale because everything they derive from is
    /// immutable after <see cref="Fit"/>. <c>_kernel</c> is <c>readonly</c> (assigned only in the
    /// constructor), <see cref="UpdateKernel"/> throws <see cref="NotSupportedException"/> (the kernel is
    /// fixed for the model's lifetime), and the training data (<c>_X</c>/<c>_y</c>) is set only inside
    /// <see cref="Fit"/> — which rebuilds ALL of these caches together in the same call. There is thus no
    /// code path that mutates the kernel or training state without rebuilding the caches, so no explicit
    /// invalidation hook is required. If a future change makes the kernel or training data mutable, that
    /// change MUST rebuild (or null) these caches.
    /// </para>
    /// </summary>
    private Tensor<T>? _baseKernel;
    private Tensor<T>? _centeredTargets;
    private Tensor<T>? _identity;

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
        Guard.NotNull(kernel);
        _kernel = kernel;
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
        _yMean = _numOps.Zero;
        _yStd = _numOps.One;
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

        // Rescale variance components from standardized-target units back to the
        // original target scale. _samples holds [lengthscale, outputVariance, noiseVariance]
        // where the variance entries were sampled against y / _yStd, so on the original
        // scale they each get multiplied by _yStd^2. Lengthscale stays unchanged because
        // it scales with the input X, not the target y. Without this rescale, callers of
        // GetSamples() would see variances under-reported by a factor of _yStd^2 — exactly
        // the discrepancy flagged in PR review.
        // The hyperparameter samples are reported as doubles; the target scale crosses that boundary once.
        double scale = _numOps.ToDouble(_numOps.Multiply(_yStd, _yStd));
        return _samples
            .Select(s => new[] { s[0], s[1] * scale, s[2] * scale })
            .ToList();
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
    public override void Fit(Matrix<T> X, Vector<T> y)
    {
        Guard.NotNull(X);
        _X = X;
        Guard.NotNull(y);

        int n = X.Rows;
        if (n == 0)
            throw new ArgumentException("Cannot fit a Gaussian Process on an empty training set (X.Rows == 0).", nameof(X));
        if (y.Length != n)
            throw new ArgumentException("X and y must have same number of samples.");

        // Center targets: standard GP practice to make zero-mean prior valid.
        // This ensures scaling/translation equivariance regardless of MCMC convergence.
        var engine = Engine;
        T yMean = engine.Mean(y);
        var deviations = Tensor<T>.FromVector(y);
        deviations = engine.TensorSubtractScalar(deviations, yMean);
        T yStd = MathHelper.Max(
            _numOps.Sqrt(_numOps.Divide(engine.TensorSumOfSquares(deviations), _numOps.FromDouble(Math.Max(n - 1, 1)))),
            _numOps.FromDouble(1e-10));

        _yMean = yMean;
        _yStd = yStd;

        // Store centered targets
        _y = engine.TensorDivideScalar(deviations, yStd).ToVector();

        // Precompute the invariant base kernel structure + centered targets for the MCMC hot loop.
        // k(xi, xj) does not depend on the sampled outputVar/noiseVar (ComputeLogPosterior scales by
        // outputVar and adds noiseVar to the diagonal), so it is constant across every evaluation —
        // computing it once here removes ~n^2/2 generic kernel calls from each slice-sampling evaluation.
        _centeredTargets = Tensor<T>.FromVector(_y).Reshape(new[] { n, 1 });
        // Base kernel matrix K_base[a, b] = k(x_a, x_b) over the training inputs (symmetric).
        var kernelMatrix = new Matrix<T>(n, n);
        for (int a = 0; a < n; a++)
        {
            var xa = GetRow(X, a);
            for (int b = a; b < n; b++)
            {
                T kval = _kernel.Calculate(xa, GetRow(X, b));
                kernelMatrix[a, b] = kval;
                kernelMatrix[b, a] = kval;
            }
        }

        _baseKernel = Tensor<T>.FromMatrix(kernelMatrix);
        _identity = engine.TensorEye<T>(n);

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
    public override (T mean, T variance) Predict(Vector<T> x)
    {
        if (!_isTrained || _X is null || _y is null || _samples is null ||
            _choleskyFactors is null || _alphaVectors is null)
            throw new InvalidOperationException("Model must be trained first.");

        // All posterior samples at once: sample s has k*_s = outputVar_s · k(x, X), mean k*_s · alpha_s and
        // variance outputVar_s + noiseVar_s - ||L_s^{-1} k*_s||².
        var engine = Engine;
        int numSamples = _samples.Count;
        int n = _X.Rows;
        var outputVars = Tensor<T>.FromVector(
            GaussianProcessEngineOps.FromDoubles<T>(_samples.Select(sample => sample[1]).ToArray()));
        var noiseVars = Tensor<T>.FromVector(
            GaussianProcessEngineOps.FromDoubles<T>(_samples.Select(sample => sample[2]).ToArray()));

        // Base cross-covariance k(x, x_i) over the training inputs
        var kernelCross = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            kernelCross[i] = _kernel.Calculate(x, GetRow(FittedX, i));
        }

        var baseCross = Tensor<T>.FromVector(kernelCross);
        var kstars = engine.TensorMultiply(
            engine.TensorBroadcastTo(outputVars.Reshape(new[] { numSamples, 1 }), new[] { numSamples, n }),
            engine.TensorBroadcastTo(baseCross.Reshape(new[] { 1, n }), new[] { numSamples, n }));

        // Mean: k*_s · alpha_s
        var alphas = engine.TensorStack(_alphaVectors.Select(alpha => Tensor<T>.FromVector(alpha)).ToArray(), 0);
        var means = engine.ReduceSum(engine.TensorMultiply(kstars, alphas), new[] { 1 }, keepDims: false);

        // Variance: k(x,x) - k*' · L^{-1} · L^{-T} · k*, with L_s · v_s = k*_s solved for every sample
        var factors = engine.TensorStack(_choleskyFactors.Select(factor => Tensor<T>.FromMatrix(factor)).ToArray(), 0);
        var v = LinalgDispatch<T>.SolveTriangular(factors, kstars.Reshape(new[] { numSamples, n, 1 }), upper: false);
        var dotV = engine.ReduceSum(engine.TensorSquare(v), new[] { 1, 2 }, keepDims: false);
        var variances = engine.TensorMax(
            engine.TensorSubtract(engine.TensorAdd(outputVars, noiseVars), dotV),
            _numOps.FromDouble(1e-10));

        // Law of total variance: E[Var] + Var[E]
        T meanOfMeans = engine.TensorMean(means);
        T meanOfVariances = engine.TensorMean(variances);
        T varianceOfMeans = engine.TensorMean(engine.TensorSquare(engine.TensorSubtractScalar(means, meanOfMeans)));
        T totalVariance = _numOps.Add(meanOfVariances, varianceOfMeans);

        // De-center: transform from standardized space back to original scale
        T yStd = _yStd;
        T deCenteredMean = _numOps.Add(_numOps.Multiply(meanOfMeans, yStd), _yMean);
        T deCenteredVariance = _numOps.Multiply(totalVariance, _numOps.Multiply(yStd, yStd));

        return (deCenteredMean, deCenteredVariance);
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
    public override void UpdateKernel(IKernelFunction<T> newKernel)
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

        // Rescale variance components back to the original target scale (see GetSamples()
        // for the full explanation — _samples are stored in standardized units).
        // The hyperparameter samples are reported as doubles; the target scale crosses that boundary once.
        double scale = _numOps.ToDouble(_numOps.Multiply(_yStd, _yStd));
        var lengthscales = _samples.Select(s => s[0]).ToArray();
        var outputVars = _samples.Select(s => s[1] * scale).ToArray();
        var noiseVars = _samples.Select(s => s[2] * scale).ToArray();

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
        if (_baseKernel is null || _centeredTargets is null || _identity is null)
            return double.NegativeInfinity;

        // logParams[0] (lengthscale) is intentionally not applied to the kernel matrix here: the generic
        // _kernel.Calculate(…) has its lengthscale fixed at kernel construction, so the sampled lengthscale
        // has always been vestigial for a generic IKernelFunction — it feeds only the prior below (and is
        // reported by GetPosteriorStatistics), never the training kernel or the predictive cross-covariance.
        // Making it a live kernel hyperparameter needs a re-parameterizable kernel and is a separate change.
        var engine = Engine;
        T outputVar = _numOps.FromDouble(Math.Exp(logParams[1]));
        T noiseVar = _numOps.FromDouble(Math.Exp(logParams[2]));

        // Cholesky of K = base·outputVar + noiseVar·I; a matrix that is not positive definite makes the
        // parameters invalid.
        var K = engine.TensorAdd(
            engine.TensorMultiplyScalar(_baseKernel, outputVar), engine.TensorMultiplyScalar(_identity, noiseVar));
        var lower = GaussianProcessEvidence.Cholesky(K);
        if (lower is null)
            return double.NegativeInfinity;

        // Log likelihood: -0.5 · (y' · K^-1 · y + log|K| + n·log(2π)). The slice sampler compares log
        // densities of its scalar log-space hyperparameters, so the evidence is read as one scalar here.
        double logLik = _numOps.ToDouble(GaussianProcessEvidence.LogMarginalLikelihood(lower, _centeredTargets));

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

        // K_s = outputVar_s · K_base + noiseVar_s · I for every sample, factored and solved as one batch.
        var engine = Engine;
        int numSamples = _samples.Count;
        int n = FittedX.Rows;
        var outputVars = Tensor<T>.FromVector(
            GaussianProcessEngineOps.FromDoubles<T>(_samples.Select(sample => sample[1]).ToArray()));
        var noiseVars = Tensor<T>.FromVector(
            GaussianProcessEngineOps.FromDoubles<T>(_samples.Select(sample => sample[2]).ToArray()));
        var batchShape = new[] { numSamples, n, n };
        var cachedKernel = _baseKernel ?? throw new InvalidOperationException("Base kernel has not been computed.");
        var baseKernel = cachedKernel.Reshape(new[] { 1, n, n });
        var identity = engine.TensorEye<T>(n).Reshape(new[] { 1, n, n });
        var kernels = engine.TensorAdd(
            engine.TensorMultiply(
                engine.TensorBroadcastTo(baseKernel, batchShape),
                engine.TensorBroadcastTo(outputVars.Reshape(new[] { numSamples, 1, 1 }), batchShape)),
            engine.TensorMultiply(
                engine.TensorBroadcastTo(identity, batchShape),
                engine.TensorBroadcastTo(noiseVars.Reshape(new[] { numSamples, 1, 1 }), batchShape)));

        var (factors, info) = LinalgDispatch<T>.CholeskyEx(kernels, upper: false);
        if (engine.TensorCountNonzero(info) > 0)
            throw new InvalidOperationException("Matrix is not positive definite.");

        var targets = engine.TensorBroadcastTo(
            Tensor<T>.FromVector(_y).Reshape(new[] { 1, n, 1 }), new[] { numSamples, n, 1 });
        var alphas = LinalgDispatch<T>.CholeskySolve(targets, factors, upper: false);

        _choleskyFactors = Enumerable.Range(0, numSamples)
            .Select(s => engine.TensorSlice(factors, new[] { s, 0, 0 }, new[] { 1, n, n }).Reshape(new[] { n, n }).ToMatrix())
            .ToList();
        _alphaVectors = Enumerable.Range(0, numSamples)
            .Select(s => engine.TensorSlice(alphas, new[] { s, 0, 0 }, new[] { 1, n, 1 }).Reshape(new[] { n }).ToVector())
            .ToList();
    }

    /// <summary>
    /// Extracts a row from a matrix.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        return matrix.GetRow(row);
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
