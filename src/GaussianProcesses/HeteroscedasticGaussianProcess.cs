using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Most Likely Heteroscedastic Gaussian Process Regression", "https://doi.org/10.1145/1273496.1273546", Year = 2007, Authors = "Kristian Kersting, Christian Plagemann, Patrick Pfaff, Wolfram Burgard")]
public partial class HeteroscedasticGaussianProcess<T> : GaussianProcessBase<T>
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
    [Buffer]
    private Matrix<T>? _X;

    /// <summary>
    /// Training target values.
    /// </summary>
    [Buffer]
    private Vector<T>? _y;

    /// <summary>
    /// Cholesky decomposition of mean GP covariance matrix.
    /// </summary>
    [Buffer]
    private Matrix<T>? _L;

    /// <summary>
    /// Alpha coefficients for mean predictions.
    /// </summary>
    [Buffer]
    private Vector<T>? _alpha;

    /// <summary>
    /// Learned noise variances at training points.
    /// </summary>
    [Buffer]
    private Vector<T>? _noiseVariances;

    /// <summary>
    /// Noise GP latent values (log noise variance).
    /// </summary>
    [Buffer]
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
    private T _yMean;

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
        _yMean = _numOps.Zero;
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
    public Vector<T>? GetNoiseVariances()
    {
        return _noiseVariances?.Clone();
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
    public override void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of rows in X must match length of y.");

        _X = X;
        int n = X.Rows;

        // Center y by subtracting the mean. GP assumes zero-mean prior,
        // so large y offsets cause poor conditioning. Mean is added back in Predict.
        var engine = AiDotNetEngine.Current;
        var targets = Tensor<T>.FromVector(y);
        _yMean = engine.TensorMean(targets);
        var centered = engine.TensorSubtractScalar(targets, _yMean);
        _y = centered.ToVector();

        // Initialize noise latent values by estimating from data variance.
        // Using the prior mean (log(0.1)) is too large for low-noise data and causes
        // under-fitting in the first EM iteration that's hard to recover from.
        // Instead, estimate initial noise from the target variance as a fraction.
        T yVar = _numOps.Divide(engine.TensorSumOfSquares(centered), _numOps.FromDouble(n));
        // Start with noise = 0.01% of target variance. A small initial noise lets the
        // mean GP fit tightly in the first EM iteration, then the E-step adjusts noise
        // upward where the data actually has high variance. Starting too large causes
        // under-fitting that the EM algorithm struggles to recover from.
        T initialNoiseLog = _numOps.Log(MathHelper.Max(
            _numOps.Multiply(yVar, _numOps.FromDouble(0.0001)), _numOps.FromDouble(1e-10)));

        _noiseLatentValues = Tensor<T>.CreateDefault(new[] { n }, initialNoiseLog).ToVector();

        // Initialize noise variances
        _noiseVariances = new Vector<T>(n);
        UpdateNoiseVariances();

        // Phase 1: Fit initial mean GP with constant small noise to get good residuals.
        // This breaks the EM feedback loop where high initial noise -> under-fit mean GP ->
        // large residuals -> noise GP estimates even higher noise.
        _noiseVariances = Tensor<T>.CreateDefault(new[] { n }, _numOps.FromDouble(_jitter)).ToVector();
        FitMeanGP();

        // Phase 2: EM refinement — alternate between fitting mean GP and updating noise.
        // The noise variance for each point is estimated from the GP residuals using a
        // secondary noise GP. This implements the Goldberg et al. (1998) algorithm.
        double prevLoss = double.MaxValue;
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // E-step: Update noise estimates given current mean GP
            UpdateNoiseLatentValues();
            UpdateNoiseVariances();

            // M-step: Re-fit mean GP with updated noise
            FitMeanGP();

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

        _noiseVariances = AiDotNetEngine.Current.Exp(_noiseLatentValues);
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

        // Kernel matrix with heteroscedastic noise + jitter on the diagonal
        var K = GaussianProcessEngineOps.AddToDiagonal(
            GaussianProcessEngineOps.AddToDiagonal(KernelMatrix(_meanKernel, _X, _X), _noiseVariances),
            _numOps.FromDouble(_jitter));

        // Adaptive jitter for numerical stability - same approach as StandardGaussianProcess.
        // Without this, the heteroscedastic noise on the diagonal can make the matrix
        // ill-conditioned, causing Cholesky to produce NaN or large alpha values.
        GaussianProcessEngineOps.AddAdaptiveJitter(K);

        // Cholesky decomposition (pivots floored at 1e-10)
        _L = Factor(K).ToMatrix();

        // Solve K * alpha = y using the properly-conditioned matrix
        _alpha = MatrixSolutionHelper.SolveLinearSystem(K, _y, Enums.MatrixDecompositionType.Cholesky);
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

        var engine = AiDotNetEngine.Current;
        int n = _X.Rows;

        // Residuals of the mean GP at the training inputs
        var predictions = engine.MatrixVectorMultiply(KernelMatrix(_meanKernel, _X, _X), _alpha);
        var residuals = Tensor<T>.FromVector(engine.Subtract(_y, predictions));

        // Target for noise GP: log of squared residuals,
        // floored at -23 (about log(1e-10)) to prevent -inf while allowing very low noise
        var logResidualsSq = engine.TensorMax(
            engine.TensorLog(engine.TensorAddScalar(engine.TensorSquare(residuals), _numOps.FromDouble(1e-15))),
            _numOps.FromDouble(-23.0));

        // Solve noise GP
        var rawNoiseKernel = KernelMatrix(_noiseKernel, _X, _X);
        var L_noise = Factor(GaussianProcessEngineOps.AddToDiagonal(
            rawNoiseKernel, _numOps.FromDouble(_noisePriorVariance + _jitter)));
        var alpha_noise = LinalgDispatch<T>.CholeskySolve(logResidualsSq.Reshape(new[] { n, 1 }), L_noise, upper: false);

        // Update noise latent values.
        // Clamp to prevent extreme values but don't blend with prior
        // (prior blending prevents convergence to the true noise level)
        var newLatent = engine.TensorMatMul(Tensor<T>.FromMatrix(rawNoiseKernel), alpha_noise).Reshape(new[] { n });
        newLatent = engine.TensorMin(
            engine.TensorMax(newLatent, _numOps.FromDouble(_noisePriorMean - 10)),
            _numOps.FromDouble(_noisePriorMean + 10));
        _noiseLatentValues = newLatent.ToVector();
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

        var engine = AiDotNetEngine.Current;
        int n = _y.Length;

        // Data fit term: 0.5 * y^T * K^{-1} * y = 0.5 * y^T * alpha
        T dataFit = _numOps.Multiply(_numOps.FromDouble(0.5), engine.DotProduct(_y, _alpha));

        // Complexity penalty: 0.5 * log|K| = sum(log(L_ii))
        var diagonal = Tensor<T>.FromVector(GaussianProcessEngineOps.Diagonal(_L));
        T logDet = engine.TensorSum(engine.TensorLog(engine.TensorMax(diagonal, _numOps.FromDouble(1e-10))));

        // Constant term
        T constant = _numOps.FromDouble(0.5 * n * Math.Log(2 * Math.PI));

        // The EM convergence check compares scalar losses against the double tolerance.
        return _numOps.ToDouble(_numOps.Add(_numOps.Add(dataFit, logDet), constant));
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
    public (Vector<T> Mean, Vector<T> PredictiveVariance, Vector<T> NoiseVariance) PredictWithUncertainty(Matrix<T> XNew)
    {
        if (!_isTrained)
            throw new InvalidOperationException("Model must be trained before prediction.");
        if (_X is null || _alpha is null || _L is null || _noiseLatentValues is null)
            throw new InvalidOperationException("Model not properly initialized.");

        var engine = AiDotNetEngine.Current;
        int n = _X.Rows;
        int nNew = XNew.Rows;

        // For noise GP prediction
        var L_noise = Factor(GaussianProcessEngineOps.AddToDiagonal(
            KernelMatrix(_noiseKernel, _X, _X), _numOps.FromDouble(_noisePriorVariance + _jitter)));
        var alpha_noise = LinalgDispatch<T>.CholeskySolve(
            Tensor<T>.FromVector(_noiseLatentValues).Reshape(new[] { n, 1 }), L_noise, upper: false);

        // k_* rows: covariance between each new point and the training points, [nNew, n]
        var kStar = Tensor<T>.FromMatrix(KernelMatrix(_meanKernel, XNew, _X));
        var kStarNoise = Tensor<T>.FromMatrix(KernelMatrix(_noiseKernel, XNew, _X));

        // Mean prediction: k_*^T * alpha + yMean (add back centered mean)
        var alpha = Tensor<T>.FromVector(_alpha).Reshape(new[] { n, 1 });
        var mean = engine.TensorAddScalar(engine.TensorMatMul(kStar, alpha).Reshape(new[] { nNew }), _yMean);

        // Predictive variance: k_** - k_*^T * K^{-1} * k_* = k_** - ||L^{-1} k_*||^2
        var v = LinalgDispatch<T>.SolveTriangular(Tensor<T>.FromMatrix(_L), engine.TensorTranspose(kStar), upper: false);
        var vTv = engine.ReduceSum(engine.TensorSquare(v), new[] { 0 }, keepDims: false);
        var kStarStar = Tensor<T>.FromVector(KernelDiagonal(_meanKernel, XNew));
        var predictiveVar = engine.TensorMax(engine.TensorSubtract(kStarStar, vTv), _numOps.FromDouble(1e-10));

        // Noise variance prediction
        var noiseVar = engine.TensorExp(engine.TensorMatMul(kStarNoise, alpha_noise).Reshape(new[] { nNew }));

        return (mean.ToVector(), predictiveVar.ToVector(), noiseVar.ToVector());
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
    public override (T mean, T variance) Predict(Vector<T> x)
    {
        // Convert single point to matrix
        var XNew = Tensor<T>.FromVector(x).Reshape(new[] { 1, x.Length }).ToMatrix();

        var (means, predVars, _) = PredictWithUncertainty(XNew);
        // Return PREDICTIVE (epistemic) variance only, not total variance.
        // The IGaussianProcess interface variance represents model uncertainty about the
        // function value, not observation noise. Epistemic variance should be low near
        // training data and high far away (posterior contraction property).
        // Use Predict(Matrix) to get both predictive and noise variances separately.
        return (means[0], predVars[0]);
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
    public override void UpdateKernel(IKernelFunction<T> kernel)
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
        var (_, predVar, noiseVar) = PredictWithUncertainty(XNew);

        return AiDotNetEngine.Current.Add(predVar, noiseVar);
    }

    #region Linear Algebra Helpers

    /// <summary>
    /// Kernel matrix k(a_i, b_j) between the rows of <paramref name="A"/> and <paramref name="B"/>.
    /// </summary>
    private static Matrix<T> KernelMatrix(IKernelFunction<T> kernel, Matrix<T> A, Matrix<T> B)
    {
        var K = new Matrix<T>(A.Rows, B.Rows);
        for (int i = 0; i < A.Rows; i++)
        {
            var ai = A.GetRow(i);
            for (int j = 0; j < B.Rows; j++)
            {
                K[i, j] = kernel.Calculate(ai, B.GetRow(j));
            }
        }

        return K;
    }

    /// <summary>
    /// Kernel self-similarities k(x_i, x_i) for every row of <paramref name="X"/>.
    /// </summary>
    private static Vector<T> KernelDiagonal(IKernelFunction<T> kernel, Matrix<T> X)
    {
        var diagonal = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            var xi = X.GetRow(i);
            diagonal[i] = kernel.Calculate(xi, xi);
        }

        return diagonal;
    }

    /// <summary>
    /// Lower Cholesky factor with every pivot floored at 1e-10, so a kernel matrix that is not
    /// positive definite still yields a usable factor.
    /// </summary>
    private static Tensor<T> Factor(Matrix<T> K)
    {
        return GaussianProcessEngineOps.ClampedCholesky(K, 1e-10);
    }

    #endregion
}
