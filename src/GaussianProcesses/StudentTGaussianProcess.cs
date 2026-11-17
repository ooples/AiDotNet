using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.GaussianProcess)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Gaussian Process Regression with Student-t Likelihood", "https://doi.org/10.5555/2986459.2986589", Year = 2011, Authors = "Jarno Vanhatalo, Pasi Jylänki, Aki Vehtari")]
public partial class StudentTGaussianProcess<T> : GaussianProcessBase<T>
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
    [Buffer]
    private Matrix<T>? _X;

    /// <summary>
    /// Training target values.
    /// </summary>
    [Buffer]
    private Vector<T>? _y;
    private Vector<T> FittedY => _y ?? throw new InvalidOperationException("GP not fitted. Call Fit() first.");

    /// <summary>
    /// Prior covariance matrix.
    /// </summary>
    [Buffer]
    private Matrix<T>? _K;

    /// <summary>
    /// Approximate posterior mean.
    /// </summary>
    [Buffer]
    private Vector<T>? _posteriorMean;

    /// <summary>
    /// Approximate posterior covariance.
    /// </summary>
    [Buffer]
    private Matrix<T>? _posteriorCov;

    /// <summary>
    /// Site natural parameters (precision).
    /// </summary>
    [Buffer]
    private Vector<T>? _sitePrecisions;

    /// <summary>
    /// Site natural parameters (precision * mean).
    /// </summary>
    [Buffer]
    private Vector<T>? _siteNaturalMeans;

    /// <summary>
    /// Outlier weights (downweights outliers).
    /// </summary>
    [Buffer]
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
    /// The scale parameter sets the Student-t likelihood width; its variance is nu * scale² / (nu - 2) for nu > 2.
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
        if (!(nu > 0) || double.IsInfinity(nu))
            throw new ArgumentException("Degrees of freedom must be finite and positive.", nameof(nu));
        if (!(scale > 0) || double.IsInfinity(scale))
            throw new ArgumentException("Scale must be finite and positive.", nameof(scale));
        if (!(damping > 0 && damping <= 1))
            throw new ArgumentException("Damping must be in (0, 1].", nameof(damping));
        if (maxIterations <= 0)
            throw new ArgumentException("Maximum EP iterations must be positive.", nameof(maxIterations));
        if (!(tolerance > 0) || double.IsInfinity(tolerance))
            throw new ArgumentException("Convergence tolerance must be finite and positive.", nameof(tolerance));
        if (!(jitter >= 0) || double.IsInfinity(jitter))
            throw new ArgumentException("Jitter must be finite and nonnegative.", nameof(jitter));

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
    /// <returns>Vector of weights in [0,1], where lower values indicate outliers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After training, this shows how much each training point
    /// is trusted. Values close to 1 are normal points; values close to 0 are likely outliers.
    /// This can help identify problematic data points in your dataset.
    /// </para>
    /// </remarks>
    public Vector<T>? GetOutlierWeights()
    {
        if (_weights is null) return null;

        return _weights.Clone();
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

        var engine = Engine;
        var belowThreshold = engine.TensorLessThan(Tensor<T>.FromVector(_weights), _numOps.FromDouble(threshold));
        return engine.TensorNonzero(belowThreshold).AsSpan().ToArray();
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
    /// <exception cref="InvalidOperationException">
    /// EP cannot form a proper posterior or cannot reach the configured fixed-point tolerance
    /// within the iteration budget. A failed fit is not available for prediction.
    /// </exception>
    public override void Fit(Matrix<T> X, Vector<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (X.Rows != y.Length)
            throw new ArgumentException("Number of rows in X must match length of y.");
        if (X.Rows == 0) throw new ArgumentException("At least one observation is required.", nameof(X));
        if (!GaussianProcessEngineOps.AllFinite(y))
            throw new ArgumentException("Training targets must be finite.", nameof(y));
        _isTrained = false;

        _X = X;
        _y = y;
        int n = X.Rows;

        // Compute prior covariance matrix
        _K = ComputeKernelMatrix(X, X);

        // Zero sites initialize exactly to the declared zero-mean prior.
        _sitePrecisions = new Vector<T>(n);
        _siteNaturalMeans = new Vector<T>(n);
        _weights = Tensor<T>.CreateDefault(new[] { n }, _numOps.One).ToVector();
        UpdatePosterior();
        bool converged = false;
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var oldPrecision = _sitePrecisions.Clone();
            var oldNaturalMean = _siteNaturalMeans.Clone();
            var oldWeights = _weights.Clone();
            var oldMean = _posteriorMean!.Clone();
            var oldCov = _posteriorCov!.Clone();
            // The scalar tilted integrals are per-observation tasks; their
            // undamped site proposals are compared and reduced on the engine.
            var sites = Enumerable.Range(0, n).Select(UpdateSite).ToArray();
            var proposedSites = new Tensor<double>(new[] { n, 2 }, new Vector<double>(
                sites.SelectMany(site => new[] { site.Precision, site.NaturalMean }).ToArray()));
            var oldSites = Engine.TensorStack(new[]
            {
                Tensor<T>.FromVector(oldPrecision).Cast<double>(),
                Tensor<T>.FromVector(oldNaturalMean).Cast<double>(),
            }, 1);
            var siteResiduals = NormalizedDifference(oldSites, proposedSites);
            double maxChange = Engine.TensorMaxValue(Engine.ReduceSum(siteResiduals, new[] { 1 }, false));
            var proposedPrecision = _sitePrecisions.Clone();
            var proposedNaturalMean = _siteNaturalMeans.Clone();
            var proposedWeights = _weights.Clone();
            // Negative Student-t sites are valid. Backtrack an update only if
            // the combined Gaussian or a leave-one-out cavity is improper.
            double fraction = 1;
            while (true)
            {
                try
                {
                    UpdatePosterior();
                    var variances = Engine.TensorDiagonal(Tensor<T>.FromMatrix(_posteriorCov!)).Cast<double>();
                    if (!GaussianProcessEngineOps.AllFinite(variances) || Engine.TensorMinValue(variances) <= 0)
                        throw new InvalidOperationException("EP update produced an improper posterior variance.");
                    var cavities = Engine.TensorSubtract(Engine.TensorReciprocal(variances),
                        Tensor<T>.FromVector(_sitePrecisions).Cast<double>());
                    if (!GaussianProcessEngineOps.AllFinite(cavities) || Engine.TensorMinValue(cavities) <= 0)
                        throw new InvalidOperationException("EP update produced an improper cavity.");
                    break;
                }
                catch (InvalidOperationException)
                {
                    fraction *= 0.5;
                    if (fraction < 1e-8)
                        throw new InvalidOperationException("EP could not find a proper damped posterior and cavities.");
                    _sitePrecisions = Blend(oldPrecision, proposedPrecision, fraction);
                    _siteNaturalMeans = Blend(oldNaturalMean, proposedNaturalMean, fraction);
                    _weights = Blend(oldWeights, proposedWeights, fraction);
                }
            }
            // Likelihood values can barely change while the moments still
            // move. Require both site and posterior fixed-point residuals.
            double posteriorChange = Math.Max(
                Engine.TensorMaxValue(NormalizedDifference(Tensor<T>.FromVector(oldMean).Cast<double>(),
                    Tensor<T>.FromVector(_posteriorMean!).Cast<double>())),
                Engine.TensorMaxValue(NormalizedDifference(Tensor<T>.FromMatrix(oldCov).Cast<double>(),
                    Tensor<T>.FromMatrix(_posteriorCov!).Cast<double>())));
            if (maxChange < _tolerance && posteriorChange < _tolerance)
            {
                converged = true;
                break;
            }
        }
        if (!converged)
            throw new InvalidOperationException($"Student-t EP did not converge within {_maxIterations} iterations.");
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
    private (double Precision, double NaturalMean) UpdateSite(int i)
    {
        if (_posteriorMean == null || _posteriorCov == null ||
            _sitePrecisions == null || _siteNaturalMeans == null || _y == null || _weights == null)
            throw new InvalidOperationException("EP site state is not initialized.");
        var y = _y;

        // Cavity distribution (posterior without site i)
        double postPrecision = 1.0 / _numOps.ToDouble(_posteriorCov[i, i]);
        double sitePrecision = _numOps.ToDouble(_sitePrecisions[i]);
        double cavityPrecision = postPrecision - sitePrecision;

        if (!(cavityPrecision > 0) || double.IsInfinity(cavityPrecision))
            throw new InvalidOperationException("EP requires a finite positive cavity precision.");

        double postMean = _numOps.ToDouble(_posteriorMean[i]);
        double siteNaturalMean = _numOps.ToDouble(_siteNaturalMeans[i]);
        double cavityMean = (postPrecision * postMean - siteNaturalMean) / cavityPrecision;

        // Compute tilted distribution moments via numerical integration
        double yi = _numOps.ToDouble(y[i]);
        var (tiltedMean, tiltedVar) = ComputeTiltedMoments(
            yi, cavityMean, 1.0 / cavityPrecision);

        // Compute new site parameters
        double tiltedPrecision = 1.0 / tiltedVar;
        double newSitePrecision = tiltedPrecision - cavityPrecision;
        double newSiteNaturalMean = tiltedPrecision * tiltedMean - cavityPrecision * cavityMean;

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

        // A small damping factor makes the update small even far from a
        // fixed point. Return the undamped moment match for engine reduction.
        return (newSitePrecision, newSiteNaturalMean);
    }

    private Tensor<double> NormalizedDifference(Tensor<double> previous, Tensor<double> current)
        => Engine.TensorDivide(Engine.TensorAbs(Engine.TensorSubtract(current, previous)),
            Engine.TensorAddScalar(Engine.TensorAbs(current), 1.0));

    private Vector<T> Blend(Vector<T> previous, Vector<T> proposed, double fraction)
    {
        var old = Tensor<T>.FromVector(previous).Cast<double>();
        var next = Tensor<T>.FromVector(proposed).Cast<double>();
        return Engine.TensorAdd(old, Engine.TensorMultiplyScalar(Engine.TensorSubtract(next, old), fraction))
            .Cast<T>().ToVector();
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
    private (double Mean, double Variance) ComputeTiltedMoments(
        double y, double cavityMean, double cavityVar)
    {
        if (!(cavityVar > 0) || double.IsInfinity(cavityVar))
            throw new InvalidOperationException("Tilted integration requires a finite positive cavity variance.");
        double std = Math.Sqrt(cavityVar);
        double location = (y - cavityMean) / std;
        double width = _scale * Math.Sqrt(_nu) / std;
        // Standardized cavity coordinates, with explicit splits around the
        // likelihood peak. A narrow peak must not fall between fixed nodes.
        var knots = new SortedSet<double>();
        for (int k = -24; k <= 24; k++) knots.Add(k * 0.5);
        void AddLikelihoodKnots()
        {
            knots.Add(location);
            for (double radius = Math.Max(width, 1e-12); radius < 24; radius *= 2)
            {
                knots.Add(location - radius);
                knots.Add(location + radius);
            }
        }
        double LogWeight(double z)
        {
            double residual = Math.Abs(y - (cavityMean + std * z));
            double logRatio = residual == 0 ? double.NegativeInfinity :
                Math.Log(residual) - Math.Log(_scale) - 0.5 * Math.Log(_nu);
            double logKernel = logRatio > 0
                ? 2 * logRatio + LogOnePlus(Math.Exp(-2 * logRatio))
                : LogOnePlus(Math.Exp(2 * logRatio));
            return -0.5 * z * z - 0.5 * (_nu + 1) * logKernel;
        }
        if (Math.Abs(location) <= 12) AddLikelihoodKnots();
        double logReference = double.NegativeInfinity;
        foreach (double z in knots) logReference = Math.Max(logReference, LogWeight(z));
        // For a very small scale an observation beyond the usual Gaussian
        // window can still carry posterior mass. Include that mode as well.
        if (!double.IsInfinity(location) && Math.Abs(location) > 12 &&
            LogWeight(location) > logReference - 40)
        {
            AddLikelihoodKnots();
            foreach (double z in knots) logReference = Math.Max(logReference, LogWeight(z));
        }
        double Integrand(double z, int moment)
        {
            double density = Math.Exp(LogWeight(z) - logReference);
            return moment == 0 ? density : moment == 1 ? density * z : density * z * z;
        }
        var points = knots.ToArray();
        double estimatedMass = 0;
        for (int k = 1; k < points.Length; k++)
        {
            double left = points[k - 1], right = points[k], middle = left + (right - left) / 2;
            estimatedMass += (right - left) *
                (Integrand(left, 0) + 4 * Integrand(middle, 0) + Integrand(right, 0)) / 6;
        }
        if (!(estimatedMass > 0) || double.IsInfinity(estimatedMass))
            throw new InvalidOperationException("Student-t tilted density could not be normalized.");
        double Integrate(int moment)
        {
            double total = 0;
            for (int k = 1; k < points.Length; k++)
            {
                double left = points[k - 1], right = points[k], middle = left + (right - left) / 2;
                double fl = Integrand(left, moment), fm = Integrand(middle, moment), fr = Integrand(right, moment);
                double estimate = (right - left) * (fl + 4 * fm + fr) / 6;
                total += Refine(left, right, fl, fm, fr, estimate, 1e-11 * estimatedMass / points.Length, 24);
            }
            return total;
            double Refine(double left, double right, double fl, double fm, double fr,
                double estimate, double tolerance, int depth)
            {
                double middle = left + (right - left) / 2;
                double lm = left + (middle - left) / 2, rm = middle + (right - middle) / 2;
                double f1 = Integrand(lm, moment), f2 = Integrand(rm, moment);
                double first = (middle - left) * (fl + 4 * f1 + fm) / 6;
                double second = (right - middle) * (fm + 4 * f2 + fr) / 6;
                double error = first + second - estimate;
                if (Math.Abs(error) <= 15 * tolerance || lm == left || rm == right)
                    return first + second + error / 15;
                if (depth == 0)
                    throw new InvalidOperationException("Student-t tilted moment quadrature did not converge.");
                return Refine(left, middle, fl, f1, fm, first, tolerance / 2, depth - 1) +
                    Refine(middle, right, fm, f2, fr, second, tolerance / 2, depth - 1);
            }
        }
        double mass = Integrate(0);
        double standardizedMean = Integrate(1) / mass;
        double standardizedVariance = Integrate(2) / mass - standardizedMean * standardizedMean;
        if (!(mass > 0) || !(standardizedVariance > 0) || double.IsInfinity(standardizedVariance))
            throw new InvalidOperationException("Student-t tilted integration produced invalid moments.");
        return (cavityMean + std * standardizedMean, cavityVar * standardizedVariance);
    }

    // Compensate for the rounding in 1 + x. Here x is in [0, 1], so this
    // provides log1p accuracy without relying on a newer framework Math API.
    private static double LogOnePlus(double x)
    {
        double sum = 1 + x;
        return sum == 1 ? x : Math.Log(sum) * (x / (sum - 1));
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
        if (_K is null || _sitePrecisions is null || _siteNaturalMeans is null) return;
        var engine = Engine;
        int n = _K.Rows;
        var prior = Tensor<T>.FromMatrix(GaussianProcessEngineOps.AddToDiagonal(_K, _numOps.FromDouble(_jitter)));
        var priorLower = CholeskyFactor(prior);
        var tau = Tensor<T>.FromVector(_sitePrecisions).Reshape(new[] { n, 1 });
        // Whitened signed-site precision: B = I + L^T diag(tau) L.
        // Unlike sqrt(tau), this represents legitimate negative EP sites.
        var B = engine.TensorAdd(engine.TensorEye<T>(n),
            engine.TensorMatMul(engine.TensorTranspose(priorLower),
                engine.TensorMultiply(engine.TensorBroadcastTo(tau, new[] { n, n }), priorLower)));
        var lower = CholeskyFactor(B);
        var covariance = engine.TensorMatMul(priorLower,
            LinalgDispatch<T>.CholeskySolve(engine.TensorTranspose(priorLower), lower, upper: false));
        _posteriorCov = covariance.ToMatrix();
        _posteriorMean = engine.TensorMatMul(covariance,
            Tensor<T>.FromVector(_siteNaturalMeans).Reshape(new[] { n, 1 })).Reshape(new[] { n }).ToVector();
    }

    /// <summary>
    /// Predicts the mean and variance for a single input point.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>Latent posterior mean (the predictive location) and observation variance. The variance is positive infinity when nu is at most two.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This predicts for a single point with uncertainty.
    /// The prediction is robust to outliers in the training data.
    /// </para>
    /// </remarks>
    public override (T mean, T variance) Predict(Vector<T> x)
    {
        // Convert single point to matrix
        var XNew = Tensor<T>.FromVector(x).Reshape(new[] { 1, x.Length }).ToMatrix();

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
    public override void UpdateKernel(IKernelFunction<T> kernel)
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
    /// <returns>Latent posterior means and observation variances, including Student-t noise variance nu * scale² / (nu - 2). Variances are positive infinity for nu at most two.</returns>
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
        if (_X is null || _K is null || _posteriorMean is null || _posteriorCov is null || _siteNaturalMeans is null)
            throw new InvalidOperationException("Model not properly initialized.");

        int n = _X.Rows;
        int nNew = XNew.Rows;
        var engine = Engine;
        var prior = Tensor<T>.FromMatrix(GaussianProcessEngineOps.AddToDiagonal(_K, _numOps.FromDouble(_jitter)));
        var priorLower = CholeskyFactor(prior);
        var kStar = Tensor<T>.FromMatrix(ComputeKernelMatrix(XNew, _X));
        var projection = LinalgDispatch<T>.CholeskySolve(engine.TensorTranspose(kStar), priorLower, upper: false);
        var mean = engine.TensorMatMul(engine.TensorTranspose(projection),
            Tensor<T>.FromVector(_posteriorMean).Reshape(new[] { n, 1 })).Reshape(new[] { nNew }).ToVector();
        // Conditional-prior variance plus projected signed-site posterior.
        var reduction = engine.ReduceSum(engine.TensorMultiply(engine.TensorTranspose(kStar), projection),
            new[] { 0 }, keepDims: false);
        var posteriorContribution = engine.ReduceSum(engine.TensorMultiply(projection,
            engine.TensorMatMul(Tensor<T>.FromMatrix(_posteriorCov), projection)),
            new[] { 0 }, keepDims: false);

        var selfKernel = new Vector<T>(XNew.Rows);
        for (int ri = 0; ri < XNew.Rows; ri++)
        {
            var rowA = XNew.GetRow(ri);
            selfKernel[ri] = _kernel.Calculate(rowA, rowA);
        }

        var kStarStar = Tensor<T>.FromVector(selfKernel);
        // Student-t scale is not its standard deviation. Observation noise
        // has no finite variance when nu <= 2, even if latent moments exist.
        double noiseVariance = _nu > 2 ? _scale * _scale * (_nu / (_nu - 2)) : double.PositiveInfinity;
        var latentVariance = engine.TensorAdd(engine.TensorSubtract(kStarStar, reduction), posteriorContribution);
        if (!GaussianProcessEngineOps.AllFinite(latentVariance) ||
            _numOps.LessThan(engine.TensorMinValue(latentVariance), _numOps.Zero))
            throw new InvalidOperationException("Student-t EP produced an invalid predictive latent variance.");
        var variance = engine.TensorAddScalar(latentVariance, _numOps.FromDouble(noiseVariance)).ToVector();
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
    private static Vector<T> GetRow(Matrix<T> matrix, int row) => matrix.GetRow(row);

    /// <summary>
    /// Invalid EP candidates must be damped, rather than hidden by flooring
    /// their Cholesky pivots.
    /// </summary>
    private static Tensor<T> CholeskyFactor(Tensor<T> matrix) =>
        GaussianProcessEngineOps.UnclampedCholesky(matrix.ToMatrix(), MathHelper.GetNumericOperations<T>().Zero)
        ?? throw new InvalidOperationException("Student-t EP precision is not positive definite.");

    #endregion
}
