using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using Bayesian probability estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This detector uses Bayesian probability to model normal data distribution.
/// It assumes data follows a multivariate Gaussian distribution with prior beliefs about
/// the parameters. Points with low likelihood under this model are considered anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Estimate mean and covariance with Bayesian priors
/// 2. Compute posterior predictive probability for each point
/// 3. Low probability points are flagged as anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have prior knowledge about the data distribution
/// - For probabilistic anomaly scoring
/// - When you want uncertainty estimates
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Prior strength (kappa0): 0.01 (weak prior)
/// - Prior degrees of freedom: n_features + 2
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Murphy, K.P. (2012). "Machine Learning: A Probabilistic Perspective."
/// Chapter on Bayesian inference for MVN.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Bayesian Anomaly Detection Methods for Social Networks", "https://doi.org/10.1214/10-AOAS370", Year = 2010, Authors = "Savage, Coull, Heard")]
public partial class BayesianDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _priorStrength;
    [Buffer]
    private Vector<T>? _posteriorMean;
    [Buffer]
    private Matrix<T>? _posteriorCovariance;
    [Buffer]
    private Matrix<T>? _posteriorPrecision;
    private int _nFeatures;

    private T _logNormalization;


    /// <summary>
    /// Gets the prior strength parameter.
    /// </summary>
    public double PriorStrength => _priorStrength;

    /// <summary>
    /// Creates a new Bayesian anomaly detector.
    /// </summary>
    /// <param name="priorStrength">
    /// Strength of the prior (kappa_0). Smaller values = weaker prior. Default is 0.01.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public BayesianDetector(double priorStrength = 0.01, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _logNormalization = NumOps.Zero;
        if (priorStrength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(priorStrength),
                "PriorStrength must be positive. Recommended is 0.01.");
        }

        _priorStrength = priorStrength;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;
        T nT = NumOps.FromDouble(n);

        // Compute sample mean and (population) sample covariance
        var engine = Engine;
        var data = AnomalyTensorOps.ToTensor(X);
        var sampleMean = engine.ReduceMean(data, new[] { 0 }, false);
        var sampleCov = AnomalyTensorOps.Covariance(AnomalyTensorOps.SubtractRow(data, sampleMean), nT);

        // Prior parameters (vague/weakly informative)
        T kappa0 = NumOps.FromDouble(_priorStrength);
        T nu0 = NumOps.FromDouble(_nFeatures + 2); // Prior degrees of freedom

        // Posterior parameters (Normal-Inverse-Wishart)
        T kappaN = NumOps.Add(kappa0, nT);
        T nuN = NumOps.Add(nu0, nT);

        // Posterior mean (kappa0 * mean + n * mean) / kappaN
        _posteriorMean = engine.TensorDivideScalar(engine.TensorAdd(
            engine.TensorMultiplyScalar(sampleMean, kappa0), engine.TensorMultiplyScalar(sampleMean, nT)), kappaN).ToVector();

        // Posterior covariance (for predictive distribution), regularized on the diagonal
        T scale = NumOps.Divide(
            NumOps.Add(kappaN, NumOps.One),
            NumOps.Multiply(kappaN, NumOps.Subtract(nuN, NumOps.FromDouble(_nFeatures - 1))));
        var posteriorCovariance = engine.TensorAdd(engine.TensorMultiplyScalar(sampleCov, scale),
            AnomalyTensorOps.ScaledIdentity(_nFeatures, NumOps.FromDouble(1e-6)));
        _posteriorCovariance = posteriorCovariance.ToMatrix();

        // Compute precision (inverse covariance)
        _posteriorPrecision = LinalgDispatch<T>.Inv(posteriorCovariance).ToMatrix();

        // Compute log normalization for MVN
        T logDet = LogDeterminant(posteriorCovariance);
        T half = NumOps.FromDouble(0.5);
        _logNormalization = NumOps.Negate(NumOps.Multiply(half,
            NumOps.Add(NumOps.FromDouble(_nFeatures * Math.Log(2 * Math.PI)), logDet)));

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    // log|det| from the LU pivots (partial pivoting); -1e30 (approximately negative infinity) when a pivot
    // is below 1e-10 in magnitude.
    private T LogDeterminant(Tensor<T> matrix)
    {
        var engine = Engine;
        var (_, _, U) = LinalgDispatch<T>.LU(matrix);
        var pivots = engine.TensorAbs(LinalgDispatch<T>.Diagonal(U));
        if (NumOps.LessThan(engine.TensorMinValue(pivots), NumOps.FromDouble(1e-10)))
        {
            return NumOps.FromDouble(-1e30);
        }

        return engine.TensorSum(engine.TensorLog(pivots));
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var posteriorMean = _posteriorMean ?? throw new InvalidOperationException("_posteriorMean has not been initialized.");
        var posteriorPrecision = _posteriorPrecision ?? throw new InvalidOperationException("_posteriorPrecision has not been initialized.");

        // Negative log likelihood (higher = more anomalous): -(logNormalization - mahalanobis^2 / 2)
        var engine = Engine;
        var centered = AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), posteriorMean.AsTensor());
        var mahalSq = AnomalyTensorOps.MahalanobisSquared(centered, AnomalyTensorOps.ToTensor(posteriorPrecision));
        return engine.TensorNegate(engine.ScalarMinusTensor(_logNormalization,
            engine.TensorMultiplyScalar(mahalSq, NumOps.FromDouble(0.5)))).ToVector();
    }
}
