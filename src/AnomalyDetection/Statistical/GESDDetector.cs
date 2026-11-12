using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using GESD-based (Generalized Extreme Studentized Deviate) scoring.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This detector uses standardized residuals (the core statistic from the GESD test)
/// to identify outliers. Points that deviate significantly from the mean are flagged as anomalies.
/// </para>
/// <para>
/// <b>Implementation:</b> This is a scoring-based approach:
/// 1. Compute the GESD statistic for each point: max|x - mean| / std across features
/// 2. Use contamination-based threshold to classify anomalies (top X% of scores)
/// Critical values are computed from alpha for reference but thresholding uses contamination.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data is approximately normally distributed
/// - You want Z-score style detection with contamination-based thresholding
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// - Contamination: 10% of data flagged as outliers
/// </para>
/// <para>
/// Reference: Rosner, B. (1983). "Percentage Points for a Generalized ESD Many-Outlier Procedure."
/// Technometrics.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Percentage Points for a Generalized ESD Many-Outlier Procedure", "https://doi.org/10.2307/1268549", Year = 1983, Authors = "Bernard Rosner")]
public partial class GESDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private readonly int _maxOutliers;
    [Buffer]
    private Vector<T>? _means;
    [Buffer]
    private Vector<T>? _stds;
    [Buffer]
    private Vector<T>? _criticalValues;
    private int _fittedN;
    private int _nFeatures;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the maximum number of outliers to detect.
    /// </summary>
    public int MaxOutliersCount => _maxOutliers;

    /// <summary>The Rosner critical values lambda_1..lambda_r (set by Fit).</summary>
    internal Vector<T>? CriticalValues => _criticalValues;

    /// <summary>
    /// Creates a new GESD anomaly detector.
    /// </summary>
    /// <param name="maxOutliers">
    /// Maximum number of outliers to detect. Default is 10.
    /// Should be set based on your expected contamination rate.
    /// </param>
    /// <param name="alpha">
    /// The significance level for the test. Default is 0.05 (5%).
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    public GESDDetector(int maxOutliers = 10, double alpha = 0.05, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (maxOutliers < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxOutliers),
                "Maximum outliers must be at least 1.");
        }

        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 and 1. Recommended value is 0.05.");
        }

        _maxOutliers = maxOutliers;
        _alpha = alpha;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _fittedN = n;

        if (n <= _maxOutliers)
        {
            throw new ArgumentException(
                $"Number of samples ({n}) must be greater than maximum outliers ({_maxOutliers}).",
                nameof(X));
        }

        _nFeatures = X.Columns;
        // R = max |x - mean| / s with s the sample standard deviation (n - 1 divisor), Rosner (1983)
        (_means, _stds) = AnomalyTensorOps.ColumnMeanStd(X, 1);

        // Precompute critical values for each iteration
        var lambdas = new double[_maxOutliers];
        for (int i = 0; i < _maxOutliers; i++)
        {
            lambdas[i] = ComputeLambda(n, i + 1, _alpha);
        }

        _criticalValues = Tensor<double>.FromVector(Vector<double>.Wrap(lambdas), new[] { _maxOutliers }).Cast<T>().ToVector();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var means = _means;
        var stds = _stds;
        if (means == null || stds == null)
        {
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
        }

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        // GESD statistic: max standardized residual R = |x - mean| / std across features (zero-std features skipped)
        return AnomalyTensorOps.MaxStandardizedDeviation(X, means, stds);
    }

    private double ComputeLambda(int n, int i, double alpha)
    {
        // Rosner (1983) GESD critical value for the i-th largest statistic (i = 1..r) of n observations:
        // lambda_i = (n - i) t_{p, n-i-1} / sqrt((n - i - 1 + t^2)(n - i + 1)), p = 1 - alpha / (2(n - i + 1))
        int df = n - i - 1;
        if (df < 1) return double.MaxValue; // Cannot compute for very small samples

        double p = 1 - alpha / (2.0 * (n - i + 1));
        double t = AnomalyTensorOps.StudentTQuantile(p, df);
        return (n - i) * t / Math.Sqrt((n - i - 1 + t * t) * (n - i + 1));
    }
}
