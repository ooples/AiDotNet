using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using Grubbs' Test for a single outlier.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Grubbs' Test (also called the maximum normed residual test) identifies
/// a single outlier in a univariate dataset. It tests whether the most extreme value is
/// significantly different from the others.
/// </para>
/// <para>
/// The test statistic is: G = max(|x_i - mean|) / std
/// where the outlier is the point furthest from the mean relative to standard deviation.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you suspect exactly one outlier in your data
/// - Data is approximately normally distributed
/// - Dataset is relatively small (n &lt; 100)
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// - For multivariate data, applies test to each feature
/// </para>
/// <para>
/// Reference: Grubbs, F. E. (1950). "Sample criteria for testing outlying observations."
/// Annals of Mathematical Statistics.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Sample Criteria for Testing Outlying Observations", "https://doi.org/10.1214/aoms/1177729885", Year = 1950, Authors = "Frank E. Grubbs")]
public partial class GrubbsTestDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    [Buffer]
    private Vector<T>? _means;
    [Buffer]
    private Vector<T>? _stds;
    [Buffer]
    private Vector<T>? _criticalValues;
    private int _nFeatures;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>The two-sided Grubbs critical value G_crit per feature (set by Fit).</summary>
    internal Vector<T>? CriticalValues => _criticalValues;

    /// <summary>
    /// Creates a new Grubbs' Test anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// The significance level for the test. Default is 0.05 (5%). It sets the two-sided critical value
    /// G_crit computed by Fit (a lower alpha gives a larger G_crit, a more conservative test); compare
    /// <c>ScoreAnomalies</c> with G_crit to apply the test itself. <c>Predict</c> labels come from the
    /// <paramref name="contamination"/> threshold and do not depend on alpha.
    /// Common values: 0.01, 0.05, 0.10.
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    public GrubbsTestDetector(double alpha = 0.05, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 and 1. Recommended value is 0.05.");
        }

        _alpha = alpha;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int nFeatures = X.Columns;

        // Grubbs' test requires at least 3 samples (df = n - 2 must be positive)
        if (n < 3)
        {
            throw new ArgumentException(
                $"Grubbs' test requires at least 3 samples, but got {n}.",
                nameof(X));
        }

        _nFeatures = nFeatures;


        // Calculate critical value for Grubbs' test
        // G_critical = ((n-1) / sqrt(n)) * sqrt(t^2 / (n - 2 + t^2))
        // where t = t_(alpha/(2n), n-2) from Student's t-distribution
        double tCritical = GetTCritical(n);
        double gCritical = ((n - 1) / Math.Sqrt(n)) * Math.Sqrt(tCritical * tCritical / (n - 2 + tCritical * tCritical));

        // G = max |x - mean| / s with s the sample standard deviation (n - 1 divisor), as Grubbs (1950)
        // and the NIST/SEMATECH e-Handbook (1.3.5.17.1) define the statistic
        (_means, _stds) = AnomalyTensorOps.ColumnMeanStd(X, 1);
        _criticalValues = Engine.Fill(nFeatures, NumOps.FromDouble(gCritical));

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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var stds = _stds ?? throw new InvalidOperationException("_stds has not been initialized.");
        var means = _means ?? throw new InvalidOperationException("_means has not been initialized.");

        // Score: the maximum Grubbs statistic G = |x - mean| / std across features (zero-std features skipped)
        return AnomalyTensorOps.MaxStandardizedDeviation(X, means, stds);
    }

    // Upper alpha / (2n) quantile of Student's t with n - 2 degrees of freedom (two-sided test).
    private double GetTCritical(int n) => AnomalyTensorOps.StudentTQuantile(1 - _alpha / (2.0 * n), n - 2);
}
