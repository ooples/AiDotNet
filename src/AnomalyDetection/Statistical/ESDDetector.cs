using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using ESD-based (Extreme Studentized Deviate) scoring.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This detector uses standardized residuals (the core statistic from the ESD test)
/// to identify outliers. Points that deviate significantly from the mean (measured in standard deviations)
/// are flagged as anomalies.
/// </para>
/// <para>
/// <b>Implementation:</b> This is a scoring-based approach following industry standards:
/// 1. Compute the ESD statistic for each point: max|x - mean| / std across features
/// 2. Use contamination-based threshold (top X% of scores are anomalies)
/// 3. CriticalValue property available for users who need statistical significance testing
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you expect multiple outliers
/// - Data is approximately normally distributed
/// - You want Z-score style detection with contamination-based thresholding
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%) - used for critical value computation
/// - Contamination: 10% of data flagged as outliers
/// </para>
/// <para>
/// Reference: Rosner, B. (1983). "Percentage Points for a Generalized ESD Many-Outlier Procedure."
/// Technometrics.
/// </para>
/// </remarks>
public class ESDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private readonly int? _maxOutliers;
    private Vector<T>? _means;
    private Vector<T>? _stds;
    private int _nFeatures;
    private int _nSamples;
    private double _criticalValue;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the maximum number of outliers to detect.
    /// </summary>
    public int? MaxOutliers => _maxOutliers;

    /// <summary>
    /// Gets the ESD critical value computed during fitting.
    /// </summary>
    /// <remarks>
    /// This value is provided for users who need statistical significance testing.
    /// Compare ScoreAnomalies results against this value if you want hypothesis-test based detection
    /// instead of contamination-based thresholding.
    /// </remarks>
    public double CriticalValue => _criticalValue;

    /// <summary>
    /// Creates a new ESD anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// The significance level for the test. Default is 0.05 (5%).
    /// </param>
    /// <param name="maxOutliers">
    /// Maximum number of outliers to detect. Default is null (auto-detect based on contamination).
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    public ESDDetector(double alpha = 0.05, int? maxOutliers = null, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 and 1. Recommended value is 0.05.");
        }

        if (maxOutliers.HasValue && maxOutliers.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxOutliers),
                "MaxOutliers must be positive when specified.");
        }

        _alpha = alpha;
        _maxOutliers = maxOutliers;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        _nSamples = X.Rows;
        _means = new Vector<T>(_nFeatures);
        _stds = new Vector<T>(_nFeatures);

        for (int j = 0; j < _nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (mean, std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
            _means[j] = mean;
            _stds[j] = std;
        }

        // Compute ESD critical value for the first iteration
        // lambda_1 = (n-1) * t_(alpha/(2n), n-2) / sqrt((n-2+t^2)*(n))
        int effectiveMaxOutliers = _maxOutliers ?? Math.Max(1, (int)(_nSamples * _contamination));
        _criticalValue = ComputeESDCriticalValue(_nSamples, 1, effectiveMaxOutliers);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double ComputeESDCriticalValue(int n, int i, int r)
    {
        // ESD critical value for iteration i when testing for up to r outliers
        // lambda_i = (n-i) * t_(p, n-i-1) / sqrt((n-i-1+t^2)*(n-i+1))
        // where p = alpha / (2*(n-i+1))
        int ni = n - i;
        if (ni < 3) return double.MaxValue; // Cannot compute for very small samples

        double p = _alpha / (2.0 * (ni + 1));
        double t = GetTCritical(p, ni - 1);

        double numerator = ni * t;
        double denominator = Math.Sqrt((ni - 1 + t * t) * (ni + 1));

        return numerator / denominator;
    }

    private double GetTCritical(double p, int df)
    {
        if (df < 1) return double.MaxValue;

        // Approximation for t-distribution critical value
        double z = GetZCritical(p);

        if (df >= 30)
        {
            return z;
        }

        // Use Wilson-Hilferty transformation for small df
        double g1 = 1.0 / df;
        return z + g1 * (z * z * z + z) / 4.0;
    }

    private double GetZCritical(double p)
    {
        // Approximation of inverse standard normal CDF
        if (p <= 0) return double.MaxValue;
        if (p >= 1) return double.MinValue;
        if (p > 0.5) return -GetZCritical(1 - p);

        double t = Math.Sqrt(-2 * Math.Log(p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Compute ESD statistic: max standardized residual across features
            T maxESD = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero std
                if (NumOps.Equals(stds[j], NumOps.Zero))
                {
                    continue;
                }

                // ESD = |x - mean| / std
                T deviation = NumOps.Abs(NumOps.Subtract(X[i, j], means[j]));
                T esdStatistic = NumOps.Divide(deviation, stds[j]);

                if (NumOps.GreaterThan(esdStatistic, maxESD))
                {
                    maxESD = esdStatistic;
                }
            }

            scores[i] = maxESD;
        }

        return scores;
    }
}
