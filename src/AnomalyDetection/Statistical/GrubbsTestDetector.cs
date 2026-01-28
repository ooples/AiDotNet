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
public class GrubbsTestDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private Vector<T>? _means;
    private Vector<T>? _stds;
    private Vector<T>? _criticalValues;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Creates a new Grubbs' Test anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// The significance level for the test. Default is 0.05 (5%).
    /// Lower values make the test more conservative (fewer outliers detected).
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

        _means = new Vector<T>(nFeatures);
        _stds = new Vector<T>(nFeatures);
        _criticalValues = new Vector<T>(nFeatures);

        // Calculate critical value for Grubbs' test
        // G_critical = ((n-1) / sqrt(n)) * sqrt(t^2 / (n - 2 + t^2))
        // where t = t_(alpha/(2n), n-2) from Student's t-distribution
        double tCritical = GetTCritical(n);
        double gCritical = ((n - 1) / Math.Sqrt(n)) * Math.Sqrt(tCritical * tCritical / (n - 2 + tCritical * tCritical));

        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (mean, std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
            _means[j] = mean;
            _stds[j] = std;
            _criticalValues[j] = NumOps.FromDouble(gCritical);
        }

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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            T maxGrubbsStatistic = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero std
                if (NumOps.Equals(_stds![j], NumOps.Zero))
                {
                    continue;
                }

                // G = |x - mean| / std
                T deviation = NumOps.Abs(NumOps.Subtract(X[i, j], _means![j]));
                T grubbsStatistic = NumOps.Divide(deviation, _stds[j]);

                if (NumOps.GreaterThan(grubbsStatistic, maxGrubbsStatistic))
                {
                    maxGrubbsStatistic = grubbsStatistic;
                }
            }

            scores[i] = maxGrubbsStatistic;
        }

        return scores;
    }

    private double GetTCritical(int n)
    {
        // Approximate t-critical value for alpha/(2n) with n-2 degrees of freedom
        // Using approximation since we don't have a full t-distribution implementation
        double adjustedAlpha = _alpha / (2 * n);
        int df = n - 2;

        // Simple approximation for t-distribution critical value
        // For large df, t approaches z (standard normal)
        // For small df, we use a lookup/approximation
        if (df >= 30)
        {
            // Use normal approximation
            return GetZCritical(adjustedAlpha);
        }
        else
        {
            // Approximate t-critical using Wilson-Hilferty transformation
            double z = GetZCritical(adjustedAlpha);
            double g1 = 1.0 / df;
            return z * Math.Sqrt((df - 2.0) / (df - 1.0)) + g1 * (z * z * z - 3 * z) / 4;
        }
    }

    private double GetZCritical(double alpha)
    {
        // Approximation of inverse standard normal CDF (probit function)
        // Using Abramowitz and Stegun approximation
        double p = alpha;
        if (p > 0.5) p = 1 - p;

        double t = Math.Sqrt(-2 * Math.Log(p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return alpha > 0.5 ? -z : z;
    }
}
