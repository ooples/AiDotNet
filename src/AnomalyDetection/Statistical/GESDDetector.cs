using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Generalized Extreme Studentized Deviate (GESD) test.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The GESD test is an iterative procedure for detecting up to k outliers
/// in a univariate dataset. It extends the ESD test by explicitly specifying the maximum number
/// of outliers to look for.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Set an upper bound k on the number of outliers
/// 2. Compute k test statistics R_1, R_2, ..., R_k
/// 3. Compute k critical values lambda_1, lambda_2, ..., lambda_k
/// 4. Find the largest i where R_i > lambda_i; there are i outliers
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have an upper bound on the number of outliers
/// - Data is approximately normally distributed
/// - You want to detect multiple outliers
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// - Max outliers k: Often set to 10% of sample size
/// </para>
/// <para>
/// Reference: Rosner, B. (1983). "Percentage Points for a Generalized ESD Many-Outlier Procedure."
/// Technometrics.
/// </para>
/// </remarks>
public class GESDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private readonly int _maxOutliers;
    private Vector<T>? _means;
    private Vector<T>? _stds;
    private double[]? _criticalValues;
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
        _means = new Vector<T>(_nFeatures);
        _stds = new Vector<T>(_nFeatures);

        for (int j = 0; j < _nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (mean, std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
            _means[j] = mean;
            _stds[j] = std;
        }

        // Precompute critical values for each iteration
        _criticalValues = new double[_maxOutliers];
        for (int i = 0; i < _maxOutliers; i++)
        {
            int currentN = n - i;
            _criticalValues[i] = ComputeLambda(currentN, _alpha);
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Compute GESD statistic: max standardized residual across features
            T maxR = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero std
                if (NumOps.Equals(_stds![j], NumOps.Zero))
                {
                    continue;
                }

                // R = |x - mean| / std
                T deviation = NumOps.Abs(NumOps.Subtract(X[i, j], _means![j]));
                T r = NumOps.Divide(deviation, _stds[j]);

                if (NumOps.GreaterThan(r, maxR))
                {
                    maxR = r;
                }
            }

            scores[i] = maxR;
        }

        return scores;
    }

    /// <summary>
    /// Computes the critical value lambda for the GESD test.
    /// </summary>
    /// <param name="n">Current sample size.</param>
    /// <param name="alpha">Significance level.</param>
    /// <returns>Critical value.</returns>
    private double ComputeLambda(int n, double alpha)
    {
        // Rosner 1983 GESD critical value formula:
        // lambda = n * t_{p, n-2} / sqrt((n - 1 + t^2) * (n + 1))
        // where p = 1 - alpha / (2(n + 1))
        // Note: n here is the current sample size (original n minus outliers removed so far)

        if (n < 3) return double.MaxValue; // Cannot compute for very small samples

        double p = 1 - alpha / (2.0 * (n + 1));
        int df = n - 2;

        // Get t-critical value
        double t = GetTCritical(p, Math.Max(1, df));

        // Compute lambda using correct Rosner formula
        double lambda = (n * t) / Math.Sqrt((n - 1 + t * t) * (n + 1));

        return lambda;
    }

    private double GetTCritical(double p, int df)
    {
        // Approximation of t-distribution quantile
        double z = GetZCritical(1 - p);

        if (df >= 30)
        {
            return z;
        }

        // Cornish-Fisher expansion for small df
        double g1 = 1.0 / df;
        double g2 = 1.0 / (df * df);

        return z + (z * z * z - 3 * z) * g1 / 4 +
               (z * z * z * z * z - 10 * z * z * z + 15 * z) * g2 / 96;
    }

    private double GetZCritical(double alpha)
    {
        // Approximation of inverse standard normal CDF
        double p = alpha;
        if (p > 0.5) p = 1 - p;
        if (p <= 0) p = 1e-10;

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
