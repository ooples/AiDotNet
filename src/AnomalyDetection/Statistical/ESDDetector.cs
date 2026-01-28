using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Extreme Studentized Deviate (ESD) test.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The ESD test is an extension of Grubbs' test that can detect multiple outliers.
/// It iteratively identifies potential outliers by finding the point furthest from the mean,
/// checking if it exceeds a critical value, and if so, removing it and repeating.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute the test statistic R_i = max|x_j - mean| / std for remaining data
/// 2. Compare R_i to the critical value lambda_i
/// 3. If R_i > lambda_i, the point is an outlier; remove it and repeat
/// 4. Continue until no more outliers are found or max outliers reached
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you expect multiple outliers
/// - Data is approximately normally distributed
/// - You have an upper bound on the number of outliers
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// - Max outliers: 10% of data or sqrt(n)
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

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the maximum number of outliers to detect.
    /// </summary>
    public int? MaxOutliers => _maxOutliers;

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

        _alpha = alpha;
        _maxOutliers = maxOutliers;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int nFeatures = X.Columns;
        _means = new Vector<T>(nFeatures);
        _stds = new Vector<T>(nFeatures);

        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (mean, std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
            _means[j] = mean;
            _stds[j] = std;
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
            // Compute ESD statistic: max standardized residual across features
            T maxESD = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero std
                if (NumOps.Equals(_stds![j], NumOps.Zero))
                {
                    continue;
                }

                // ESD = |x - mean| / std
                T deviation = NumOps.Abs(NumOps.Subtract(X[i, j], _means![j]));
                T esdStatistic = NumOps.Divide(deviation, _stds[j]);

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
