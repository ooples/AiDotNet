using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Z-Score method (standard score).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Z-Score measures how many standard deviations a value is from the mean.
/// A Z-Score of 0 means the value equals the mean. A Z-Score of 2 means the value is 2 standard
/// deviations above the mean. Points with extreme Z-Scores (typically |Z| > 3) are anomalies.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Your data is approximately normally distributed
/// - You want a simple, interpretable method
/// - You have single-variable or multi-variable data where anomalies are extreme in at least one dimension
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Threshold: 3.0 (flags ~0.3% of normally distributed data)
/// - Contamination: 0.1 (10%) - used for automatic threshold tuning
/// </para>
/// </remarks>
public class ZScoreDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _zThreshold;
    private Vector<T>? _means;
    private Vector<T>? _stds;

    /// <summary>
    /// Gets the Z-Score threshold. Points with |Z| > threshold are anomalies.
    /// </summary>
    public double ZThreshold => _zThreshold;

    /// <summary>
    /// Gets the fitted mean values for each feature.
    /// </summary>
    public Vector<T>? Means => _means;

    /// <summary>
    /// Gets the fitted standard deviation values for each feature.
    /// </summary>
    public Vector<T>? StandardDeviations => _stds;

    /// <summary>
    /// Creates a new Z-Score anomaly detector.
    /// </summary>
    /// <param name="zThreshold">
    /// The Z-Score threshold for anomaly detection. Points with |Z| > threshold in any feature
    /// are classified as anomalies. Default is 3.0 (industry standard).
    /// Common values:
    /// - 2.0: More sensitive, flags ~5% of normal data
    /// - 3.0: Standard, flags ~0.3% of normal data
    /// - 4.0: Less sensitive, flags ~0.006% of normal data
    /// </param>
    /// <param name="contamination">
    /// Expected proportion of anomalies. Used to auto-tune threshold if zThreshold is not provided.
    /// Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public ZScoreDetector(double zThreshold = 3.0, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (zThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(zThreshold),
                "Z-Score threshold must be positive. Common values are 2.0, 3.0, or 4.0.");
        }

        _zThreshold = zThreshold;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int nFeatures = X.Columns;
        _means = new Vector<T>(nFeatures);
        _stds = new Vector<T>(nFeatures);

        // Calculate mean and standard deviation for each feature
        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (mean, std) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
            _means[j] = mean;
            _stds[j] = std;
        }

        // Calculate anomaly scores for training data and set threshold
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
            // Maximum absolute Z-score across all features
            T maxAbsZScore = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero standard deviation (constant features)
                if (NumOps.Equals(_stds![j], NumOps.Zero))
                {
                    continue;
                }

                // Z = (x - mean) / std
                T zScore = NumOps.Divide(
                    NumOps.Subtract(X[i, j], _means![j]),
                    _stds[j]);

                T absZScore = NumOps.Abs(zScore);

                if (NumOps.GreaterThan(absZScore, maxAbsZScore))
                {
                    maxAbsZScore = absZScore;
                }
            }

            scores[i] = maxAbsZScore;
        }

        return scores;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = ScoreAnomalies(X);
        var predictions = new Vector<T>(scores.Length);
        T zThresholdT = NumOps.FromDouble(_zThreshold);

        for (int i = 0; i < scores.Length; i++)
        {
            // Points with max |Z| > threshold are anomalies (-1), otherwise inliers (1)
            predictions[i] = NumOps.GreaterThan(scores[i], zThresholdT)
                ? NumOps.FromDouble(-1)
                : NumOps.FromDouble(1);
        }

        return predictions;
    }
}
