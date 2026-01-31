using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Interquartile Range (IQR) method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The IQR method is based on quartiles, which divide your data into four parts:
/// - Q1 (25th percentile): 25% of data is below this value
/// - Q3 (75th percentile): 75% of data is below this value
/// - IQR = Q3 - Q1: The range containing the middle 50% of data
///
/// Outliers are points below Q1 - k*IQR or above Q3 + k*IQR, where k is the multiplier (typically 1.5).
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Your data may not be normally distributed
/// - You want a robust method that isn't affected by extreme outliers
/// - This is the same method used in box plots
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Multiplier: 1.5 (identifies "mild" outliers)
/// - Use 3.0 for "extreme" outliers only
/// </para>
/// </remarks>
public class IQRDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _multiplier;
    private Vector<T>? _q1;
    private Vector<T>? _q3;
    private Vector<T>? _iqr;
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;

    /// <summary>
    /// Gets the IQR multiplier for determining outlier boundaries.
    /// </summary>
    public double Multiplier => _multiplier;

    /// <summary>
    /// Gets the Q1 (25th percentile) values for each feature.
    /// </summary>
    public Vector<T>? Q1 => _q1;

    /// <summary>
    /// Gets the Q3 (75th percentile) values for each feature.
    /// </summary>
    public Vector<T>? Q3 => _q3;

    /// <summary>
    /// Gets the IQR values for each feature.
    /// </summary>
    public Vector<T>? IQR => _iqr;

    /// <summary>
    /// Gets the lower bounds (Q1 - multiplier * IQR) for each feature.
    /// </summary>
    public Vector<T>? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the upper bounds (Q3 + multiplier * IQR) for each feature.
    /// </summary>
    public Vector<T>? UpperBounds => _upperBounds;

    /// <summary>
    /// Creates a new IQR-based anomaly detector.
    /// </summary>
    /// <param name="multiplier">
    /// The multiplier applied to the IQR to determine outlier boundaries. Default is 1.5 (industry standard).
    /// Common values:
    /// - 1.5: "Mild" outliers (standard for box plots)
    /// - 3.0: "Extreme" outliers only
    /// </param>
    /// <param name="contamination">
    /// Expected proportion of anomalies. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public IQRDetector(double multiplier = 1.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (multiplier <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(multiplier),
                "Multiplier must be positive. Common values are 1.5 (mild outliers) or 3.0 (extreme outliers).");
        }

        _multiplier = multiplier;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int nFeatures = X.Columns;
        _q1 = new Vector<T>(nFeatures);
        _q3 = new Vector<T>(nFeatures);
        _iqr = new Vector<T>(nFeatures);
        _lowerBounds = new Vector<T>(nFeatures);
        _upperBounds = new Vector<T>(nFeatures);

        T multiplierT = NumOps.FromDouble(_multiplier);

        // Calculate quartiles for each feature
        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var quartile = new Quartile<T>(column);

            _q1[j] = quartile.Q1;
            _q3[j] = quartile.Q3;
            _iqr[j] = NumOps.Subtract(_q3[j], _q1[j]);

            // Lower bound = Q1 - k * IQR
            _lowerBounds[j] = NumOps.Subtract(_q1[j], NumOps.Multiply(multiplierT, _iqr[j]));

            // Upper bound = Q3 + k * IQR
            _upperBounds[j] = NumOps.Add(_q3[j], NumOps.Multiply(multiplierT, _iqr[j]));
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
            // Calculate how far outside the IQR bounds each point is
            // Score = max deviation from bounds normalized by IQR
            T maxScore = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                T value = X[i, j];
                T score = NumOps.Zero;

                // Skip features with zero IQR (constant features)
                if (NumOps.Equals(_iqr![j], NumOps.Zero))
                {
                    continue;
                }

                if (NumOps.LessThan(value, _lowerBounds![j]))
                {
                    // How far below lower bound, normalized by IQR
                    score = NumOps.Divide(
                        NumOps.Subtract(_lowerBounds[j], value),
                        _iqr[j]);
                }
                else if (NumOps.GreaterThan(value, _upperBounds![j]))
                {
                    // How far above upper bound, normalized by IQR
                    score = NumOps.Divide(
                        NumOps.Subtract(value, _upperBounds[j]),
                        _iqr[j]);
                }

                if (NumOps.GreaterThan(score, maxScore))
                {
                    maxScore = score;
                }
            }

            scores[i] = maxScore;
        }

        return scores;
    }
}
