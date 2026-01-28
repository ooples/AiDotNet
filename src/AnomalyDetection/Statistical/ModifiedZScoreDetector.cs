using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Modified Z-Score method based on Median Absolute Deviation (MAD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Modified Z-Score is a robust alternative to the standard Z-Score.
/// Instead of using mean and standard deviation (which are sensitive to outliers), it uses:
/// - Median: The middle value when data is sorted
/// - MAD: Median Absolute Deviation, a robust measure of spread
///
/// Modified Z-Score = 0.6745 * (x - median) / MAD
/// (0.6745 is a scaling factor to make MAD comparable to standard deviation for normal distributions)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Your data contains extreme outliers that would skew mean/std
/// - Your data is not normally distributed
/// - You want a method that remains reliable even with 50% outliers
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Threshold: 3.5 (Iglewicz and Hoaglin recommendation)
/// - Alternative: 3.0 for more sensitivity
/// </para>
/// </remarks>
public class ModifiedZScoreDetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>
    /// Scaling factor to make MAD comparable to standard deviation for normal distributions.
    /// k = 1 / Phi^(-1)(0.75) ≈ 0.6745, where Phi is the standard normal CDF.
    /// </summary>
    private const double MAD_SCALE_FACTOR = 0.6745;

    private readonly double _modifiedZThreshold;
    private Vector<T>? _medians;
    private Vector<T>? _mads;

    /// <summary>
    /// Gets the Modified Z-Score threshold. Points with |Modified Z| > threshold are anomalies.
    /// </summary>
    public double ModifiedZThreshold => _modifiedZThreshold;

    /// <summary>
    /// Gets the fitted median values for each feature.
    /// </summary>
    public Vector<T>? Medians => _medians;

    /// <summary>
    /// Gets the fitted MAD (Median Absolute Deviation) values for each feature.
    /// </summary>
    public Vector<T>? MADs => _mads;

    /// <summary>
    /// Creates a new Modified Z-Score (MAD-based) anomaly detector.
    /// </summary>
    /// <param name="modifiedZThreshold">
    /// The Modified Z-Score threshold for anomaly detection. Default is 3.5 (Iglewicz and Hoaglin recommendation).
    /// Common values:
    /// - 3.0: More sensitive
    /// - 3.5: Standard (recommended by Iglewicz and Hoaglin)
    /// - 4.0: Less sensitive
    /// </param>
    /// <param name="contamination">
    /// Expected proportion of anomalies. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public ModifiedZScoreDetector(double modifiedZThreshold = 3.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (modifiedZThreshold <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modifiedZThreshold),
                "Modified Z-Score threshold must be positive. Recommended value is 3.5.");
        }

        _modifiedZThreshold = modifiedZThreshold;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int nFeatures = X.Columns;
        _medians = new Vector<T>(nFeatures);
        _mads = new Vector<T>(nFeatures);

        // Calculate median and MAD for each feature
        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);

            // Calculate median
            _medians[j] = CalculateMedian(column);

            // Calculate MAD = median(|x - median|)
            _mads[j] = CalculateMAD(column, _medians[j]);
        }

        // Calculate anomaly scores for training data and set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private T CalculateMedian(Vector<T> values)
    {
        var sorted = values.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

        int n = sorted.Length;
        if (n == 0)
        {
            return NumOps.Zero;
        }

        if (n % 2 == 0)
        {
            // Average of two middle values
            return NumOps.Divide(
                NumOps.Add(sorted[n / 2 - 1], sorted[n / 2]),
                NumOps.FromDouble(2));
        }
        else
        {
            return sorted[n / 2];
        }
    }

    private T CalculateMAD(Vector<T> values, T median)
    {
        // Calculate |x - median| for each value
        var absoluteDeviations = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            absoluteDeviations[i] = NumOps.Abs(NumOps.Subtract(values[i], median));
        }

        // MAD is the median of absolute deviations
        return CalculateMedian(absoluteDeviations);
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
        T scaleFactor = NumOps.FromDouble(MAD_SCALE_FACTOR);

        for (int i = 0; i < X.Rows; i++)
        {
            // Maximum absolute Modified Z-Score across all features
            T maxAbsModifiedZ = NumOps.Zero;

            for (int j = 0; j < X.Columns; j++)
            {
                // Skip features with zero MAD (constant features)
                if (NumOps.Equals(_mads![j], NumOps.Zero))
                {
                    continue;
                }

                // Modified Z = k * (x - median) / MAD
                // where k = 0.6745 (scaling factor)
                T deviation = NumOps.Subtract(X[i, j], _medians![j]);
                T modifiedZ = NumOps.Divide(
                    NumOps.Multiply(scaleFactor, deviation),
                    _mads[j]);

                T absModifiedZ = NumOps.Abs(modifiedZ);

                if (NumOps.GreaterThan(absModifiedZ, maxAbsModifiedZ))
                {
                    maxAbsModifiedZ = absModifiedZ;
                }
            }

            scores[i] = maxAbsModifiedZ;
        }

        return scores;
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = ScoreAnomalies(X);
        var predictions = new Vector<T>(scores.Length);
        T thresholdT = NumOps.FromDouble(_modifiedZThreshold);

        for (int i = 0; i < scores.Length; i++)
        {
            // Points with max |Modified Z| > threshold are anomalies (-1), otherwise inliers (1)
            predictions[i] = NumOps.GreaterThan(scores[i], thresholdT)
                ? NumOps.FromDouble(-1)
                : NumOps.FromDouble(1);
        }

        return predictions;
    }
}
