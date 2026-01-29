using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using Copula-Based Outlier Detection (COPOD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> COPOD uses copulas (statistical functions that describe dependencies
/// between variables) to model the joint probability of data points. Points with very low
/// joint probability are anomalies - they are statistically unlikely given the data distribution.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Transform each feature to empirical probability using ECDF
/// 2. Model the joint distribution using empirical copula
/// 3. Compute the negative log probability as the anomaly score
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When features have different marginal distributions
/// - When you want a parameter-free method
/// - High-dimensional data with complex dependencies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Contamination: 0.1 (10%)
/// - No hyperparameters to tune (parameter-free method)
/// </para>
/// <para>
/// Reference: Li, Z., et al. (2020). "COPOD: Copula-Based Outlier Detection." ICDM.
/// </para>
/// </remarks>
public class COPODDetector<T> : AnomalyDetectorBase<T>
{
    private double[][]? _sortedFeatureValues;
    private int _nFeatures;
    private int _nSamples;

    /// <summary>
    /// Creates a new COPOD anomaly detector.
    /// </summary>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public COPODDetector(double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        _nSamples = X.Rows;

        // Pre-sort feature values for O(log n) ECDF lookups
        _sortedFeatureValues = new double[_nFeatures][];
        for (int j = 0; j < _nFeatures; j++)
        {
            var values = new double[_nSamples];
            for (int i = 0; i < _nSamples; i++)
            {
                values[i] = NumOps.ToDouble(X[i, j]);
            }
            Array.Sort(values);
            _sortedFeatureValues[j] = values;
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
            double score = 0;

            // For each feature, compute the empirical tail probability
            for (int j = 0; j < _nFeatures; j++)
            {
                double value = NumOps.ToDouble(X[i, j]);

                // Compute empirical CDF (left tail) and survival function (right tail)
                double leftTailProb = ComputeECDF(j, value);
                double rightTailProb = 1 - leftTailProb;

                // Add negative log of minimum tail probability (skewness-corrected)
                // This follows the COPOD paper's approach
                double minTailProb = Math.Min(leftTailProb, rightTailProb);

                // Avoid log(0)
                minTailProb = Math.Max(minTailProb, 1e-10);

                score += -Math.Log(minTailProb);
            }

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private double ComputeECDF(int featureIndex, double value)
    {
        var sortedValues = _sortedFeatureValues;
        if (sortedValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Use binary search for O(log n) lookup
        var featureValues = sortedValues[featureIndex];
        int index = Array.BinarySearch(featureValues, value);

        // BinarySearch returns bitwise complement of insertion point if not found
        int count;
        if (index >= 0)
        {
            // Value found - count all values <= value (handle duplicates)
            count = index + 1;
            while (count < featureValues.Length && featureValues[count] <= value)
            {
                count++;
            }
        }
        else
        {
            // Value not found - insertion point is the count of values < value
            count = ~index;
        }

        // Return proportion, adjusted to avoid 0 and 1
        return (count + 0.5) / (_nSamples + 1);
    }
}
