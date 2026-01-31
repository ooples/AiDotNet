using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using ECOD (Empirical Cumulative Distribution Functions for Outlier Detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ECOD uses the cumulative distribution function (CDF) to identify
/// outliers. Points with extreme values in any dimension have low probability under the
/// empirical distribution and are flagged as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute empirical CDF for each feature
/// 2. Calculate tail probabilities for each point
/// 3. Combine probabilities across features
/// 4. Points with very low probabilities are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Large datasets (linear complexity O(n))
/// - High-dimensional data
/// - When you need a fast, parameter-free method
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - No parameters to tune (parameter-free)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Li, Z., et al. (2022). "ECOD: Unsupervised Outlier Detection Using
/// Empirical Cumulative Distribution Functions." IEEE TKDE.
/// </para>
/// </remarks>
public class ECODDetector<T> : AnomalyDetectorBase<T>
{
    private double[][]? _sortedFeatureValues;
    private int _nFeatures;
    private int _nTrainingSamples;

    /// <summary>
    /// Creates a new ECOD anomaly detector.
    /// </summary>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ECODDetector(double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        _nTrainingSamples = X.Rows;

        // Store sorted values for each feature to compute empirical CDF
        _sortedFeatureValues = new double[_nFeatures][];

        for (int j = 0; j < _nFeatures; j++)
        {
            var featureValues = new double[X.Rows];
            for (int i = 0; i < X.Rows; i++)
            {
                featureValues[i] = NumOps.ToDouble(X[i, j]);
            }
            _sortedFeatureValues[j] = featureValues.OrderBy(v => v).ToArray();
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
            double combinedScore = 0;

            for (int j = 0; j < _nFeatures; j++)
            {
                double value = NumOps.ToDouble(X[i, j]);

                // Compute empirical CDF: F(x) = proportion of values <= x
                double leftCdf = ComputeECDF(value, _sortedFeatureValues![j]);
                double rightCdf = 1.0 - leftCdf;

                // Use the minimum of left and right tail probabilities
                double tailProb = Math.Min(leftCdf, rightCdf);

                // Avoid log(0) by clamping
                tailProb = Math.Max(tailProb, 1e-10);

                // Negative log probability (higher = more anomalous)
                combinedScore += -Math.Log(tailProb);
            }

            scores[i] = NumOps.FromDouble(combinedScore);
        }

        return scores;
    }

    private double ComputeECDF(double value, double[] sortedValues)
    {
        // Binary search to find the rank
        int left = 0;
        int right = sortedValues.Length - 1;

        while (left <= right)
        {
            int mid = (left + right) / 2;
            if (sortedValues[mid] <= value)
                left = mid + 1;
            else
                right = mid - 1;
        }

        // CDF = (number of values <= x) / n
        return (double)left / sortedValues.Length;
    }
}
