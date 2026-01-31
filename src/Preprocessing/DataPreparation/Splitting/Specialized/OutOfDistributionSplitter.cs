using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;

/// <summary>
/// Out-of-Distribution (OOD) splitter that places outlier samples in the test set.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This splitter identifies samples that are "unusual" or
/// "outliers" compared to the bulk of the data, and places them in the test set.
/// This helps evaluate how well your model handles edge cases.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Compute the centroid (average) of all samples
/// 2. Calculate each sample's distance from the centroid
/// 3. Place the most distant samples (outliers) in the test set
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Testing model robustness to unusual inputs
/// - Evaluating edge case handling
/// - Safety-critical applications where OOD detection matters
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class OutOfDistributionSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly double _oodPercentile;

    /// <summary>
    /// Creates a new Out-of-Distribution splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="oodPercentile">Percentile threshold for OOD samples (0-1). Default is 0.9 (top 10% most distant).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public OutOfDistributionSplitter(
        double testSize = 0.2,
        double oodPercentile = 0.9,
        int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (oodPercentile < 0 || oodPercentile > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(oodPercentile), "OOD percentile must be between 0 and 1.");
        }

        _testSize = testSize;
        _oodPercentile = oodPercentile;
    }

    /// <inheritdoc/>
    public override string Description => $"Out-of-Distribution split ({_testSize * 100:F0}% test, top {(1 - _oodPercentile) * 100:F0}% OOD)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        // Compute centroid
        var centroid = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++)
        {
            double sum = 0;
            for (int i = 0; i < nSamples; i++)
            {
                sum += Convert.ToDouble(X[i, j]);
            }
            centroid[j] = sum / nSamples;
        }

        // Compute standard deviation per feature for normalization
        var stdDev = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++)
        {
            double sumSq = 0;
            for (int i = 0; i < nSamples; i++)
            {
                double diff = Convert.ToDouble(X[i, j]) - centroid[j];
                sumSq += diff * diff;
            }
            stdDev[j] = Math.Sqrt(sumSq / nSamples);
            if (stdDev[j] < 1e-10) stdDev[j] = 1; // Avoid division by zero
        }

        // Compute Mahalanobis-like distance for each sample
        var distances = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            double dist = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                double normalizedDiff = (Convert.ToDouble(X[i, j]) - centroid[j]) / stdDev[j];
                dist += normalizedDiff * normalizedDiff;
            }
            distances[i] = Math.Sqrt(dist);
        }

        // Sort by distance (most distant = OOD)
        var sortedIndices = Enumerable.Range(0, nSamples)
            .OrderByDescending(i => distances[i])
            .ToArray();

        // OOD samples go to test, others can go to either based on testSize
        int oodCount = (int)((1 - _oodPercentile) * nSamples);
        var oodIndices = new HashSet<int>(sortedIndices.Take(oodCount));

        var testIndices = new List<int>();
        var trainIndices = new List<int>();

        // First, add OOD samples to test
        foreach (int idx in sortedIndices.Take(oodCount))
        {
            if (testIndices.Count < targetTestSize)
            {
                testIndices.Add(idx);
            }
            else
            {
                trainIndices.Add(idx);
            }
        }

        // Then add remaining samples
        var remainingIndices = sortedIndices.Skip(oodCount).ToArray();
        ShuffleIndices(remainingIndices);

        foreach (int idx in remainingIndices)
        {
            if (testIndices.Count < targetTestSize)
            {
                testIndices.Add(idx);
            }
            else
            {
                trainIndices.Add(idx);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
