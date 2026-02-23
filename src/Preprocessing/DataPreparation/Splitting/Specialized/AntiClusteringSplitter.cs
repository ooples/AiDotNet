using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;

/// <summary>
/// Anti-clustering splitter that maximizes diversity within each split.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> While clustering groups similar items together,
/// anti-clustering does the opposite - it ensures each group (train/test)
/// contains a diverse mix of samples covering the entire data space.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Compute pairwise distances between all samples
/// 2. Iteratively assign samples to train/test sets
/// 3. At each step, maximize the diversity within each set
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want both train and test to be representative
/// - Survey sampling where you want diverse groups
/// - A/B testing to ensure comparable groups
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class AntiClusteringSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;

    /// <summary>
    /// Creates a new anti-clustering splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle initial order. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public AntiClusteringSplitter(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Anti-Clustering split ({_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        // Compute feature centroids for diversity calculation
        var centroids = new double[nFeatures];
        for (int j = 0; j < nFeatures; j++)
        {
            double sum = 0;
            for (int i = 0; i < nSamples; i++)
            {
                sum += Convert.ToDouble(X[i, j]);
            }
            centroids[j] = sum / nSamples;
        }

        // Calculate distance from centroid for each sample (for diversity scoring)
        var distances = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            double dist = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                double diff = Convert.ToDouble(X[i, j]) - centroids[j];
                dist += diff * diff;
            }
            distances[i] = Math.Sqrt(dist);
        }

        // Sort samples by distance from centroid
        var sortedIndices = Enumerable.Range(0, nSamples)
            .OrderBy(i => distances[i])
            .ToArray();

        if (_shuffle)
        {
            // Add some randomness while maintaining overall diversity
            for (int i = 0; i < sortedIndices.Length - 1; i += 2)
            {
                if (_random.NextDouble() < 0.3) // 30% chance to swap adjacent pairs
                {
                    (sortedIndices[i], sortedIndices[i + 1]) = (sortedIndices[i + 1], sortedIndices[i]);
                }
            }
        }

        // Distribute samples alternately to ensure both sets are diverse
        // Use round-robin assignment based on distance bins
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        int testFrequency = (int)Math.Ceiling(1.0 / _testSize);
        int counter = 0;

        foreach (int idx in sortedIndices)
        {
            if (testIndices.Count < targetTestSize && counter % testFrequency == 0)
            {
                testIndices.Add(idx);
            }
            else
            {
                trainIndices.Add(idx);
            }
            counter++;
        }

        // Ensure we have exact test size
        while (testIndices.Count < targetTestSize && trainIndices.Count > 0)
        {
            int idx = trainIndices[trainIndices.Count - 1];
            trainIndices.RemoveAt(trainIndices.Count - 1);
            testIndices.Add(idx);
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
