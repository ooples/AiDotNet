using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;

/// <summary>
/// Similarity-based splitter that splits data based on sample similarity scores.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This splitter uses similarity between samples to create splits.
/// Samples that are very similar to training data are placed in the test set to evaluate
/// interpolation, while dissimilar samples test extrapolation capability.
/// </para>
/// <para>
/// <b>Modes:</b>
/// - Interpolation Test: Test samples similar to training (tests within-distribution generalization)
/// - Extrapolation Test: Test samples dissimilar to training (tests out-of-distribution performance)
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want to specifically test interpolation vs extrapolation
/// - For robustness evaluation
/// - When similarity structure in data is meaningful
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SimilarityBasedSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly double _threshold;
    private readonly bool _testSimilar;

    /// <summary>
    /// Creates a new similarity-based splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="threshold">Similarity threshold (0-1). Default is 0.5.</param>
    /// <param name="testSimilar">If true, similar samples go to test; if false, dissimilar go to test. Default is false.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SimilarityBasedSplitter(
        double testSize = 0.2,
        double threshold = 0.5,
        bool testSimilar = false,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (threshold < 0 || threshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be between 0 and 1.");
        }

        _testSize = testSize;
        _threshold = threshold;
        _testSimilar = testSimilar;
    }

    /// <inheritdoc/>
    public override string Description => $"Similarity-Based split ({(_testSimilar ? "similar" : "dissimilar")} test, {_testSize * 100:F0}%)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        // Compute pairwise cosine similarities
        var similarities = new double[nSamples, nSamples];
        var norms = new double[nSamples];

        // Calculate norms
        for (int i = 0; i < nSamples; i++)
        {
            double norm = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                double val = Convert.ToDouble(X[i, j]);
                norm += val * val;
            }
            norms[i] = Math.Sqrt(norm);
        }

        // Calculate similarities
        for (int i = 0; i < nSamples; i++)
        {
            similarities[i, i] = 1.0;
            for (int j = i + 1; j < nSamples; j++)
            {
                double dot = 0;
                for (int k = 0; k < nFeatures; k++)
                {
                    dot += Convert.ToDouble(X[i, k]) * Convert.ToDouble(X[j, k]);
                }

                double sim = (norms[i] > 0 && norms[j] > 0) ? dot / (norms[i] * norms[j]) : 0;
                similarities[i, j] = sim;
                similarities[j, i] = sim;
            }
        }

        // Calculate average similarity for each sample
        var avgSimilarity = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            double sum = 0;
            for (int j = 0; j < nSamples; j++)
            {
                if (i != j) sum += similarities[i, j];
            }
            avgSimilarity[i] = sum / (nSamples - 1);
        }

        // Sort by similarity
        var sortedIndices = Enumerable.Range(0, nSamples)
            .OrderBy(i => _testSimilar ? -avgSimilarity[i] : avgSimilarity[i])
            .ToArray();

        if (_shuffle)
        {
            // Partial shuffle to maintain overall ordering while adding randomness
            for (int i = 0; i < sortedIndices.Length - 1; i++)
            {
                if (_random.NextDouble() < 0.2) // 20% chance to swap with neighbor
                {
                    (sortedIndices[i], sortedIndices[i + 1]) = (sortedIndices[i + 1], sortedIndices[i]);
                }
            }
        }

        // First targetTestSize samples go to test (most similar/dissimilar based on mode)
        var testIndices = sortedIndices.Take(targetTestSize).ToArray();
        var trainIndices = sortedIndices.Skip(targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }
}
