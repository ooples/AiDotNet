using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;

/// <summary>
/// Adversarial validation splitter that identifies samples most similar to test distribution.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sometimes your training and test data come from different distributions
/// (e.g., train from 2022, test from 2023). Adversarial validation helps detect this problem.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Train a classifier to distinguish train vs test samples
/// 2. If the classifier performs well (AUC > 0.5), there's distribution shift
/// 3. Use the classifier's predictions to create a more realistic validation set
/// 4. Put samples most "test-like" into validation
/// </para>
/// <para>
/// <b>Practical Use:</b>
/// This splitter doesn't actually train a classifier - it creates indices that you can use
/// after running adversarial validation externally. You provide the probability scores
/// from your adversarial classifier.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Kaggle competitions where public/private test differs from train
/// - Time-based production scenarios
/// - Any situation with potential train/test distribution shift
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class AdversarialValidationSplitter<T> : DataSplitterBase<T>
{
    private readonly double[] _testProbabilities;
    private readonly double _testSize;

    /// <summary>
    /// Creates an adversarial validation splitter.
    /// </summary>
    /// <param name="testProbabilities">
    /// For each sample, the probability it belongs to the test distribution
    /// (output from an adversarial classifier).
    /// </param>
    /// <param name="testSize">
    /// Proportion of samples to use as validation (selecting most test-like samples).
    /// Default is 0.2 (20%).
    /// </param>
    /// <param name="randomSeed">Random seed for any ties. Default is 42.</param>
    public AdversarialValidationSplitter(double[] testProbabilities, double testSize = 0.2, int randomSeed = 42)
        : base(shuffle: false, randomSeed)
    {
        if (testProbabilities is null || testProbabilities.Length == 0)
        {
            throw new ArgumentNullException(nameof(testProbabilities),
                "Test probabilities array cannot be null or empty.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testProbabilities = testProbabilities;
        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Adversarial Validation split ({_testSize * 100:F0}% test-like samples)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (X.Rows != _testProbabilities.Length)
        {
            throw new ArgumentException(
                $"Test probabilities length ({_testProbabilities.Length}) must match number of samples ({X.Rows}).");
        }

        int nSamples = X.Rows;
        int testSize = Math.Max(1, (int)(nSamples * _testSize));

        // Sort indices by test probability (descending)
        var sortedIndices = Enumerable.Range(0, nSamples)
            .OrderByDescending(i => _testProbabilities[i])
            .ToArray();

        // Most test-like samples go to validation/test
        var testIndices = sortedIndices.Take(testSize).ToArray();
        var trainIndices = sortedIndices.Skip(testSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }
}
