using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// K-Fold cross-validation splitter that divides data into k equal folds.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> K-Fold cross-validation is one of the most important techniques
/// in machine learning for getting reliable performance estimates.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Divide data into k equal parts (folds)
/// 2. For each fold:
///    - Use that fold as the test set
///    - Use the remaining k-1 folds as training
/// 3. Average the results across all k evaluations
/// </para>
/// <para>
/// <b>Visual Example (5-Fold):</b>
/// <code>
/// Fold 1: [Test ][Train][Train][Train][Train]
/// Fold 2: [Train][Test ][Train][Train][Train]
/// Fold 3: [Train][Train][Test ][Train][Train]
/// Fold 4: [Train][Train][Train][Test ][Train]
/// Fold 5: [Train][Train][Train][Train][Test ]
/// </code>
/// </para>
/// <para>
/// <b>Industry Standard:</b>
/// - k=5 for large datasets (>10,000 samples)
/// - k=10 for smaller datasets (1,000-10,000 samples)
/// </para>
/// <para>
/// <b>Why Use K-Fold?</b>
/// - More reliable than a single train/test split
/// - Every sample gets tested exactly once
/// - Good for limited data where you can't afford to hold out a large test set
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class KFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;

    /// <summary>
    /// Creates a new K-Fold cross-validation splitter.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5 (industry standard for large datasets).</param>
    /// <param name="shuffle">Whether to shuffle data before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <exception cref="ArgumentOutOfRangeException">If k is less than 2.</exception>
    public KFoldSplitter(int k = 5, bool shuffle = true, int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "Number of folds (k) must be at least 2. Use 5 or 10 for most cases.");
        }

        _k = k;
    }

    /// <inheritdoc/>
    public override int NumSplits => _k;

    /// <inheritdoc/>
    public override string Description => $"{_k}-Fold cross-validation";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        // Return first fold for single-split call
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;

        if (_k > nSamples)
        {
            throw new ArgumentException(
                $"Cannot have more folds ({_k}) than samples ({nSamples}). " +
                $"Use LeaveOneOutSplitter for very small datasets.");
        }

        var indices = GetShuffledIndices(nSamples);

        // Calculate fold sizes - handle unequal division
        int baseFoldSize = nSamples / _k;
        int remainder = nSamples % _k;

        int currentIndex = 0;
        for (int fold = 0; fold < _k; fold++)
        {
            // Some folds get one extra sample to handle unequal division
            int foldSize = baseFoldSize + (fold < remainder ? 1 : 0);

            // Test indices for this fold
            var testIndices = indices.Skip(currentIndex).Take(foldSize).ToArray();

            // Train indices: everything except this fold
            var trainIndices = new List<int>();
            for (int i = 0; i < currentIndex; i++)
            {
                trainIndices.Add(indices[i]);
            }
            for (int i = currentIndex + foldSize; i < nSamples; i++)
            {
                trainIndices.Add(indices[i]);
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices,
                foldIndex: fold, totalFolds: _k);

            currentIndex += foldSize;
        }
    }
}
