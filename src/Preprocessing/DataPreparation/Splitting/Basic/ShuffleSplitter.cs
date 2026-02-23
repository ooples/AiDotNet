using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Basic;

/// <summary>
/// Monte Carlo cross-validation splitter that creates multiple random train/test splits.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Unlike K-Fold which systematically rotates through the data,
/// Monte Carlo CV (also called Shuffle-Split) randomly samples train/test sets multiple times.
/// Each split is independent - the same sample might be in the test set multiple times
/// across different splits.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Split 1: [Random 80%][Random 20%]
/// Split 2: [Random 80%][Random 20%]
/// Split 3: [Random 80%][Random 20%]
/// ... (some samples may repeat in test sets)
/// </code>
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you want flexibility in train/test proportions
/// - When data order doesn't matter
/// - As an alternative to K-Fold when you want more control
/// </para>
/// <para>
/// <b>Comparison to K-Fold:</b>
/// - K-Fold: Every sample appears in test exactly once
/// - Shuffle-Split: Samples may appear in test zero, one, or multiple times
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class ShuffleSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nSplits;
    private readonly double _testSize;
    private readonly double? _trainSize;

    /// <summary>
    /// Creates a new shuffle splitter (Monte Carlo CV).
    /// </summary>
    /// <param name="nSplits">Number of random splits to generate. Default is 10.</param>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="trainSize">Optional proportion for train set. If null, uses 1 - testSize.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ShuffleSplitter(
        int nSplits = 10,
        double testSize = 0.2,
        double? trainSize = null,
        int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (nSplits < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nSplits), "Number of splits must be at least 1.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (trainSize.HasValue && (trainSize.Value <= 0 || trainSize.Value >= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(trainSize), "Train size must be between 0 and 1.");
        }

        if (trainSize.HasValue && trainSize.Value + testSize > 1)
        {
            throw new ArgumentException("Train size + test size cannot exceed 1.");
        }

        _nSplits = nSplits;
        _testSize = testSize;
        _trainSize = trainSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Shuffle-Split ({_nSplits} splits, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        // Return first split for single-split call
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int testSize = Math.Max(1, (int)(nSamples * _testSize));
        int trainSize = _trainSize.HasValue
            ? Math.Max(1, (int)(nSamples * _trainSize.Value))
            : nSamples - testSize;

        for (int split = 0; split < _nSplits; split++)
        {
            // Get fresh shuffled indices for each split
            var indices = GetIndices(nSamples);
            ShuffleIndices(indices);

            var trainIndices = indices.Take(trainSize).ToArray();
            var testIndices = indices.Skip(trainSize).Take(testSize).ToArray();

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split, totalFolds: _nSplits);
        }
    }
}
