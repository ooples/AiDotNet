using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Basic;

/// <summary>
/// Creates multiple independent holdout test sets for robust evaluation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This splitter creates multiple independent train/test splits,
/// where each test set (holdout) is completely separate. Unlike K-Fold where test sets
/// don't overlap, holdout splits can have different random samples each time.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When you need multiple independent evaluations
/// - For statistical significance testing
/// - When you want to assess model stability across different splits
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class HoldoutSplitter<T> : DataSplitterBase<T>
{
    private readonly int _numHoldouts;
    private readonly double _testSize;

    /// <summary>
    /// Creates a new holdout splitter.
    /// </summary>
    /// <param name="numHoldouts">Number of holdout test sets to create. Default is 5.</param>
    /// <param name="testSize">Proportion for each test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public HoldoutSplitter(
        int numHoldouts = 5,
        double testSize = 0.2,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (numHoldouts < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numHoldouts), "Number of holdouts must be at least 1.");
        }

        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _numHoldouts = numHoldouts;
        _testSize = testSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _numHoldouts;

    /// <inheritdoc/>
    public override string Description => $"Holdout ({_numHoldouts} holdouts, {_testSize * 100:F0}% test each)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int testSize = Math.Max(1, (int)(nSamples * _testSize));
        int trainSize = nSamples - testSize;

        for (int holdout = 0; holdout < _numHoldouts; holdout++)
        {
            var indices = GetIndices(nSamples);
            ShuffleIndices(indices);

            var trainIndices = indices.Take(trainSize).ToArray();
            var testIndices = indices.Skip(trainSize).ToArray();

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: holdout, totalFolds: _numHoldouts);
        }
    }
}
