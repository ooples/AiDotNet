using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Anchored walk-forward validation with a fixed starting point.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is similar to regular walk-forward, but the training
/// always starts from the same point (anchored) rather than sliding.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Step 1: Train[0-100]  → Test[101-110]
/// Step 2: Train[0-110]  → Test[111-120]
/// Step 3: Train[0-120]  → Test[121-130]
/// </code>
/// Training always starts at index 0 (the anchor).
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When historical data remains relevant
/// - When you want maximum training data
/// - Traditional time series forecasting
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class AnchoredWalkForwardSplitter<T> : DataSplitterBase<T>
{
    private readonly int _initialTrainSize;
    private readonly int _testSize;
    private readonly int _stepSize;
    private int _nSplits;

    /// <summary>
    /// Creates a new anchored walk-forward splitter.
    /// </summary>
    /// <param name="initialTrainSize">Initial training set size.</param>
    /// <param name="testSize">Size of each test period.</param>
    /// <param name="stepSize">How much to advance each iteration (null = testSize).</param>
    public AnchoredWalkForwardSplitter(int initialTrainSize, int testSize, int? stepSize = null)
        : base(shuffle: false, randomSeed: 42)
    {
        if (initialTrainSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(initialTrainSize), "Initial train size must be at least 1.");
        }

        if (testSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be at least 1.");
        }

        _initialTrainSize = initialTrainSize;
        _testSize = testSize;
        _stepSize = stepSize ?? testSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Anchored Walk-Forward (initial={_initialTrainSize}, test={_testSize})";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int n = X.Rows;
        _nSplits = 0;

        int trainEnd = _initialTrainSize;
        int split = 0;

        while (trainEnd + _testSize <= n)
        {
            // Anchored: always start from 0
            var trainIndices = Enumerable.Range(0, trainEnd).ToArray();
            var testIndices = Enumerable.Range(trainEnd, _testSize).ToArray();

            _nSplits++;

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split++, totalFolds: _nSplits);

            trainEnd += _stepSize;
        }
    }
}
