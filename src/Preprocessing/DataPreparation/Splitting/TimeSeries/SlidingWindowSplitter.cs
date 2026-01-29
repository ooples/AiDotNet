using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Sliding window splitter with fixed-size training window that moves through time.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Unlike the expanding window time series split, a sliding window
/// keeps the training size fixed and "slides" forward through time.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Split 1: Train[1-100]   → Test[101-120]
/// Split 2: Train[21-120]  → Test[121-140]
/// Split 3: Train[41-140]  → Test[141-160]
/// Split 4: Train[61-160]  → Test[161-180]
/// </code>
/// Notice: Training window is always the same size, just shifted forward.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When old data becomes less relevant (concept drift)
/// - When you want fixed compute per training iteration
/// - When training on recent data only is preferred
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SlidingWindowSplitter<T> : DataSplitterBase<T>
{
    private readonly int _trainSize;
    private readonly int _testSize;
    private readonly int _stepSize;
    private readonly int _gap;
    private int _nSplits;

    /// <summary>
    /// Creates a new sliding window splitter.
    /// </summary>
    /// <param name="trainSize">Fixed size of the training window.</param>
    /// <param name="testSize">Size of the test window.</param>
    /// <param name="stepSize">How much to slide the window forward each time. If null, uses testSize.</param>
    /// <param name="gap">Gap between train and test (prevents leakage). Default is 0.</param>
    public SlidingWindowSplitter(
        int trainSize,
        int testSize,
        int? stepSize = null,
        int gap = 0)
        : base(shuffle: false, randomSeed: 42)
    {
        if (trainSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(trainSize), "Train size must be at least 1.");
        }

        if (testSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be at least 1.");
        }

        if (gap < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gap), "Gap cannot be negative.");
        }

        _trainSize = trainSize;
        _testSize = testSize;
        _stepSize = stepSize ?? testSize;
        _gap = gap;

        if (_stepSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stepSize), "Step size must be at least 1.");
        }
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Sliding Window (train={_trainSize}, test={_testSize}, step={_stepSize})";

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

        // Calculate number of possible splits
        int minRequired = _trainSize + _gap + _testSize;
        if (n < minRequired)
        {
            throw new ArgumentException(
                $"Not enough samples ({n}) for sliding window. Need at least {minRequired}.");
        }

        _nSplits = 0;
        int split = 0;
        int trainStart = 0;

        while (true)
        {
            int trainEnd = trainStart + _trainSize;
            int testStart = trainEnd + _gap;
            int testEnd = testStart + _testSize;

            if (testEnd > n)
            {
                break;
            }

            _nSplits++;

            var trainIndices = Enumerable.Range(trainStart, _trainSize).ToArray();
            var testIndices = Enumerable.Range(testStart, _testSize).ToArray();

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split++, totalFolds: _nSplits);

            trainStart += _stepSize;
        }
    }
}
