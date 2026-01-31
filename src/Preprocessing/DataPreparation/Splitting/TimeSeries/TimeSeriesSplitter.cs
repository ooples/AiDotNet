using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Time series splitter with expanding training window (no shuffling).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Time series data is ordered by time. Unlike regular data,
/// you CANNOT shuffle it because the order matters - the past predicts the future,
/// not the other way around.
/// </para>
/// <para>
/// <b>How It Works (Expanding Window):</b>
/// <code>
/// Split 1: Train[1-100]  → Test[101-120]
/// Split 2: Train[1-120]  → Test[121-140]
/// Split 3: Train[1-140]  → Test[141-160]
/// Split 4: Train[1-160]  → Test[161-180]
/// </code>
/// Notice: Training window grows, always starting from the beginning.
/// </para>
/// <para>
/// <b>Critical Rule:</b> Test data MUST always be after training data in time.
/// Never let future data leak into training!
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Stock prices, weather forecasting
/// - Any data where time order matters
/// - When you want to simulate real deployment
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TimeSeriesSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nSplits;
    private readonly int? _maxTrainSize;
    private readonly int? _testSize;
    private readonly int _gap;

    /// <summary>
    /// Creates a new time series splitter with expanding window.
    /// </summary>
    /// <param name="nSplits">Number of train/test splits. Default is 5.</param>
    /// <param name="maxTrainSize">Maximum training set size (null = no limit, keeps growing).</param>
    /// <param name="testSize">Fixed test set size (null = auto-calculate based on splits).</param>
    /// <param name="gap">Number of samples to skip between train and test (prevents leakage). Default is 0.</param>
    public TimeSeriesSplitter(
        int nSplits = 5,
        int? maxTrainSize = null,
        int? testSize = null,
        int gap = 0)
        : base(shuffle: false, randomSeed: 42)  // Never shuffle time series!
    {
        if (nSplits < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nSplits), "Number of splits must be at least 1.");
        }

        if (gap < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gap), "Gap cannot be negative.");
        }

        _nSplits = nSplits;
        _maxTrainSize = maxTrainSize;
        _testSize = testSize;
        _gap = gap;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description =>
        _maxTrainSize.HasValue
            ? $"Time Series Split ({_nSplits} splits, max train={_maxTrainSize})"
            : $"Time Series Split ({_nSplits} splits, expanding window)";

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

        // Calculate test size
        int testSize = _testSize ?? n / (_nSplits + 1);
        testSize = Math.Max(1, testSize);

        // Initial training size
        int minTrainSize = testSize;

        // Check we have enough data
        int minRequired = minTrainSize + _gap + testSize;
        if (n < minRequired)
        {
            throw new ArgumentException(
                $"Not enough samples ({n}) for {_nSplits} splits. " +
                $"Need at least {minRequired} samples.");
        }

        for (int split = 0; split < _nSplits; split++)
        {
            // Calculate split points
            int trainEnd = minTrainSize + split * testSize;
            int testStart = trainEnd + _gap;
            int testEnd = testStart + testSize;

            // Apply max train size if specified
            int trainStart = 0;
            if (_maxTrainSize.HasValue && trainEnd > _maxTrainSize.Value)
            {
                trainStart = trainEnd - _maxTrainSize.Value;
            }

            // Check bounds
            if (testEnd > n)
            {
                break; // Not enough data for more splits
            }

            var trainIndices = Enumerable.Range(trainStart, trainEnd - trainStart).ToArray();
            var testIndices = Enumerable.Range(testStart, testEnd - testStart).ToArray();

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split, totalFolds: _nSplits);
        }
    }
}
