using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Time series splitter with a gap (purge) between training and test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In time series, data points close together are often correlated.
/// If your training data is right next to your test data, some information might "leak"
/// from the test period into training.
/// </para>
/// <para>
/// <b>The Gap Solution:</b>
/// Adding a gap (also called "purge") between train and test ensures no leakage:
/// <code>
/// Split 1: [Train][GAP][Test]
///          [1-80][81-90][91-100]
/// </code>
/// The gap samples are neither trained on nor tested - they're a buffer zone.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When features depend on recent past values (rolling averages, momentum)
/// - Financial time series with overlapping windows
/// - When data points are auto-correlated
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BlockedTimeSeriesSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nSplits;
    private readonly int _gap;

    /// <summary>
    /// Creates a new blocked time series splitter.
    /// </summary>
    /// <param name="nSplits">Number of splits. Default is 5.</param>
    /// <param name="gap">Number of samples to skip between train and test. Default is 0.</param>
    public BlockedTimeSeriesSplitter(int nSplits = 5, int gap = 0)
        : base(shuffle: false, randomSeed: 42)
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
        _gap = gap;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Blocked Time Series ({_nSplits} splits, gap={_gap})";

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
        int testSize = n / (_nSplits + 1);

        if (testSize < 1)
        {
            throw new ArgumentException($"Not enough samples ({n}) for {_nSplits} splits.");
        }

        for (int split = 0; split < _nSplits; split++)
        {
            int trainEnd = testSize * (split + 1);
            int testStart = trainEnd + _gap;
            int testEnd = testStart + testSize;

            if (testEnd > n)
            {
                break;
            }

            var trainIndices = Enumerable.Range(0, trainEnd).ToArray();
            var testIndices = Enumerable.Range(testStart, testSize).ToArray();

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split, totalFolds: _nSplits);
        }
    }
}
