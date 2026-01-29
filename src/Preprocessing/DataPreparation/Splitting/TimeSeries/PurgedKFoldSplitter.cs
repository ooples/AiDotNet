using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Purged K-Fold cross-validation for time series with overlapping labels.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In financial time series, features often look into the future
/// (e.g., 5-day rolling average). This creates a problem: if your training data
/// includes day 95 and your test includes day 100, the rolling average for day 100
/// uses data from days 96-100, which includes your training period!
/// </para>
/// <para>
/// <b>The Solution - Purging:</b>
/// Remove (purge) samples around the test period that could leak information:
/// <code>
/// [Train][Purged][Test][Embargo][Train continues...]
/// </code>
/// - Purge: Remove samples BEFORE test that could contaminate it
/// - Embargo: Remove samples AFTER test that could be affected by it
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Financial time series with overlapping windows
/// - Any time series where features depend on future values
/// - When you calculate rolling statistics
/// </para>
/// <para>
/// <b>Reference:</b> Based on Marcos López de Prado's methodology from
/// "Advances in Financial Machine Learning"
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PurgedKFoldSplitter<T> : DataSplitterBase<T>
{
    private readonly int _k;
    private readonly int _purgeSize;
    private readonly int _embargoSize;

    /// <summary>
    /// Creates a new Purged K-Fold splitter for financial time series.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="purgeSize">Samples to remove before each test fold.</param>
    /// <param name="embargoSize">Samples to remove after each test fold.</param>
    public PurgedKFoldSplitter(int k = 5, int purgeSize = 0, int embargoSize = 0)
        : base(shuffle: false, randomSeed: 42)
    {
        if (k < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "Number of folds must be at least 2.");
        }

        if (purgeSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(purgeSize), "Purge size cannot be negative.");
        }

        if (embargoSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embargoSize), "Embargo size cannot be negative.");
        }

        _k = k;
        _purgeSize = purgeSize;
        _embargoSize = embargoSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _k;

    /// <inheritdoc/>
    public override string Description => $"Purged {_k}-Fold (purge={_purgeSize}, embargo={_embargoSize})";

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
        int foldSize = n / _k;

        if (foldSize <= _purgeSize + _embargoSize)
        {
            throw new ArgumentException(
                $"Not enough samples. Fold size ({foldSize}) must be larger than " +
                $"purge ({_purgeSize}) + embargo ({_embargoSize}).");
        }

        for (int fold = 0; fold < _k; fold++)
        {
            // Calculate test fold boundaries
            int testStart = fold * foldSize;
            int testEnd = (fold == _k - 1) ? n : (fold + 1) * foldSize;

            // Calculate purge/embargo boundaries
            int purgeStart = Math.Max(0, testStart - _purgeSize);
            int embargoEnd = Math.Min(n, testEnd + _embargoSize);

            var testIndices = Enumerable.Range(testStart, testEnd - testStart).ToArray();

            // Train indices: everything except test, purge zone, and embargo zone
            var trainIndices = new List<int>();
            for (int i = 0; i < purgeStart; i++)
            {
                trainIndices.Add(i);
            }
            for (int i = embargoEnd; i < n; i++)
            {
                trainIndices.Add(i);
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices,
                foldIndex: fold, totalFolds: _k);
        }
    }
}
