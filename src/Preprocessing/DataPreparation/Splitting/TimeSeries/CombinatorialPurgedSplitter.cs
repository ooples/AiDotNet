using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Combinatorial Purged Cross-Validation splitter for time series data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is an advanced cross-validation method designed specifically
/// for financial time series where data leakage can invalidate backtests.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Divide data into n groups based on time periods
/// 2. Generate all combinations of k groups for testing
/// 3. For each combination, remaining groups form training set
/// 4. Apply purging (remove samples near test boundaries) and embargo (block after test)
/// </para>
/// <para>
/// <b>Use Cases:</b>
/// - Financial backtesting where overlapping data causes leakage
/// - Time series with autocorrelation that persists across samples
/// - Generating many independent train/test splits from limited time series
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class CombinatorialPurgedSplitter<T> : DataSplitterBase<T>
{
    private readonly int _nGroups;
    private readonly int _nTestGroups;
    private readonly int _purgeSize;
    private readonly int _embargoSize;

    /// <summary>
    /// Creates a new Combinatorial Purged CV splitter.
    /// </summary>
    /// <param name="nGroups">Number of time groups to divide data into. Default is 6.</param>
    /// <param name="nTestGroups">Number of groups to use for testing in each split. Default is 2.</param>
    /// <param name="purgeSize">Number of samples to remove near test boundaries. Default is 0.</param>
    /// <param name="embargoSize">Number of samples to block after test period. Default is 0.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public CombinatorialPurgedSplitter(
        int nGroups = 6,
        int nTestGroups = 2,
        int purgeSize = 0,
        int embargoSize = 0,
        int randomSeed = 42)
        : base(shuffle: false, randomSeed) // Never shuffle time series
    {
        if (nGroups < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(nGroups), "Number of groups must be at least 2.");
        }

        if (nTestGroups < 1 || nTestGroups >= nGroups)
        {
            throw new ArgumentOutOfRangeException(nameof(nTestGroups),
                $"Number of test groups must be between 1 and {nGroups - 1}.");
        }

        if (purgeSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(purgeSize), "Purge size cannot be negative.");
        }

        if (embargoSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embargoSize), "Embargo size cannot be negative.");
        }

        _nGroups = nGroups;
        _nTestGroups = nTestGroups;
        _purgeSize = purgeSize;
        _embargoSize = embargoSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => Combinations(_nGroups, _nTestGroups);

    /// <inheritdoc/>
    public override string Description => $"Combinatorial Purged CV ({_nGroups} groups, {_nTestGroups} test, purge={_purgeSize}, embargo={_embargoSize})";

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
        int groupSize = nSamples / _nGroups;

        if (groupSize < 1)
        {
            throw new ArgumentException(
                $"Not enough samples ({nSamples}) for {_nGroups} groups.");
        }

        // Calculate group boundaries
        var groupBoundaries = new int[_nGroups + 1];
        for (int i = 0; i <= _nGroups; i++)
        {
            groupBoundaries[i] = Math.Min(i * groupSize, nSamples);
        }
        groupBoundaries[_nGroups] = nSamples; // Ensure last group gets remaining samples

        // Generate all combinations of test groups
        var allCombinations = GetCombinations(Enumerable.Range(0, _nGroups).ToArray(), _nTestGroups);

        int splitIndex = 0;
        foreach (var testGroupIndices in allCombinations)
        {
            var testIndices = new List<int>();
            var trainIndices = new List<int>();

            // Collect test indices
            foreach (int groupIdx in testGroupIndices)
            {
                for (int i = groupBoundaries[groupIdx]; i < groupBoundaries[groupIdx + 1]; i++)
                {
                    testIndices.Add(i);
                }
            }

            // Collect train indices with purging and embargo
            var testGroupSet = new HashSet<int>(testGroupIndices);
            for (int groupIdx = 0; groupIdx < _nGroups; groupIdx++)
            {
                if (testGroupSet.Contains(groupIdx))
                {
                    continue;
                }

                int groupStart = groupBoundaries[groupIdx];
                int groupEnd = groupBoundaries[groupIdx + 1];

                // Check if this group is adjacent to a test group
                bool adjacentToTestBefore = testGroupSet.Contains(groupIdx - 1);
                bool adjacentToTestAfter = testGroupSet.Contains(groupIdx + 1);

                // Apply purging at boundaries
                int effectiveStart = groupStart;
                int effectiveEnd = groupEnd;

                if (adjacentToTestBefore)
                {
                    effectiveStart = Math.Min(groupStart + _embargoSize, groupEnd);
                }

                if (adjacentToTestAfter)
                {
                    effectiveEnd = Math.Max(groupEnd - _purgeSize, effectiveStart);
                }

                for (int i = effectiveStart; i < effectiveEnd; i++)
                {
                    trainIndices.Add(i);
                }
            }

            if (trainIndices.Count == 0)
            {
                continue; // Skip if no valid training data
            }

            yield return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray(),
                foldIndex: splitIndex, totalFolds: NumSplits);

            splitIndex++;
        }
    }

    private static int Combinations(int n, int k)
    {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;

        int result = 1;
        for (int i = 0; i < k; i++)
        {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    private static IEnumerable<int[]> GetCombinations(int[] elements, int k)
    {
        if (k == 0)
        {
            yield return Array.Empty<int>();
            yield break;
        }

        if (elements.Length == k)
        {
            yield return elements.ToArray();
            yield break;
        }

        for (int i = 0; i <= elements.Length - k; i++)
        {
            int head = elements[i];
            int[] tail = elements.Skip(i + 1).ToArray();

            foreach (var combination in GetCombinations(tail, k - 1))
            {
                yield return new[] { head }.Concat(combination).ToArray();
            }
        }
    }
}
