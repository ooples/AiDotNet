using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Purged K-Fold: K-Fold with temporal purging to prevent data leakage in financial/time-dependent data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Purged K-Fold adds a gap between training and test sets to prevent
/// temporal data leakage:
/// <list type="bullet">
/// <item>Standard K-Fold can leak information when observations overlap in time</item>
/// <item>Purging removes training samples that are temporally close to test samples</item>
/// <item>Essential for financial data where future information must not influence past predictions</item>
/// </list>
/// </para>
/// <para>
/// <b>Example:</b> If you're predicting stock returns using 5-day windows:
/// <list type="bullet">
/// <item>Test period: Day 100-110</item>
/// <item>Without purging: Training might include days 95-99 (overlapping windows!)</item>
/// <item>With purging: Training excludes days 95-114 (5-day buffer on each side)</item>
/// </list>
/// </para>
/// </remarks>
public class PurgedKFoldStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _k;
    private readonly int _purgeGap;
    private readonly int[]? _timeIndices;

    /// <summary>
    /// Initializes Purged K-Fold cross-validation.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="purgeGap">Number of time steps to purge on each side of test set. Default is 1.</param>
    /// <param name="timeIndices">Optional array of time indices. If null, assumes samples are already sorted by time.</param>
    public PurgedKFoldStrategy(int k = 5, int purgeGap = 1, int[]? timeIndices = null)
    {
        if (k < 2) throw new ArgumentException("K must be at least 2.", nameof(k));
        if (purgeGap < 0) throw new ArgumentException("Purge gap cannot be negative.", nameof(purgeGap));

        _k = k;
        _purgeGap = purgeGap;
        _timeIndices = timeIndices;
    }

    public string Name => $"Purged {_k}-Fold";
    public int NumSplits => _k;
    public string Description => $"Purged {_k}-fold with {_purgeGap} time step gap to prevent temporal leakage.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize <= 0)
            throw new ArgumentException("Data size must be positive.", nameof(dataSize));
        if (dataSize < _k)
            throw new ArgumentException($"Cannot have {_k} folds with only {dataSize} samples.", nameof(dataSize));

        // Validate timeIndices length if provided
        if (_timeIndices != null && _timeIndices.Length != dataSize)
            throw new ArgumentException($"Time indices length ({_timeIndices.Length}) must match data size ({dataSize}).", nameof(dataSize));

        // Get time-ordered indices
        int[] timeOrder;
        if (_timeIndices != null)
        {
            // Sort sample indices by their time values
            var indexedTimes = _timeIndices.Select((t, i) => (time: t, index: i)).OrderBy(x => x.time).ToArray();
            timeOrder = indexedTimes.Select(x => x.index).ToArray();
        }
        else
        {
            // Assume data is already time-ordered
            timeOrder = new int[dataSize];
            for (int i = 0; i < dataSize; i++) timeOrder[i] = i;
        }

        // Calculate fold boundaries
        int baseFoldSize = dataSize / _k;
        int remainder = dataSize % _k;

        int startIdx = 0;
        for (int fold = 0; fold < _k; fold++)
        {
            int foldSize = baseFoldSize + (fold < remainder ? 1 : 0);
            int endIdx = startIdx + foldSize;

            // Test indices are this fold's time-ordered samples
            var testIndices = new int[foldSize];
            for (int i = 0; i < foldSize; i++)
                testIndices[i] = timeOrder[startIdx + i];

            // Determine purge boundaries
            int purgeStart = Math.Max(0, startIdx - _purgeGap);
            int purgeEnd = Math.Min(dataSize, endIdx + _purgeGap);

            // Train indices are all samples outside the purge zone
            var trainIndices = new List<int>();
            for (int i = 0; i < purgeStart; i++)
                trainIndices.Add(timeOrder[i]);
            for (int i = purgeEnd; i < dataSize; i++)
                trainIndices.Add(timeOrder[i]);

            // Always yield the fold - if training set is empty, caller should handle appropriately
            // rather than silently skipping folds (which could mislead about total folds)
            yield return (trainIndices.ToArray(), testIndices);

            startIdx = endIdx;
        }
    }
}
