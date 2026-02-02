using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Time Series Split: expanding window cross-validation that respects temporal order.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time series data cannot be shuffled because order matters.
/// Time Series Split uses an expanding training window:
/// <list type="bullet">
/// <item>Training data always comes BEFORE validation data</item>
/// <item>Simulates real-world forecasting where you predict future from past</item>
/// <item>Avoids data leakage from future information</item>
/// </list>
/// </para>
/// <para>
/// <b>Example with 5 splits:</b>
/// <code>
/// Split 1: Train [0-19] | Validate [20-39]
/// Split 2: Train [0-39] | Validate [40-59]
/// Split 3: Train [0-59] | Validate [60-79]
/// Split 4: Train [0-79] | Validate [80-99]
/// Split 5: Train [0-99] | Validate [100-119]
/// </code>
/// Notice how training data grows with each split (expanding window).
/// </para>
/// </remarks>
public class TimeSeriesSplitStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numSplits;
    private readonly int? _maxTrainSize;
    private readonly int? _testSize;
    private readonly int _gap;

    /// <summary>
    /// Initializes Time Series Split cross-validation.
    /// </summary>
    /// <param name="numSplits">Number of splits. Default is 5.</param>
    /// <param name="maxTrainSize">Maximum training set size. If null, uses all available prior data.</param>
    /// <param name="testSize">Fixed test set size. If null, automatically calculated.</param>
    /// <param name="gap">Number of samples to skip between train and test sets. Default is 0.</param>
    public TimeSeriesSplitStrategy(int numSplits = 5, int? maxTrainSize = null, int? testSize = null, int gap = 0)
    {
        if (numSplits < 2) throw new ArgumentException("Number of splits must be at least 2.", nameof(numSplits));
        if (gap < 0) throw new ArgumentException("Gap cannot be negative.", nameof(gap));

        _numSplits = numSplits;
        _maxTrainSize = maxTrainSize;
        _testSize = testSize;
        _gap = gap;
    }

    public string Name => "TimeSeriesSplit";
    public int NumSplits => _numSplits;
    public string Description => $"Time series split with {_numSplits} expanding windows{(_gap > 0 ? $" and gap of {_gap}" : "")}.";

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize < _numSplits + 1)
            throw new ArgumentException($"Cannot have {_numSplits} splits with only {dataSize} samples.", nameof(dataSize));

        // Calculate test size for each fold
        int testSizePerFold = _testSize ?? dataSize / (_numSplits + 1);
        if (testSizePerFold < 1) testSizePerFold = 1;

        // Minimum training size for first fold
        int minTrainSize = testSizePerFold;

        for (int i = 0; i < _numSplits; i++)
        {
            // Calculate test indices
            int testStart = minTrainSize + (i * testSizePerFold) + _gap;
            int testEnd = Math.Min(testStart + testSizePerFold, dataSize);

            if (testStart >= dataSize) break;

            // Calculate train indices
            int trainStart = 0;
            int trainEnd = testStart - _gap;

            // Apply max train size constraint if specified
            if (_maxTrainSize.HasValue && trainEnd - trainStart > _maxTrainSize.Value)
            {
                trainStart = trainEnd - _maxTrainSize.Value;
            }

            if (trainEnd <= trainStart) continue;

            // Generate indices
            var trainIndices = new int[trainEnd - trainStart];
            for (int j = 0; j < trainIndices.Length; j++)
                trainIndices[j] = trainStart + j;

            var validationIndices = new int[testEnd - testStart];
            for (int j = 0; j < validationIndices.Length; j++)
                validationIndices[j] = testStart + j;

            yield return (trainIndices, validationIndices);
        }
    }
}
