using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Sliding Window cross-validation for time series with fixed-size training window.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Unlike expanding window (TimeSeriesSplit), sliding window:
/// <list type="bullet">
/// <item>Uses a fixed-size training window that "slides" forward</item>
/// <item>Better for non-stationary time series where old data becomes less relevant</item>
/// <item>Each fold trains on the same amount of data</item>
/// <item>Useful when you believe recent history is more predictive than distant past</item>
/// </list>
/// </para>
/// <para><b>Example:</b> With window_size=100 and test_size=20:
/// <list type="bullet">
/// <item>Fold 1: Train [0-99], Test [100-119]</item>
/// <item>Fold 2: Train [20-119], Test [120-139]</item>
/// <item>Fold 3: Train [40-139], Test [140-159]</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SlidingWindowStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _windowSize;
    private readonly int _testSize;
    private readonly int _step;
    private int _numSplits;

    public string Name => $"Sliding Window (size={_windowSize})";
    public string Description => "Time series CV with fixed-size sliding training window.";
    public int NumSplits => _numSplits;

    /// <summary>
    /// Initializes Sliding Window cross-validation.
    /// </summary>
    /// <param name="windowSize">Fixed training window size.</param>
    /// <param name="testSize">Number of samples in each test set.</param>
    /// <param name="step">How much to slide the window each fold. Default: same as testSize.</param>
    public SlidingWindowStrategy(int windowSize, int testSize, int? step = null)
    {
        if (windowSize < 1)
            throw new ArgumentException("Window size must be at least 1.");
        if (testSize < 1)
            throw new ArgumentException("Test size must be at least 1.");

        _windowSize = windowSize;
        _testSize = testSize;
        _step = step ?? testSize;
    }

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize < _windowSize + _testSize)
            throw new ArgumentException($"Dataset size ({dataSize}) must be at least window_size + test_size ({_windowSize + _testSize}).");

        int foldCount = 0;
        int trainStart = 0;

        while (trainStart + _windowSize + _testSize <= dataSize)
        {
            var trainIndices = Enumerable.Range(trainStart, _windowSize).ToArray();
            var testIndices = Enumerable.Range(trainStart + _windowSize, _testSize).ToArray();

            foldCount++;
            yield return (trainIndices, testIndices);

            trainStart += _step;
        }

        _numSplits = foldCount;
    }
}
