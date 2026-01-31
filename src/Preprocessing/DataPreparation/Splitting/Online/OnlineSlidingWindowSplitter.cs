using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Online;

/// <summary>
/// Online sliding window splitter for streaming data with concept drift adaptation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Unlike the landmark window, a sliding window maintains a fixed-size
/// training set that moves forward in time. Old samples are "forgotten" as new ones arrive.
/// This is essential when data patterns change over time (concept drift).
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Maintain a window of the most recent N samples for training
/// 2. Test on the next batch of samples
/// 3. Slide the window forward, dropping oldest samples
/// </para>
/// <para>
/// <b>Window Strategies:</b>
/// - Fixed: Constant window size throughout
/// - Adaptive: Adjust window size based on performance
/// - Fading: Weight recent samples more heavily (conceptually)
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Non-stationary data (concept drift)
/// - Real-time systems with memory constraints
/// - When recent data is more relevant than old data
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class OnlineSlidingWindowSplitter<T> : DataSplitterBase<T>
{
    private readonly int _windowSize;
    private readonly int _stepSize;
    private readonly int _testSize;
    private readonly int _gapSize;

    /// <summary>
    /// Creates a new online sliding window splitter.
    /// </summary>
    /// <param name="windowSize">Size of the training window. Default is 500.</param>
    /// <param name="stepSize">How many samples to advance each iteration. Default is 100.</param>
    /// <param name="testSize">Number of samples to test on. Default is 100.</param>
    /// <param name="gapSize">Gap between training and test to prevent leakage. Default is 0.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public OnlineSlidingWindowSplitter(
        int windowSize = 500,
        int stepSize = 100,
        int testSize = 100,
        int gapSize = 0,
        int randomSeed = 42)
        : base(shuffle: false, randomSeed) // No shuffle for streaming data
    {
        if (windowSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(windowSize),
                "Window size must be at least 1.");
        }

        if (stepSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stepSize),
                "Step size must be at least 1.");
        }

        if (testSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize),
                "Test size must be at least 1.");
        }

        if (gapSize < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gapSize),
                "Gap size cannot be negative.");
        }

        _windowSize = windowSize;
        _stepSize = stepSize;
        _testSize = testSize;
        _gapSize = gapSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Online Sliding Window (size={_windowSize}, step={_stepSize}, test={_testSize})";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);
        var splits = GetSplits(X, y).ToList();
        return splits.Count > 0 ? splits[0] : throw new InvalidOperationException("No splits generated.");
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int minRequired = _windowSize + _gapSize + _testSize;

        if (nSamples < minRequired)
        {
            throw new ArgumentException(
                $"Need at least {minRequired} samples (window={_windowSize}, gap={_gapSize}, test={_testSize}).");
        }

        var indices = GetIndices(nSamples);
        int foldIndex = 0;

        // Slide the window through the data
        int position = 0;
        while (position + _windowSize + _gapSize + _testSize <= nSamples)
        {
            // Training window
            int trainStart = position;
            int trainEnd = position + _windowSize;

            // Test window (after gap)
            int testStart = trainEnd + _gapSize;
            int testEnd = Math.Min(testStart + _testSize, nSamples);

            var trainIndices = indices.Skip(trainStart).Take(trainEnd - trainStart).ToArray();
            var testIndices = indices.Skip(testStart).Take(testEnd - testStart).ToArray();

            int totalFolds = (nSamples - minRequired + _stepSize) / _stepSize;

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: foldIndex++, totalFolds: Math.Max(1, totalFolds));

            position += _stepSize;
        }
    }
}
