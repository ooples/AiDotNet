using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Online;

/// <summary>
/// Landmark window splitter for online learning with growing training set.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A landmark window keeps all historical data from a fixed starting
/// point (the "landmark"). As new data arrives, the training set grows but always starts
/// from the same point. This is useful when concept drift is minimal and all history is relevant.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Define a landmark (start point), typically timestamp 0
/// 2. Training set includes all data from landmark to current time
/// 3. Test on the next batch of samples
/// 4. Training window expands over time
/// </para>
/// <para>
/// <b>Comparison:</b>
/// - Sliding window: Fixed size, moves forward (forgets old data)
/// - Landmark window: Variable size, keeps all history
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - When all historical data remains relevant
/// - Stationary data distributions
/// - Long-term pattern learning
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LandmarkWindowSplitter<T> : DataSplitterBase<T>
{
    private readonly int _landmarkPosition;
    private readonly int _initialTestPosition;
    private readonly int _stepSize;
    private readonly int _testWindowSize;

    /// <summary>
    /// Creates a new landmark window splitter.
    /// </summary>
    /// <param name="landmarkPosition">Starting position of the landmark (training start). Default is 0.</param>
    /// <param name="initialTestPosition">First position to start testing from. Default is 100.</param>
    /// <param name="stepSize">How many samples to advance each iteration. Default is 50.</param>
    /// <param name="testWindowSize">Size of the test window. Default is 50.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LandmarkWindowSplitter(
        int landmarkPosition = 0,
        int initialTestPosition = 100,
        int stepSize = 50,
        int testWindowSize = 50,
        int randomSeed = 42)
        : base(shuffle: false, randomSeed) // No shuffle for streaming data
    {
        if (landmarkPosition < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(landmarkPosition),
                "Landmark position cannot be negative.");
        }

        if (initialTestPosition <= landmarkPosition)
        {
            throw new ArgumentOutOfRangeException(nameof(initialTestPosition),
                "Initial test position must be after landmark position.");
        }

        if (stepSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(stepSize),
                "Step size must be at least 1.");
        }

        if (testWindowSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testWindowSize),
                "Test window size must be at least 1.");
        }

        _landmarkPosition = landmarkPosition;
        _initialTestPosition = initialTestPosition;
        _stepSize = stepSize;
        _testWindowSize = testWindowSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Landmark Window (start={_landmarkPosition}, step={_stepSize}, test={_testWindowSize})";

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

        if (nSamples <= _initialTestPosition)
        {
            throw new ArgumentException(
                $"Need more than {_initialTestPosition} samples for landmark window splitting.");
        }

        var indices = GetIndices(nSamples);
        int foldIndex = 0;

        // Iterate through data with expanding training set
        for (int testStart = _initialTestPosition; testStart < nSamples; testStart += _stepSize)
        {
            int testEnd = Math.Min(testStart + _testWindowSize, nSamples);

            // Training: from landmark to just before test window
            var trainIndices = indices.Skip(_landmarkPosition).Take(testStart - _landmarkPosition).ToArray();
            var testIndices = indices.Skip(testStart).Take(testEnd - testStart).ToArray();

            int totalFolds = (nSamples - _initialTestPosition + _stepSize - 1) / _stepSize;

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: foldIndex++, totalFolds: totalFolds);
        }
    }
}
