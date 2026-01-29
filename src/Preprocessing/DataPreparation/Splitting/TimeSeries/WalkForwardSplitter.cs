using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Walk-forward validation that simulates production deployment over time.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Walk-forward validation simulates how your model would actually
/// be deployed and retrained over time.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// <code>
/// Step 1: Train on Jan-Mar → Predict Apr
/// Step 2: Train on Jan-Apr → Predict May
/// Step 3: Train on Jan-May → Predict Jun
/// ...
/// </code>
/// </para>
/// <para>
/// <b>Why Use Walk-Forward?</b>
/// - Most realistic evaluation for production deployment
/// - Tests how the model adapts as new data arrives
/// - Standard for algorithmic trading and forecasting
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class WalkForwardSplitter<T> : DataSplitterBase<T>
{
    private readonly int _initialTrainSize;
    private readonly int _testSize;
    private readonly int _stepSize;
    private int _nSplits;

    /// <summary>
    /// Creates a new walk-forward validation splitter.
    /// </summary>
    /// <param name="initialTrainSize">Initial training set size.</param>
    /// <param name="testSize">Size of each test period.</param>
    /// <param name="stepSize">How much to advance each iteration (null = testSize).</param>
    public WalkForwardSplitter(int initialTrainSize, int testSize, int? stepSize = null)
        : base(shuffle: false, randomSeed: 42)
    {
        if (initialTrainSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(initialTrainSize), "Initial train size must be at least 1.");
        }

        if (testSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be at least 1.");
        }

        _initialTrainSize = initialTrainSize;
        _testSize = testSize;
        _stepSize = stepSize ?? testSize;
    }

    /// <inheritdoc/>
    public override int NumSplits => _nSplits;

    /// <inheritdoc/>
    public override string Description => $"Walk-Forward (initial={_initialTrainSize}, test={_testSize}, step={_stepSize})";

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
        _nSplits = 0;

        int trainEnd = _initialTrainSize;
        int split = 0;

        while (trainEnd + _testSize <= n)
        {
            var trainIndices = Enumerable.Range(0, trainEnd).ToArray();
            var testIndices = Enumerable.Range(trainEnd, _testSize).ToArray();

            _nSplits++;

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: split++, totalFolds: _nSplits);

            trainEnd += _stepSize;
        }
    }
}
