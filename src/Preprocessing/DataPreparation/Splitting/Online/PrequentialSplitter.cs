using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Online;

/// <summary>
/// Prequential (predictive sequential) evaluation splitter for online/streaming data.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Prequential evaluation is the standard way to evaluate models
/// on streaming data. For each new sample: first use it as a test sample (predict its label),
/// then add it to the training set. This simulates real-world continuous learning.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Start with an initial training window
/// 2. For each subsequent sample:
///    - Test: Predict using current model
///    - Train: Update model with the true label
/// 3. Evaluation is "test-then-train" on every sample
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Streaming/online learning scenarios
/// - Concept drift detection
/// - Real-time prediction systems
/// - Evaluating adaptive algorithms
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PrequentialSplitter<T> : DataSplitterBase<T>
{
    private readonly int _initialTrainSize;
    private readonly int _batchSize;
    private readonly bool _forgetOldSamples;
    private readonly int _maxTrainSize;

    /// <summary>
    /// Creates a new prequential evaluation splitter.
    /// </summary>
    /// <param name="initialTrainSize">Size of initial training set. Default is 100.</param>
    /// <param name="batchSize">Number of samples per evaluation batch. Default is 1 (true online).</param>
    /// <param name="forgetOldSamples">Whether to remove old samples (sliding window). Default is false.</param>
    /// <param name="maxTrainSize">Maximum training size when forgetting. Default is 1000.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public PrequentialSplitter(
        int initialTrainSize = 100,
        int batchSize = 1,
        bool forgetOldSamples = false,
        int maxTrainSize = 1000,
        int randomSeed = 42)
        : base(shuffle: false, randomSeed) // No shuffle for streaming data
    {
        if (initialTrainSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(initialTrainSize),
                "Initial training size must be at least 1.");
        }

        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                "Batch size must be at least 1.");
        }

        _initialTrainSize = initialTrainSize;
        _batchSize = batchSize;
        _forgetOldSamples = forgetOldSamples;
        _maxTrainSize = maxTrainSize;
    }

    /// <inheritdoc/>
    public override string Description => $"Prequential (init={_initialTrainSize}, batch={_batchSize}{(_forgetOldSamples ? ", sliding" : "")})";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        // Return first evaluation split
        ValidateInputs(X, y);
        var splits = GetSplits(X, y).ToList();
        return splits.Count > 0 ? splits[0] : throw new InvalidOperationException("No splits generated.");
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;

        if (nSamples <= _initialTrainSize)
        {
            throw new ArgumentException(
                $"Need more than {_initialTrainSize} samples for prequential evaluation.");
        }

        // Sequential indices (no shuffling for streaming)
        var indices = GetIndices(nSamples);
        int foldIndex = 0;

        // Process in batches after initial training
        for (int testStart = _initialTrainSize; testStart < nSamples; testStart += _batchSize)
        {
            int testEnd = Math.Min(testStart + _batchSize, nSamples);

            // Training set: all samples before this batch
            int trainStart = _forgetOldSamples
                ? Math.Max(0, testStart - _maxTrainSize)
                : 0;

            var trainIndices = indices.Skip(trainStart).Take(testStart - trainStart).ToArray();
            var testIndices = indices.Skip(testStart).Take(testEnd - testStart).ToArray();

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: foldIndex++, totalFolds: (nSamples - _initialTrainSize + _batchSize - 1) / _batchSize);
        }
    }

    /// <inheritdoc/>
    public override int NumSplits
    {
        get
        {
            // This is a rough estimate; actual number depends on data size
            return 1; // Returns at least 1 split
        }
    }
}
