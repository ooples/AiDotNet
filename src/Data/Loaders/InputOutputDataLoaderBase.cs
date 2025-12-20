using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Abstract base class for input-output data loaders providing common supervised learning functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// InputOutputDataLoaderBase provides shared implementation for all supervised learning data loaders including:
/// - Feature (X) and label (Y) data management
/// - Train/validation/test splitting
/// - Shuffling and batching capabilities
/// - Progress tracking through ICountable
/// </para>
/// <para><b>For Beginners:</b> This base class handles common input-output data operations:
/// - Storing features (X) and labels (Y) for supervised learning
/// - Splitting data into training, validation, and test sets
/// - Shuffling data to improve training
/// - Iterating through data in batches
///
/// Concrete implementations (CsvDataLoader, ImageDataLoader) extend this
/// to load specific data formats.
/// </para>
/// </remarks>
public abstract class InputOutputDataLoaderBase<T, TInput, TOutput> :
    DataLoaderBase<T>,
    IInputOutputDataLoader<T, TInput, TOutput>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Storage for loaded feature data.
    /// </summary>
    protected TInput? LoadedFeatures;

    /// <summary>
    /// Storage for loaded label data.
    /// </summary>
    protected TOutput? LoadedLabels;

    /// <summary>
    /// Indices for current data ordering (used for shuffling).
    /// </summary>
    protected int[]? Indices;

    private int _batchSize = 32;
    private int _currentBatchStartIndex;
    private bool _isShuffled;

    /// <inheritdoc/>
    public TInput Features
    {
        get
        {
            EnsureLoaded();
            return LoadedFeatures!;
        }
    }

    /// <inheritdoc/>
    public TOutput Labels
    {
        get
        {
            EnsureLoaded();
            return LoadedLabels!;
        }
    }

    /// <inheritdoc/>
    public abstract int FeatureCount { get; }

    /// <inheritdoc/>
    public abstract int OutputDimension { get; }

    /// <inheritdoc/>
    public override int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    /// <inheritdoc/>
    public bool HasNext => _currentBatchStartIndex < TotalCount;

    /// <inheritdoc/>
    public bool IsShuffled => _isShuffled;

    /// <inheritdoc/>
    public (TInput Features, TOutput Labels) GetNextBatch()
    {
        EnsureLoaded();

        if (!HasNext)
        {
            throw new InvalidOperationException("No more batches available. Call Reset() to start over.");
        }

        int batchStart = _currentBatchStartIndex;
        int batchEnd = Math.Min(batchStart + BatchSize, TotalCount);
        int actualBatchSize = batchEnd - batchStart;

        // Get indices for this batch (respecting shuffle order)
        var batchIndices = new int[actualBatchSize];
        for (int i = 0; i < actualBatchSize; i++)
        {
            batchIndices[i] = Indices![batchStart + i];
        }

        // Extract features and labels for this batch
        var (features, labels) = ExtractBatch(batchIndices);

        _currentBatchStartIndex = batchEnd;
        AdvanceIndex(actualBatchSize);
        AdvanceBatchIndex();

        return (features, labels);
    }

    /// <inheritdoc/>
    public bool TryGetNextBatch(out (TInput Features, TOutput Labels) batch)
    {
        if (!HasNext)
        {
            batch = default;
            return false;
        }

        batch = GetNextBatch();
        return true;
    }

    /// <inheritdoc/>
    public virtual void Shuffle(int? seed = null)
    {
        EnsureLoaded();

        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Fisher-Yates shuffle
        for (int i = Indices!.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (Indices[i], Indices[j]) = (Indices[j], Indices[i]);
        }

        _isShuffled = true;
    }

    /// <inheritdoc/>
    public virtual void Unshuffle()
    {
        EnsureLoaded();

        // Restore original order
        for (int i = 0; i < Indices!.Length; i++)
        {
            Indices[i] = i;
        }

        _isShuffled = false;
    }

    /// <inheritdoc/>
    public abstract (IInputOutputDataLoader<T, TInput, TOutput> Train,
        IInputOutputDataLoader<T, TInput, TOutput> Validation,
        IInputOutputDataLoader<T, TInput, TOutput> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null);

    /// <inheritdoc/>
    protected override void OnReset()
    {
        _currentBatchStartIndex = 0;
    }

    /// <summary>
    /// Initializes indices array after data is loaded.
    /// </summary>
    /// <param name="count">The number of samples in the dataset.</param>
    protected void InitializeIndices(int count)
    {
        Indices = new int[count];
        for (int i = 0; i < count; i++)
        {
            Indices[i] = i;
        }
    }

    /// <summary>
    /// Extracts a batch of features and labels at the specified indices.
    /// </summary>
    /// <param name="indices">The indices of samples to extract.</param>
    /// <returns>A tuple containing the features and labels for the batch.</returns>
    /// <remarks>
    /// Derived classes must implement this to extract data based on their specific
    /// TInput and TOutput types.
    /// </remarks>
    protected abstract (TInput Features, TOutput Labels) ExtractBatch(int[] indices);

    /// <summary>
    /// Validates split ratios.
    /// </summary>
    /// <param name="trainRatio">Training ratio.</param>
    /// <param name="validationRatio">Validation ratio.</param>
    /// <exception cref="ArgumentException">Thrown when ratios are invalid.</exception>
    protected static void ValidateSplitRatios(double trainRatio, double validationRatio)
    {
        if (trainRatio <= 0 || trainRatio >= 1)
        {
            throw new ArgumentException("Train ratio must be between 0 and 1 (exclusive).", nameof(trainRatio));
        }

        if (validationRatio < 0 || validationRatio >= 1)
        {
            throw new ArgumentException("Validation ratio must be between 0 and 1 (exclusive).", nameof(validationRatio));
        }

        if (trainRatio + validationRatio >= 1)
        {
            throw new ArgumentException("Train ratio + validation ratio must be less than 1.");
        }
    }

    /// <summary>
    /// Computes split sizes from ratios and total count.
    /// </summary>
    /// <param name="totalCount">Total number of samples.</param>
    /// <param name="trainRatio">Training ratio.</param>
    /// <param name="validationRatio">Validation ratio.</param>
    /// <returns>A tuple containing train, validation, and test sizes.</returns>
    protected static (int TrainSize, int ValidationSize, int TestSize) ComputeSplitSizes(
        int totalCount,
        double trainRatio,
        double validationRatio)
    {
        int trainSize = (int)(totalCount * trainRatio);
        int validationSize = (int)(totalCount * validationRatio);
        int testSize = totalCount - trainSize - validationSize;

        return (trainSize, validationSize, testSize);
    }
}
