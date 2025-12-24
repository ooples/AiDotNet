using AiDotNet.Data.Sampling;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Optimizers;

/// <summary>
/// Provides batch iteration utilities for optimization input data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
/// <remarks>
/// <para>
/// OptimizationDataBatcher provides efficient batch iteration over optimization input data,
/// integrating with the DataLoader batching infrastructure for consistent behavior.
/// </para>
/// <para><b>For Beginners:</b> When training machine learning models, you typically don't
/// feed all your data at once. Instead, you break it into smaller "batches" and train
/// on each batch. This class makes that process easy and efficient:
///
/// <code>
/// var batcher = new OptimizationDataBatcher&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(
///     inputData, batchSize: 32, shuffle: true);
///
/// foreach (var (xBatch, yBatch, indices) in batcher.GetBatches())
/// {
///     // Train on this batch
///     var gradient = CalculateGradient(xBatch, yBatch);
///     UpdateParameters(gradient);
/// }
/// </code>
/// </para>
/// </remarks>
public class OptimizationDataBatcher<T, TInput, TOutput>
{
    private readonly OptimizationInputData<T, TInput, TOutput> _inputData;
    private readonly int _batchSize;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly int? _seed;
    private readonly IDataSampler? _sampler;
    private readonly int _dataSize;

    /// <summary>
    /// Initializes a new instance of the OptimizationDataBatcher class.
    /// </summary>
    /// <param name="inputData">The optimization input data containing training, validation, and test sets.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="shuffle">Whether to shuffle the data before batching. Default is true.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch. Default is false.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <param name="sampler">Optional custom sampler for advanced sampling strategies.</param>
    public OptimizationDataBatcher(
        OptimizationInputData<T, TInput, TOutput> inputData,
        int batchSize,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        IDataSampler? sampler = null)
    {
        _inputData = inputData ?? throw new ArgumentNullException(nameof(inputData));
        _batchSize = batchSize > 0 ? batchSize : throw new ArgumentOutOfRangeException(nameof(batchSize));
        _shuffle = shuffle;
        _dropLast = dropLast;
        _seed = seed;
        _sampler = sampler;
        _dataSize = InputHelper<T, TInput>.GetBatchSize(inputData.XTrain);
    }

    /// <summary>
    /// Gets the total number of samples in the training data.
    /// </summary>
    public int DataSize => _dataSize;

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize => _batchSize;

    /// <summary>
    /// Gets the number of batches per epoch.
    /// </summary>
    public int NumBatches
    {
        get
        {
            int batches = _dataSize / _batchSize;
            if (!_dropLast && _dataSize % _batchSize > 0)
            {
                batches++;
            }
            return batches;
        }
    }

    /// <summary>
    /// Iterates through training data in batches.
    /// </summary>
    /// <returns>An enumerable of tuples containing batch inputs, outputs, and indices.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each iteration gives you:
    /// - xBatch: The input features for this batch
    /// - yBatch: The corresponding target values
    /// - indices: The original indices of samples in this batch (useful for tracking)
    /// </para>
    /// </remarks>
    public IEnumerable<(TInput XBatch, TOutput YBatch, int[] Indices)> GetBatches()
    {
        // Get indices using sampler or default shuffling
        int[] indices = GetIndices();

        // Iterate through batches
        int numBatches = _dropLast ? _dataSize / _batchSize : (_dataSize + _batchSize - 1) / _batchSize;

        for (int b = 0; b < numBatches; b++)
        {
            int startIdx = b * _batchSize;
            int endIdx = Math.Min(startIdx + _batchSize, indices.Length);

            // Skip if we're dropping last and this would be incomplete
            if (_dropLast && endIdx - startIdx < _batchSize)
            {
                continue;
            }

            // Extract batch indices
            int[] batchIndices = new int[endIdx - startIdx];
            Array.Copy(indices, startIdx, batchIndices, 0, batchIndices.Length);

            // Extract batch data
            var (xBatch, yBatch) = ExtractBatch(batchIndices);

            yield return (xBatch, yBatch, batchIndices);
        }
    }

    /// <summary>
    /// Iterates through training data in batches, returning only the indices.
    /// </summary>
    /// <returns>An enumerable of index arrays for each batch.</returns>
    /// <remarks>
    /// <para>
    /// This is useful when you need to calculate gradients or perform operations
    /// that only require the indices, not the actual data.
    /// </para>
    /// </remarks>
    public IEnumerable<int[]> GetBatchIndices()
    {
        int[] indices = GetIndices();
        int numBatches = _dropLast ? _dataSize / _batchSize : (_dataSize + _batchSize - 1) / _batchSize;

        for (int b = 0; b < numBatches; b++)
        {
            int startIdx = b * _batchSize;
            int endIdx = Math.Min(startIdx + _batchSize, indices.Length);

            if (_dropLast && endIdx - startIdx < _batchSize)
            {
                continue;
            }

            int[] batchIndices = new int[endIdx - startIdx];
            Array.Copy(indices, startIdx, batchIndices, 0, batchIndices.Length);

            yield return batchIndices;
        }
    }

    /// <summary>
    /// Gets all indices for iteration, optionally shuffled.
    /// </summary>
    private int[] GetIndices()
    {
        int[] indices;

        if (_sampler != null)
        {
            // Use custom sampler
            indices = _sampler.GetIndices().Take(_dataSize).ToArray();
        }
        else
        {
            // Create sequential indices
            indices = new int[_dataSize];
            for (int i = 0; i < _dataSize; i++)
            {
                indices[i] = i;
            }

            // Shuffle if requested
            if (_shuffle)
            {
                Random random = _seed.HasValue
                    ? RandomHelper.CreateSeededRandom(_seed.Value)
                    : RandomHelper.CreateSecureRandom();

                // Fisher-Yates shuffle
                for (int i = indices.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (indices[i], indices[j]) = (indices[j], indices[i]);
                }
            }
        }

        return indices;
    }

    /// <summary>
    /// Extracts a batch of data given the indices.
    /// </summary>
    private (TInput XBatch, TOutput YBatch) ExtractBatch(int[] batchIndices)
    {
        var xBatch = SelectRows(_inputData.XTrain, batchIndices);
        var yBatch = SelectRows(_inputData.YTrain, batchIndices);
        return (xBatch, yBatch);
    }

    /// <summary>
    /// Selects rows from an input structure at the specified indices.
    /// </summary>
    private static TData SelectRows<TData>(TData data, int[] indices)
    {
        if (data is Matrix<T> matrix)
        {
            var result = new Matrix<T>(indices.Length, matrix.Columns);
            for (int i = 0; i < indices.Length; i++)
            {
                result.SetRow(i, matrix.GetRow(indices[i]));
            }
            return (TData)(object)result;
        }

        if (data is Vector<T> vector)
        {
            var result = new Vector<T>(indices.Length);
            for (int i = 0; i < indices.Length; i++)
            {
                result[i] = vector[indices[i]];
            }
            return (TData)(object)result;
        }

        if (data is Tensor<T> tensor)
        {
            // Clone shape but change first dimension
            var newShape = (int[])tensor.Shape.Clone();
            newShape[0] = indices.Length;
            var result = new Tensor<T>(newShape);

            for (int i = 0; i < indices.Length; i++)
            {
                TensorCopyHelper.CopySample(tensor, result, indices[i], i);
            }
            return (TData)(object)result;
        }

        throw new NotSupportedException($"Unsupported data type: {typeof(TData).Name}");
    }

    /// <summary>
    /// Creates a new batcher with a different sampler.
    /// </summary>
    /// <param name="sampler">The new sampler to use.</param>
    /// <returns>A new OptimizationDataBatcher with the specified sampler.</returns>
    public OptimizationDataBatcher<T, TInput, TOutput> WithSampler(IDataSampler sampler)
    {
        return new OptimizationDataBatcher<T, TInput, TOutput>(
            _inputData, _batchSize, _shuffle, _dropLast, _seed, sampler);
    }

    /// <summary>
    /// Creates a new batcher with weighted sampling for class balancing.
    /// </summary>
    /// <typeparam name="TWeight">The numeric type for weights.</typeparam>
    /// <param name="labels">The class labels for each sample.</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <returns>A new OptimizationDataBatcher with weighted sampling.</returns>
    public OptimizationDataBatcher<T, TInput, TOutput> WithClassBalancing<TWeight>(
        IReadOnlyList<int> labels,
        int numClasses)
    {
        var weights = WeightedSampler<double>.CreateBalancedWeights(labels, numClasses);
        var sampler = new WeightedSampler<double>(weights, _dataSize, replacement: false, _seed);
        return WithSampler(sampler);
    }

    /// <summary>
    /// Creates a new batcher with curriculum learning.
    /// </summary>
    /// <typeparam name="TDifficulty">The numeric type for difficulty scores.</typeparam>
    /// <param name="difficulties">Difficulty score for each sample (0 = easiest, 1 = hardest).</param>
    /// <param name="totalEpochs">Total number of epochs for curriculum completion.</param>
    /// <param name="strategy">The curriculum progression strategy.</param>
    /// <returns>A new OptimizationDataBatcher with curriculum learning.</returns>
    public OptimizationDataBatcher<T, TInput, TOutput> WithCurriculumLearning<TDifficulty>(
        IEnumerable<TDifficulty> difficulties,
        int totalEpochs,
        CurriculumStrategy strategy = CurriculumStrategy.Linear)
    {
        var sampler = new CurriculumSampler<TDifficulty>(difficulties, totalEpochs, strategy, _seed);
        return WithSampler(sampler);
    }
}

/// <summary>
/// Extension methods for optimization data batching.
/// </summary>
public static class OptimizationDataBatcherExtensions
{
    /// <summary>
    /// Creates a batcher for the optimization input data.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <typeparam name="TInput">The input type.</typeparam>
    /// <typeparam name="TOutput">The output type.</typeparam>
    /// <param name="inputData">The optimization input data.</param>
    /// <param name="batchSize">Number of samples per batch.</param>
    /// <param name="shuffle">Whether to shuffle. Default is true.</param>
    /// <param name="dropLast">Whether to drop last incomplete batch. Default is false.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>An OptimizationDataBatcher instance.</returns>
    public static OptimizationDataBatcher<T, TInput, TOutput> CreateBatcher<T, TInput, TOutput>(
        this OptimizationInputData<T, TInput, TOutput> inputData,
        int batchSize,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        return new OptimizationDataBatcher<T, TInput, TOutput>(
            inputData, batchSize, shuffle, dropLast, seed);
    }
}
