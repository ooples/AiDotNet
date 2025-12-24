using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// A sampler that randomly shuffles the dataset indices each epoch.
/// </summary>
/// <remarks>
/// <para>
/// RandomSampler is the default sampling strategy for most training scenarios.
/// It shuffles the dataset indices using the Fisher-Yates algorithm for O(n) time complexity.
/// </para>
/// <para><b>For Beginners:</b> This sampler shuffles your data randomly before each training epoch.
/// Shuffling is important because:
/// - It prevents the model from learning patterns based on data order
/// - It helps the model generalize better
/// - It ensures different batches each epoch for varied gradient updates
///
/// Example:
/// <code>
/// var sampler = new RandomSampler(datasetSize: 1000, seed: 42);
/// foreach (var index in sampler.GetIndices())
/// {
///     // Process sample at index
/// }
/// </code>
/// </para>
/// </remarks>
public class RandomSampler : IDataSampler
{
    private readonly int _datasetSize;
    private Random _random;

    /// <summary>
    /// Initializes a new instance of the RandomSampler class.
    /// </summary>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when datasetSize is less than 1.</exception>
    public RandomSampler(int datasetSize, int? seed = null)
    {
        if (datasetSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(datasetSize), "Dataset size must be at least 1.");
        }

        _datasetSize = datasetSize;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public int Length => _datasetSize;

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        // Create and shuffle indices using Fisher-Yates algorithm
        int[] indices = new int[_datasetSize];
        for (int i = 0; i < _datasetSize; i++)
        {
            indices[i] = i;
        }

        // Fisher-Yates shuffle
        for (int i = _datasetSize - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Yield shuffled indices
        for (int i = 0; i < _datasetSize; i++)
        {
            yield return indices[i];
        }
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _random = RandomHelper.CreateSeededRandom(seed);
    }
}

/// <summary>
/// A sampler that returns indices in sequential order without shuffling.
/// </summary>
/// <remarks>
/// <para>
/// SequentialSampler is useful during evaluation/inference when you want
/// deterministic, reproducible results without any randomness.
/// </para>
/// <para><b>For Beginners:</b> This sampler goes through your data in order (0, 1, 2, 3, ...).
/// Use this when:
/// - Evaluating your model (you want consistent results)
/// - Making predictions on new data
/// - Debugging to isolate issues from randomness
/// </para>
/// </remarks>
public class SequentialSampler : IDataSampler
{
    private readonly int _datasetSize;

    /// <summary>
    /// Initializes a new instance of the SequentialSampler class.
    /// </summary>
    /// <param name="datasetSize">The total number of samples in the dataset.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when datasetSize is less than 1.</exception>
    public SequentialSampler(int datasetSize)
    {
        if (datasetSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(datasetSize), "Dataset size must be at least 1.");
        }

        _datasetSize = datasetSize;
    }

    /// <inheritdoc/>
    public int Length => _datasetSize;

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        for (int i = 0; i < _datasetSize; i++)
        {
            yield return i;
        }
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        // No-op for sequential sampler
    }
}

/// <summary>
/// A sampler that returns a subset of indices.
/// </summary>
/// <remarks>
/// <para>
/// SubsetSampler is useful when you want to train on a specific subset of your data,
/// such as a validation split or a filtered dataset.
/// </para>
/// </remarks>
public class SubsetSampler : IDataSampler
{
    private readonly int[] _indices;
    private Random _random;
    private bool _shuffle;

    /// <summary>
    /// Initializes a new instance of the SubsetSampler class.
    /// </summary>
    /// <param name="indices">The subset of indices to sample from.</param>
    /// <param name="shuffle">Whether to shuffle the indices each epoch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SubsetSampler(IEnumerable<int> indices, bool shuffle = false, int? seed = null)
    {
        _indices = indices?.ToArray() ?? throw new ArgumentNullException(nameof(indices));
        _shuffle = shuffle;
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public int Length => _indices.Length;

    /// <summary>
    /// Gets or sets whether to shuffle the indices each epoch.
    /// </summary>
    public bool Shuffle
    {
        get => _shuffle;
        set => _shuffle = value;
    }

    /// <inheritdoc/>
    public IEnumerable<int> GetIndices()
    {
        if (_shuffle)
        {
            int[] shuffled = new int[_indices.Length];
            Array.Copy(_indices, shuffled, _indices.Length);

            // Fisher-Yates shuffle
            for (int i = shuffled.Length - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
            }

            for (int i = 0; i < shuffled.Length; i++)
            {
                yield return shuffled[i];
            }
        }
        else
        {
            for (int i = 0; i < _indices.Length; i++)
            {
                yield return _indices[i];
            }
        }
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        _random = RandomHelper.CreateSeededRandom(seed);
    }
}
