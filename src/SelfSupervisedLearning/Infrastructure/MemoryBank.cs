using AiDotNet.Helpers;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;

namespace AiDotNet.SelfSupervisedLearning.Infrastructure;

/// <summary>
/// FIFO memory queue for storing embeddings in contrastive learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A memory bank is a queue that stores embeddings from previous
/// batches. This provides a large pool of negative samples for contrastive learning without
/// requiring huge batch sizes.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item>New embeddings are added to the end of the queue (Enqueue)</item>
/// <item>When the queue is full, oldest embeddings are removed (FIFO)</item>
/// <item>All stored embeddings serve as negative samples for contrastive loss</item>
/// </list>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// // Create memory bank with 65536 entries
/// var memoryBank = new MemoryBank&lt;float&gt;(capacity: 65536, embeddingDim: 128);
///
/// // Training loop:
/// var negatives = memoryBank.GetAll();  // Get negative samples
/// var loss = ComputeContrastiveLoss(queries, keys, negatives);
/// memoryBank.Enqueue(momentumEncoderOutput);  // Add new embeddings
/// </code>
/// </remarks>
public class MemoryBank<T> : IMemoryBank<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly T[] _storage;
    private readonly int _capacity;
    private readonly int _embeddingDim;
    private int _currentSize;
    private int _pointer;

    private readonly Random _random;

    /// <inheritdoc />
    public int Capacity => _capacity;

    /// <inheritdoc />
    public int CurrentSize => _currentSize;

    /// <inheritdoc />
    public int EmbeddingDimension => _embeddingDim;

    /// <inheritdoc />
    public bool IsFull => _currentSize >= _capacity;

    /// <summary>
    /// Initializes a new instance of the MemoryBank class.
    /// </summary>
    /// <param name="capacity">Maximum number of embeddings to store (e.g., 65536).</param>
    /// <param name="embeddingDim">Dimension of each embedding (e.g., 128).</param>
    /// <param name="seed">Optional random seed for sampling.</param>
    public MemoryBank(int capacity, int embeddingDim, int? seed = null)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        _capacity = capacity;
        _embeddingDim = embeddingDim;
        _storage = new T[capacity * embeddingDim];
        _currentSize = 0;
        _pointer = 0;

        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;

        // Initialize storage with zeros
        for (int i = 0; i < _storage.Length; i++)
        {
            _storage[i] = NumOps.Zero;
        }
    }

    /// <inheritdoc />
    public void Enqueue(Tensor<T> embeddings)
    {
        if (embeddings is null) throw new ArgumentNullException(nameof(embeddings));

        var batchSize = embeddings.Shape[0];
        var dim = embeddings.Shape[1];

        if (dim != _embeddingDim)
        {
            throw new ArgumentException(
                $"Embedding dimension mismatch. Expected {_embeddingDim}, got {dim}",
                nameof(embeddings));
        }

        for (int b = 0; b < batchSize; b++)
        {
            // Copy embedding to storage at current pointer position
            int storageOffset = _pointer * _embeddingDim;
            for (int d = 0; d < _embeddingDim; d++)
            {
                _storage[storageOffset + d] = embeddings[b, d];
            }

            // Update pointer (circular buffer)
            _pointer = (_pointer + 1) % _capacity;

            // Update size if not yet full
            if (_currentSize < _capacity)
            {
                _currentSize++;
            }
        }
    }

    /// <inheritdoc />
    public Tensor<T> GetAll()
    {
        if (_currentSize == 0)
        {
            return new Tensor<T>(Array.Empty<T>(), [0, _embeddingDim]);
        }

        var result = new T[_currentSize * _embeddingDim];

        if (IsFull)
        {
            // When full, we need to reorder so oldest is first
            // Start from pointer (oldest) and wrap around
            int destOffset = 0;
            for (int i = 0; i < _capacity; i++)
            {
                int srcIndex = ((_pointer + i) % _capacity) * _embeddingDim;
                for (int d = 0; d < _embeddingDim; d++)
                {
                    result[destOffset++] = _storage[srcIndex + d];
                }
            }
        }
        else
        {
            // Not full yet, just copy from start
            Array.Copy(_storage, result, _currentSize * _embeddingDim);
        }

        return new Tensor<T>(result, [_currentSize, _embeddingDim]);
    }

    /// <inheritdoc />
    public Tensor<T> Sample(int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (_currentSize == 0)
        {
            return new Tensor<T>(Array.Empty<T>(), [0, _embeddingDim]);
        }

        count = Math.Min(count, _currentSize);
        var result = new T[count * _embeddingDim];

        // Generate random indices without replacement
        var indices = Enumerable.Range(0, _currentSize)
            .OrderBy(_ => _random.Next())
            .Take(count)
            .ToArray();

        for (int i = 0; i < count; i++)
        {
            int srcOffset = indices[i] * _embeddingDim;
            int destOffset = i * _embeddingDim;
            for (int d = 0; d < _embeddingDim; d++)
            {
                result[destOffset + d] = _storage[srcOffset + d];
            }
        }

        return new Tensor<T>(result, [count, _embeddingDim]);
    }

    /// <inheritdoc />
    public void Clear()
    {
        _currentSize = 0;
        _pointer = 0;

        for (int i = 0; i < _storage.Length; i++)
        {
            _storage[i] = NumOps.Zero;
        }
    }

    /// <inheritdoc />
    public void UpdateWithMomentum(int[] indices, Tensor<T> newEmbeddings, double momentum)
    {
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (newEmbeddings is null) throw new ArgumentNullException(nameof(newEmbeddings));
        if (momentum < 0 || momentum > 1) throw new ArgumentOutOfRangeException(nameof(momentum));

        if (indices.Length != newEmbeddings.Shape[0])
        {
            throw new ArgumentException("Number of indices must match number of embeddings");
        }

        var m = NumOps.FromDouble(momentum);
        var oneMinusM = NumOps.FromDouble(1.0 - momentum);

        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];
            if (idx < 0 || idx >= _currentSize)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {idx} out of range");
            }

            int storageOffset = idx * _embeddingDim;
            for (int d = 0; d < _embeddingDim; d++)
            {
                // EMA update: new = m * old + (1 - m) * new
                var oldVal = _storage[storageOffset + d];
                var newVal = newEmbeddings[i, d];
                _storage[storageOffset + d] = NumOps.Add(
                    NumOps.Multiply(m, oldVal),
                    NumOps.Multiply(oneMinusM, newVal));
            }
        }
    }

    /// <summary>
    /// Gets the embedding at a specific index.
    /// </summary>
    /// <param name="index">The index of the embedding to retrieve.</param>
    /// <returns>The embedding tensor [1, embeddingDim].</returns>
    public Tensor<T> GetAt(int index)
    {
        if (index < 0 || index >= _currentSize)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }

        var result = new T[_embeddingDim];
        int storageOffset = index * _embeddingDim;

        for (int d = 0; d < _embeddingDim; d++)
        {
            result[d] = _storage[storageOffset + d];
        }

        return new Tensor<T>(result, [1, _embeddingDim]);
    }

    /// <summary>
    /// Sets the embedding at a specific index.
    /// </summary>
    /// <param name="index">The index to set.</param>
    /// <param name="embedding">The embedding tensor [1, embeddingDim] or [embeddingDim].</param>
    public void SetAt(int index, Tensor<T> embedding)
    {
        if (index < 0 || index >= _capacity)
        {
            throw new ArgumentOutOfRangeException(nameof(index));
        }
        if (embedding is null) throw new ArgumentNullException(nameof(embedding));

        int storageOffset = index * _embeddingDim;

        // Handle both [1, dim] and [dim] shapes
        var flatData = embedding.Data;
        if (flatData.Length != _embeddingDim)
        {
            throw new ArgumentException(
                $"Embedding dimension mismatch. Expected {_embeddingDim}, got {flatData.Length}",
                nameof(embedding));
        }

        for (int d = 0; d < _embeddingDim; d++)
        {
            _storage[storageOffset + d] = flatData[d];
        }

        // Update size if setting beyond current size
        if (index >= _currentSize)
        {
            _currentSize = index + 1;
        }
    }
}
