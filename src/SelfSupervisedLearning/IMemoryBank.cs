namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Defines the contract for memory banks used in contrastive learning methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A memory bank is a queue that stores embeddings from previous
/// batches. This provides a large pool of negative samples without needing huge batch sizes.</para>
///
/// <para><b>Why use a memory bank?</b></para>
/// <list type="bullet">
/// <item>Provides many negative samples (e.g., 65536) without large batch sizes</item>
/// <item>More memory-efficient than SimCLR's in-batch negatives approach</item>
/// <item>Consistent negative distribution across training</item>
/// </list>
///
/// <para><b>Used by:</b> MoCo, MoCo v2 (not MoCo v3, SimCLR, BYOL, or SimSiam)</para>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// // Create memory bank with 65536 entries
/// var memoryBank = new MemoryBank&lt;float&gt;(65536, embeddingDim: 128);
///
/// // During training:
/// var negatives = memoryBank.GetAll();  // Get all negatives
/// memoryBank.Enqueue(currentEmbeddings);  // Add new embeddings
/// </code>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("MemoryBank")]
public interface IMemoryBank<T>
{
    /// <summary>
    /// Gets the maximum capacity of the memory bank (queue size).
    /// </summary>
    /// <remarks>
    /// <para>Typical values: 4096-65536. MoCo uses 65536 by default.</para>
    /// </remarks>
    int Capacity { get; }

    /// <summary>
    /// Gets the current number of stored embeddings.
    /// </summary>
    int CurrentSize { get; }

    /// <summary>
    /// Gets the embedding dimension of stored vectors.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets whether the memory bank is full (has reached capacity).
    /// </summary>
    bool IsFull { get; }

    /// <summary>
    /// Adds new embeddings to the memory bank (FIFO queue).
    /// </summary>
    /// <param name="embeddings">The embeddings to add (batch of vectors).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> New embeddings are added to the end of the queue.
    /// When the queue is full, the oldest embeddings are removed (first-in, first-out).</para>
    /// </remarks>
    void Enqueue(Tensor<T> embeddings);

    /// <summary>
    /// Gets all stored embeddings for use as negative samples.
    /// </summary>
    /// <returns>A tensor containing all stored embeddings [CurrentSize, EmbeddingDimension].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> These embeddings serve as negative samples in contrastive loss.
    /// The more negatives you have, the harder and more informative the contrastive task becomes.</para>
    /// </remarks>
    Tensor<T> GetAll();

    /// <summary>
    /// Gets a random subset of stored embeddings.
    /// </summary>
    /// <param name="count">The number of embeddings to retrieve.</param>
    /// <returns>A tensor of randomly sampled embeddings [count, EmbeddingDimension].</returns>
    /// <remarks>
    /// <para>Useful when you want fewer negatives than the full memory bank.</para>
    /// </remarks>
    Tensor<T> Sample(int count);

    /// <summary>
    /// Clears all stored embeddings and resets the memory bank.
    /// </summary>
    void Clear();

    /// <summary>
    /// Updates embeddings by averaging with new values (for soft updates).
    /// </summary>
    /// <param name="indices">Indices of embeddings to update.</param>
    /// <param name="newEmbeddings">New embedding values.</param>
    /// <param name="momentum">Momentum for exponential moving average (0-1).</param>
    /// <remarks>
    /// <para>Some memory bank variants use soft updates: new = momentum * old + (1 - momentum) * new</para>
    /// </remarks>
    void UpdateWithMomentum(int[] indices, Tensor<T> newEmbeddings, double momentum);
}
