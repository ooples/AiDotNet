namespace AiDotNet.Memory;

/// <summary>
/// Provides statistics about the current state of a tensor pool.
/// Use this class to monitor pool usage and tune pooling parameters.
/// </summary>
/// <remarks>
/// <para>
/// Pool statistics help you understand:
/// - How much memory is currently being used by pooled tensors
/// - How many tensors are available for reuse
/// - Whether the pool is operating efficiently
/// </para>
/// <para>
/// Example usage:
/// <code>
/// var pool = new TensorPool&lt;float&gt;();
/// // ... use the pool for inference operations ...
///
/// var stats = pool.GetStatistics();
/// Console.WriteLine($"Pool using {stats.MemoryUtilizationPercent:F1}% of max capacity");
/// Console.WriteLine($"Tensors in pool: {stats.PooledTensorCount}");
/// </code>
/// </para>
/// </remarks>
public class PoolStatistics
{
    /// <summary>
    /// Gets or sets the number of tensors currently held in the pool.
    /// These tensors are available for immediate reuse without allocation.
    /// </summary>
    /// <value>
    /// The count of pooled tensors across all shape buckets.
    /// </value>
    public int PooledTensorCount { get; set; }

    /// <summary>
    /// Gets or sets the total memory currently used by pooled tensors, in bytes.
    /// </summary>
    /// <value>
    /// The current memory usage in bytes.
    /// </value>
    /// <remarks>
    /// This value represents the memory reserved for pooled tensors.
    /// It does not include tensors that have been rented and are currently in use.
    /// </remarks>
    public long CurrentMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the maximum memory allowed for pooling, in bytes.
    /// This corresponds to <see cref="PoolingOptions.MaxPoolSizeBytes"/>.
    /// </summary>
    /// <value>
    /// The maximum pool memory capacity in bytes.
    /// </value>
    public long MaxMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the number of shape buckets in the pool.
    /// Each unique tensor shape has its own bucket for storing reusable tensors.
    /// </summary>
    /// <value>
    /// The number of distinct shape buckets.
    /// </value>
    /// <remarks>
    /// A high number of buckets may indicate many different tensor shapes in use,
    /// which can reduce pooling effectiveness. Consider standardizing tensor shapes
    /// when possible to improve reuse rates.
    /// </remarks>
    public int TensorBuckets { get; set; }

    /// <summary>
    /// Gets the percentage of maximum pool memory currently in use.
    /// </summary>
    /// <value>
    /// A value from 0 to 100 representing memory utilization percentage.
    /// Returns 0 if <see cref="MaxMemoryBytes"/> is 0.
    /// </value>
    /// <remarks>
    /// <para>
    /// Use this metric to determine if your pool size is appropriate:
    /// - Values consistently near 100% suggest increasing <see cref="PoolingOptions.MaxPoolSizeMB"/>
    /// - Values consistently near 0% suggest decreasing the pool size to save memory
    /// - Values between 40-80% typically indicate good pool sizing
    /// </para>
    /// </remarks>
    public double MemoryUtilizationPercent => MaxMemoryBytes > 0
        ? (CurrentMemoryBytes * 100.0 / MaxMemoryBytes)
        : 0;
}
