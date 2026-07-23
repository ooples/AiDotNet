namespace AiDotNet.Memory;

/// <summary>
/// Configuration options for the tensor pool, which manages memory reuse during neural network operations.
/// The tensor pool helps reduce memory allocations and garbage collection pressure by reusing tensor buffers.
/// </summary>
/// <remarks>
/// <para>
/// Tensor pooling is especially beneficial for:
/// - Inference operations with consistent input sizes
/// - Training loops where tensor shapes are predictable
/// - High-throughput scenarios where allocation overhead matters
/// </para>
/// <para>
/// Example usage:
/// <code>
/// var options = new PoolingOptions
/// {
///     MaxPoolSizeMB = 512,        // Allow up to 512 MB of pooled tensors
///     MaxItemsPerBucket = 20,     // Keep up to 20 tensors per shape bucket
///     Enabled = true              // Enable pooling
/// };
/// var pool = new TensorPool&lt;float&gt;(options);
/// </code>
/// </para>
/// </remarks>
public class PoolingOptions
{
    /// <summary>
    /// Gets or sets the maximum memory size of the tensor pool in bytes.
    /// When this limit is reached, new tensors will not be pooled and will be garbage collected normally.
    /// </summary>
    /// <value>
    /// The maximum pool size in bytes. Default is 256 MB (268,435,456 bytes).
    /// </value>
    /// <remarks>
    /// Consider your application's memory constraints when setting this value.
    /// For memory-constrained environments, use a smaller value.
    /// For high-performance inference, consider increasing this value.
    /// </remarks>
    public long MaxPoolSizeBytes { get; set; } = 256L * 1024 * 1024;

    /// <summary>
    /// Gets or sets the maximum memory size of the tensor pool in megabytes.
    /// This is a convenience property that converts to/from <see cref="MaxPoolSizeBytes"/>.
    /// </summary>
    /// <value>
    /// The maximum pool size in MB. Default is 256 MB.
    /// </value>
    /// <example>
    /// <code>
    /// // Set pool size to 512 MB
    /// options.MaxPoolSizeMB = 512;
    /// </code>
    /// </example>
    public int MaxPoolSizeMB
    {
        get => (int)(MaxPoolSizeBytes / (1024 * 1024));
        set => MaxPoolSizeBytes = value * 1024L * 1024;
    }

    /// <summary>
    /// Gets or sets the maximum number of elements a single tensor can have to be eligible for pooling.
    /// Tensors larger than this will not be pooled and will be allocated/deallocated normally.
    /// </summary>
    /// <value>
    /// The maximum element count for poolable tensors. Default is 10,000,000 elements.
    /// </value>
    /// <remarks>
    /// Very large tensors are typically not good candidates for pooling because:
    /// - They consume significant memory in the pool
    /// - They are less likely to be reused (specific shapes)
    /// - Allocation overhead is proportionally smaller for large buffers
    /// </remarks>
    public int MaxElementsToPool { get; set; } = 10_000_000;

    /// <summary>
    /// Gets or sets the maximum number of tensors to keep in each shape bucket.
    /// Tensors are grouped by shape, and this limits how many of each shape are retained.
    /// </summary>
    /// <value>
    /// The maximum tensors per bucket. Default is 10.
    /// </value>
    /// <remarks>
    /// <para>
    /// Higher values allow more tensor reuse but consume more memory.
    /// Lower values reduce memory usage but may increase allocations.
    /// </para>
    /// <para>
    /// For batch processing with consistent batch sizes, a value of 5-20 is typically optimal.
    /// For varied batch sizes or dynamic shapes, consider a higher value like 20-50.
    /// </para>
    /// </remarks>
    public int MaxItemsPerBucket { get; set; } = 10;

    /// <summary>
    /// Gets or sets a value indicating whether tensor pooling is enabled.
    /// When disabled, all tensor operations will allocate new memory instead of reusing pooled buffers.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable pooling (default); <c>false</c> to disable pooling.
    /// </value>
    /// <remarks>
    /// Disabling pooling can be useful for:
    /// - Debugging memory issues
    /// - Profiling actual memory usage patterns
    /// - Scenarios where tensor shapes are highly unpredictable
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to use weak references for pooled tensors.
    /// When enabled, pooled tensors can be garbage collected under memory pressure.
    /// </summary>
    /// <value>
    /// <c>true</c> to use weak references; <c>false</c> to use strong references (default).
    /// </value>
    /// <remarks>
    /// <para>
    /// Weak references allow the garbage collector to reclaim pooled tensors if memory is low,
    /// which can help prevent OutOfMemoryException in memory-constrained scenarios.
    /// </para>
    /// <para>
    /// However, weak references may reduce pooling effectiveness since tensors can be
    /// unexpectedly collected. For best performance with sufficient memory, keep this disabled.
    /// </para>
    /// </remarks>
    public bool UseWeakReferences { get; set; } = false;
}
