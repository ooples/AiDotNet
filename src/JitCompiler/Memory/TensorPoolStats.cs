namespace AiDotNet.JitCompiler.Memory;

/// <summary>
/// Statistics about the tensor pool state.
/// </summary>
public class TensorPoolStats
{
    /// <summary>
    /// Total number of buffers currently in the pool.
    /// </summary>
    public int TotalPooledBuffers { get; set; }

    /// <summary>
    /// Estimated memory usage of pooled buffers in bytes.
    /// </summary>
    public long EstimatedMemoryBytes { get; set; }

    /// <summary>
    /// Number of unique tensor shapes being pooled.
    /// </summary>
    public int UniqueShapes { get; set; }
}
