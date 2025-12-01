namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Handle to GPU memory allocation.
/// </summary>
/// <remarks>
/// <para>
/// Represents a block of memory allocated on the GPU. The handle tracks the
/// allocation size and state, allowing for proper resource management.
/// </para>
/// </remarks>
public interface IGPUMemoryHandle : IDisposable
{
    /// <summary>
    /// Gets the size of the allocation in bytes.
    /// </summary>
    long SizeBytes { get; }

    /// <summary>
    /// Gets whether the memory is still allocated.
    /// </summary>
    bool IsAllocated { get; }
}
