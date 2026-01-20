namespace AiDotNet.Tensors.Engines.Optimized;

/// <summary>
/// Represents a CPU tensor stored in an optimized memory format (e.g., oneDNN blocked format)
/// for accelerated operations.
/// </summary>
/// <remarks>
/// <para>
/// This interface extends <see cref="IOptimizedTensor{T}"/> for CPU-specific optimizations.
/// The primary use case is for oneDNN blocked formats that enable SIMD-optimized convolution
/// operations. Tensors in blocked format can be passed directly between consecutive
/// convolution layers without reordering back to NCHW format.
/// </para>
/// <para><b>For Beginners:</b> Modern CPUs can process multiple numbers at once using SIMD
/// (Single Instruction, Multiple Data) instructions. To take advantage of this, data needs
/// to be arranged in specific patterns called "blocked formats".
///
/// For example, instead of storing channels as [C0, C1, C2, C3, ...], a blocked format
/// might store them as [C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15]
/// together, making it easy to load all 16 values into a SIMD register at once.
///
/// This interface tracks which tensors are in this optimized format so we can skip
/// the conversion step when doing multiple operations in sequence.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensor elements (e.g., float, double).</typeparam>
public interface ICpuOptimizedTensor<T> : IOptimizedTensor<T>
{
    /// <summary>
    /// Gets the native memory handle pointing to the optimized format data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a pointer to the native memory allocated by oneDNN or another library
    /// for storing the blocked format data. The data layout is library-specific and
    /// should not be accessed directly except through the corresponding library's APIs.
    /// </para>
    /// <para><b>Warning:</b> This handle is for internal use only. Accessing the memory
    /// directly without understanding the blocked format layout will produce incorrect results.
    /// </para>
    /// </remarks>
    IntPtr NativeMemoryHandle { get; }

    /// <summary>
    /// Gets the oneDNN format tag used for this tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value corresponds to oneDNN's format_tag enum (e.g., nChw16c = 73).
    /// A value of 0 (undef) indicates the format is not a recognized oneDNN tag.
    /// A value of 1 (any) indicates the format was chosen by oneDNN at primitive creation time.
    /// </para>
    /// </remarks>
    int OneDnnFormatTag { get; }

    /// <summary>
    /// Gets whether this tensor can be used directly in oneDNN operations without reordering.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returns true if the tensor is in a format compatible with oneDNN's internal operations.
    /// When true, the tensor can be passed directly to oneDNN primitives, avoiding the
    /// overhead of format conversion (reorder operations).
    /// </para>
    /// <para><b>For Beginners:</b> If this is true, you can use this tensor directly in
    /// the next convolution operation without any conversion penalty. This is the key
    /// to achieving the performance benefits of blocked format persistence.
    /// </para>
    /// </remarks>
    bool IsOneDnnCompatible { get; }

    /// <summary>
    /// Gets the oneDNN memory descriptor handle for this tensor's format.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This handle can be used to query the exact memory layout or to create
    /// compatible memory objects in oneDNN. The descriptor remains valid as long
    /// as this tensor exists.
    /// </para>
    /// </remarks>
    IntPtr OneDnnMemoryDescriptor { get; }
}
