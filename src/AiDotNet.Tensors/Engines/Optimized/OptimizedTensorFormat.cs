namespace AiDotNet.Tensors.Engines.Optimized;

/// <summary>
/// Describes the memory format of an optimized tensor.
/// </summary>
/// <remarks>
/// <para>
/// This enum is used to track the current memory layout of tensors that have been
/// optimized for hardware-specific operations. Different formats offer different
/// performance characteristics depending on the operation being performed.
/// </para>
/// <para><b>For Beginners:</b> Think of this like different ways to organize a filing cabinet.
///
/// Some formats are better for certain operations:
/// - Standard format is easy to read and write, but may not be the fastest
/// - GPU-resident format means the data lives on the graphics card for fast parallel processing
/// - Blocked formats (like oneDNN uses) rearrange data for better CPU cache usage
///
/// By tracking the format, we can avoid converting back and forth unnecessarily,
/// which saves time especially when doing multiple operations in sequence.
/// </para>
/// </remarks>
public enum OptimizedTensorFormat
{
    /// <summary>
    /// Standard row-major NCHW format (Batch, Channels, Height, Width).
    /// This is the default format that most operations expect.
    /// </summary>
    Standard = 0,

    /// <summary>
    /// GPU-resident memory (CUDA, HIP, Metal, etc.).
    /// Data is stored on the GPU and operations execute there.
    /// </summary>
    GpuResident = 1,

    /// <summary>
    /// oneDNN blocked format (nChw16c, nChw8c, etc.).
    /// Data is arranged in blocks optimized for SIMD vectorization.
    /// Used by Intel oneDNN for optimized convolution operations.
    /// </summary>
    OneDnnBlocked = 2,

    /// <summary>
    /// oneDNN format chosen by format_tag_any.
    /// The specific format is determined by oneDNN at primitive creation time
    /// to maximize performance for the given operation and hardware.
    /// </summary>
    OneDnnOptimal = 3,

    /// <summary>
    /// BLAS-packed format for matrix operations.
    /// Data is pre-packed for optimized matrix multiplication routines.
    /// </summary>
    BlasPacked = 4
}
