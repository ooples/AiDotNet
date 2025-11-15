namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of GPU accelerator to use.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different types of hardware for GPU acceleration.
///
/// - CUDA: NVIDIA graphics cards (fastest, most common)
/// - OpenCL: Works on NVIDIA, AMD, Intel (more compatible)
/// - CPU: Uses CPU as fallback (no GPU needed, slower)
/// - Default: Automatically picks the best available option
/// </para>
/// </remarks>
public enum GpuDeviceType
{
    /// <summary>
    /// Automatically select the best available GPU accelerator.
    /// </summary>
    Default,

    /// <summary>
    /// Use CUDA (NVIDIA GPUs only).
    /// </summary>
    CUDA,

    /// <summary>
    /// Use OpenCL (works on NVIDIA, AMD, Intel).
    /// </summary>
    OpenCL,

    /// <summary>
    /// Use CPU as fallback accelerator.
    /// </summary>
    CPU
}
