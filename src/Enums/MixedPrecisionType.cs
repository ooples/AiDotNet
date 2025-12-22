namespace AiDotNet.Enums;

/// <summary>
/// Types of mixed precision training data types.
/// </summary>
/// <remarks>
/// <para>
/// Mixed precision training uses lower precision floating point numbers
/// to speed up training and reduce memory usage while maintaining accuracy.
/// </para>
/// <para><b>For Beginners:</b> FP16 works on most GPUs, BF16 is better on newer
/// hardware (Ampere and later). If unsure, start with FP16.</para>
/// </remarks>
public enum MixedPrecisionType
{
    /// <summary>
    /// Full precision (FP32). No mixed precision.
    /// </summary>
    /// <remarks>
    /// Uses 32-bit floating point for all operations.
    /// Maximum precision but highest memory usage and slowest.
    /// </remarks>
    None = 0,

    /// <summary>
    /// Half precision (FP16) mixed precision.
    /// </summary>
    /// <remarks>
    /// Uses 16-bit floating point for forward/backward pass.
    /// Widely supported on GPUs since Pascal (GTX 10 series).
    /// Requires loss scaling to handle small gradients.
    /// </remarks>
    FP16 = 1,

    /// <summary>
    /// Brain floating point (BF16) mixed precision.
    /// </summary>
    /// <remarks>
    /// 16-bit format with same exponent range as FP32.
    /// Better numerical stability than FP16, no loss scaling needed.
    /// Requires Ampere or newer GPU (RTX 30 series, A100, etc.).
    /// </remarks>
    BF16 = 2,

    /// <summary>
    /// TensorFloat-32 (TF32) precision.
    /// </summary>
    /// <remarks>
    /// NVIDIA format that uses 19 bits total (10-bit mantissa).
    /// Automatically enabled on Ampere GPUs for matmul operations.
    /// Good balance of speed and precision.
    /// </remarks>
    TF32 = 3
}
