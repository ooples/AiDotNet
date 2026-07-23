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
    TF32 = 3,

    /// <summary>
    /// FP8 E4M3 format (4 exponent bits, 3 mantissa bits).
    /// </summary>
    /// <remarks>
    /// <para>
    /// 8-bit floating point with higher precision but smaller dynamic range.
    /// Best for weights and activations where values are well-bounded.
    /// </para>
    /// <para>
    /// <b>Range:</b> ±448, <b>Precision:</b> ~3.3 decimal digits.
    /// </para>
    /// <para>
    /// <b>Hardware:</b> Requires NVIDIA H100/H200 or newer GPU with FP8 Tensor Cores.
    /// Provides up to 2x throughput compared to FP16.
    /// </para>
    /// <para>
    /// <b>Reference:</b> NVIDIA FP8 formats for deep learning (2022).
    /// </para>
    /// </remarks>
    FP8_E4M3 = 4,

    /// <summary>
    /// FP8 E5M2 format (5 exponent bits, 2 mantissa bits).
    /// </summary>
    /// <remarks>
    /// <para>
    /// 8-bit floating point with larger dynamic range but lower precision.
    /// Best for gradients which can have large range but don't need high precision.
    /// </para>
    /// <para>
    /// <b>Range:</b> ±57344, <b>Precision:</b> ~2 decimal digits.
    /// </para>
    /// <para>
    /// <b>Hardware:</b> Requires NVIDIA H100/H200 or newer GPU with FP8 Tensor Cores.
    /// </para>
    /// <para>
    /// <b>Best Practice:</b> Use E4M3 for forward pass (weights/activations),
    /// E5M2 for backward pass (gradients).
    /// </para>
    /// </remarks>
    FP8_E5M2 = 5,

    /// <summary>
    /// Hybrid FP8 mode using E4M3 for forward pass and E5M2 for backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Recommended FP8 configuration that combines the strengths of both formats:
    /// </para>
    /// <list type="bullet">
    /// <item><description><b>Forward pass (E4M3):</b> Higher precision for weights and activations</description></item>
    /// <item><description><b>Backward pass (E5M2):</b> Larger range for gradients</description></item>
    /// </list>
    /// <para>
    /// <b>Hardware:</b> Requires NVIDIA H100/H200 or newer GPU with FP8 Tensor Cores.
    /// </para>
    /// <para>
    /// <b>Research:</b> This hybrid approach is recommended by NVIDIA and achieves
    /// near-BF16 accuracy while providing 2x throughput improvement.
    /// </para>
    /// </remarks>
    FP8_Hybrid = 6
}
