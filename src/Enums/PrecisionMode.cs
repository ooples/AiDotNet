namespace AiDotNet.Enums;

/// <summary>
/// Defines the numeric precision mode for neural network training and computation.
/// </summary>
public enum PrecisionMode
{
    /// <summary>
    /// Full precision using 32-bit floating-point (float/FP32).
    /// Default mode for standard training.
    /// </summary>
    FP32,

    /// <summary>
    /// Half precision using 16-bit floating-point (Half/FP16).
    /// Faster on modern GPUs with Tensor Cores but limited range [6e-8, 65504].
    /// </summary>
    FP16,

    /// <summary>
    /// Mixed precision training: FP16 for forward/backward passes, FP32 for parameter updates.
    /// Combines speed of FP16 with numerical stability of FP32.
    /// Recommended for large models on GPU.
    /// </summary>
    Mixed,

    /// <summary>
    /// Brain float 16 (bfloat16) format.
    /// Same range as FP32 but reduced precision (8 bits mantissa).
    /// Better numerical stability than FP16, used by Google TPUs.
    /// </summary>
    BF16,

    /// <summary>
    /// Double precision using 64-bit floating-point (double/FP64).
    /// Maximum numerical precision, but slower and uses more memory.
    /// </summary>
    FP64
}
