namespace AiDotNet.Enums;

/// <summary>
/// Specifies the numerical precision tier used for matrix multiplications, convolutions, attention, and activation
/// tensors during compiled or eager model execution.
/// </summary>
public enum ComputePrecisionTier
{
    /// <summary>
    /// Full IEEE-754 32-bit floating-point execution and accumulation.
    /// </summary>
    FP32 = 0,

    /// <summary>
    /// TensorFloat-32 Tensor Core math (19-bit mantissa/exponent input rounding with FP32 accumulation) on supported
    /// GPUs, falling back to IEEE-754 FP32 on CPU or pre-Ampere devices.
    /// </summary>
    TF32 = 1,

    /// <summary>
    /// Brain Float 16-bit (<c>bfloat16</c>) forward/backward compute with FP32 master weights and reductions.
    /// </summary>
    BF16 = 2,

    /// <summary>
    /// IEEE-754 16-bit (<c>float16</c>) forward/backward compute with dynamic loss scaling and FP32 master weights.
    /// </summary>
    FP16 = 3,
}
