namespace AiDotNet.Enums;

/// <summary>
/// Specifies where a tensor's data is stored.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This tells you whether tensor data is in regular memory (CPU) or graphics card memory (GPU).
///
/// - CPU: Normal computer memory, accessible by your program directly
/// - GPU: Graphics card memory, much faster for parallel operations but requires special access
/// - Distributed: Spread across multiple computers or GPUs
/// </para>
/// </remarks>
public enum TensorLocation
{
    /// <summary>
    /// Tensor data is stored in CPU memory (system RAM).
    /// </summary>
    CPU,

    /// <summary>
    /// Tensor data is stored in GPU memory (VRAM).
    /// </summary>
    GPU,

    /// <summary>
    /// Tensor data is distributed across multiple devices.
    /// </summary>
    Distributed
}
