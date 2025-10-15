namespace AiDotNet.Enums;

/// <summary>
/// Represents the type of GPU available for hardware acceleration.
/// </summary>
public enum GPUType
{
    /// <summary>
    /// No GPU is available.
    /// </summary>
    None,

    /// <summary>
    /// NVIDIA GPU with CUDA support.
    /// </summary>
    NVIDIA,

    /// <summary>
    /// AMD GPU with ROCm or OpenCL support.
    /// </summary>
    AMD,

    /// <summary>
    /// Intel integrated or discrete GPU.
    /// </summary>
    Intel,

    /// <summary>
    /// Apple Silicon GPU (M1, M2, etc.).
    /// </summary>
    AppleSilicon,

    /// <summary>
    /// Qualcomm Adreno GPU.
    /// </summary>
    Qualcomm,

    /// <summary>
    /// ARM Mali GPU.
    /// </summary>
    ARM,

    /// <summary>
    /// Other or unknown GPU type.
    /// </summary>
    Other
}