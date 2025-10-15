namespace AiDotNet.Enums;

/// <summary>
/// Represents the type of Neural Processing Unit (NPU) available for hardware acceleration.
/// </summary>
public enum NPUType
{
    /// <summary>
    /// No NPU is available.
    /// </summary>
    None,

    /// <summary>
    /// Apple Neural Engine.
    /// </summary>
    AppleNeuralEngine,

    /// <summary>
    /// Google Edge TPU.
    /// </summary>
    GoogleEdgeTPU,

    /// <summary>
    /// Intel Neural Compute Stick or Movidius.
    /// </summary>
    IntelNCS,

    /// <summary>
    /// Qualcomm Hexagon DSP with AI acceleration.
    /// </summary>
    QualcommHexagon,

    /// <summary>
    /// NVIDIA Deep Learning Accelerator (DLA).
    /// </summary>
    NVDLA,

    /// <summary>
    /// Huawei Ascend AI processor.
    /// </summary>
    HuaweiAscend,

    /// <summary>
    /// Samsung Exynos NPU.
    /// </summary>
    SamsungExynos,

    /// <summary>
    /// MediaTek APU (AI Processing Unit).
    /// </summary>
    MediaTekAPU,

    /// <summary>
    /// Other or unknown NPU type.
    /// </summary>
    Other
}