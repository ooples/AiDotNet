namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Identifies the DirectGpu backend type for hardware capability reporting.
/// </summary>
public enum GpuBackendType
{
    None,
    Cuda,
    OpenCl,
    Hip,
    Unknown
}
