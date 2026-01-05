namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// Defines the type of GPU stream for operation scheduling.
/// Different stream types enable compute/transfer overlap.
/// </summary>
public enum GpuStreamType
{
    /// <summary>
    /// Default stream that synchronizes with all other streams.
    /// Operations on this stream block until all prior operations complete.
    /// </summary>
    Default = 0,

    /// <summary>
    /// Compute stream for kernel execution.
    /// Multiple compute streams can run kernels concurrently if resources allow.
    /// </summary>
    Compute = 1,

    /// <summary>
    /// Transfer stream for host-to-device memory copies.
    /// Can overlap with compute operations on separate streams.
    /// </summary>
    HostToDevice = 2,

    /// <summary>
    /// Transfer stream for device-to-host memory copies.
    /// Can overlap with compute operations on separate streams.
    /// </summary>
    DeviceToHost = 3,

    /// <summary>
    /// Transfer stream for device-to-device memory copies.
    /// Used for multi-GPU scenarios or internal buffer management.
    /// </summary>
    DeviceToDevice = 4
}
