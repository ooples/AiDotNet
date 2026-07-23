namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// Information about an NNAPI-capable device.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class describes a hardware accelerator available for NNAPI.
/// Android devices may have multiple accelerators (CPU, GPU, NPU, DSP) each with different
/// capabilities and performance characteristics.
/// </para>
/// </remarks>
public class NNAPIDeviceInfo
{
    /// <summary>
    /// Gets or sets the device name (e.g., "Qualcomm Adreno GPU", "Google EdgeTPU").
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the device type.
    /// </summary>
    public NNAPIDevice Type { get; set; } = NNAPIDevice.Auto;

    /// <summary>
    /// Gets or sets the NNAPI feature level supported by this device.
    /// </summary>
    /// <remarks>
    /// Higher feature levels support more operations. Android 10+ = level 4, Android 11+ = level 5.
    /// </remarks>
    public int FeatureLevel { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether this device supports FP16 operations.
    /// </summary>
    public bool SupportsFp16 { get; set; } = false;

    /// <summary>
    /// Gets or sets whether this device supports INT8 quantized operations.
    /// </summary>
    public bool SupportsInt8 { get; set; } = false;

    /// <summary>
    /// Gets or sets the relative performance score (higher is faster, 0-100).
    /// </summary>
    public int PerformanceScore { get; set; } = 0;

    /// <summary>
    /// Gets or sets the relative power efficiency score (higher is more efficient, 0-100).
    /// </summary>
    public int PowerEfficiencyScore { get; set; } = 0;
}
