namespace AiDotNet.Deployment.Mobile.TensorFlowLite;

/// <summary>
/// TensorFlow Lite target type for compatibility.
/// </summary>
public enum TFLiteTargetType
{
    /// <summary>Default target (all operations)</summary>
    Default,

    /// <summary>Integer operations only</summary>
    IntegerOnly,

    /// <summary>Optimized for mobile GPU</summary>
    MobileGPU,

    /// <summary>Optimized for Edge TPU</summary>
    EdgeTPU
}

/// <summary>
/// TensorFlow Lite target specification with detailed platform requirements.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class specifies what hardware and software requirements
/// your TFLite model needs to run. Use this to ensure your model works on your target devices.
/// </para>
/// </remarks>
public class TFLiteTargetSpec
{
    /// <summary>
    /// Gets or sets the target type.
    /// </summary>
    public TFLiteTargetType TargetType { get; set; } = TFLiteTargetType.Default;

    /// <summary>
    /// Gets or sets the minimum Android SDK version required (default: 21 = Android 5.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines which Android versions can run your model:
    /// - 21 = Android 5.0 (oldest supported)
    /// - 26 = Android 8.0 (required for NNAPI)
    /// - 29 = Android 10 (NNAPI 1.2 with better performance)
    /// </para>
    /// </remarks>
    public int AndroidMinSdkVersion { get; set; } = 21;

    /// <summary>
    /// Gets or sets whether to support GPU acceleration (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GPU acceleration can make inference 2-10x faster on supported devices.
    /// However, not all operations are supported on GPU. Enable only if your model is GPU-compatible.
    /// </para>
    /// </remarks>
    public bool SupportGpu { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to support Hexagon DSP acceleration (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Hexagon DSP is Qualcomm's digital signal processor found in Snapdragon chips.
    /// It's very efficient for quantized (INT8) models. Only enable for Qualcomm devices.
    /// </para>
    /// </remarks>
    public bool SupportHexagonDsp { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to support Edge TPU acceleration (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Google's Edge TPU is a specialized AI accelerator found in Coral devices
    /// and some Pixel phones. Requires specially compiled models.
    /// </para>
    /// </remarks>
    public bool SupportEdgeTpu { get; set; } = false;

    /// <summary>
    /// Gets or sets the minimum iOS version for iOS targets (default: "12.0").
    /// </summary>
    public string MinimumIosVersion { get; set; } = "12.0";

    /// <summary>
    /// Creates a spec for Android with NNAPI support.
    /// </summary>
    public static TFLiteTargetSpec ForAndroidNnapi()
    {
        return new TFLiteTargetSpec
        {
            AndroidMinSdkVersion = 26, // NNAPI available from Android 8.0
            SupportGpu = true,
            SupportHexagonDsp = false
        };
    }

    /// <summary>
    /// Creates a spec for Qualcomm Snapdragon devices.
    /// </summary>
    public static TFLiteTargetSpec ForQualcomm()
    {
        return new TFLiteTargetSpec
        {
            AndroidMinSdkVersion = 26,
            SupportGpu = true,
            SupportHexagonDsp = true
        };
    }

    /// <summary>
    /// Creates a spec for Google Coral/Edge TPU devices.
    /// </summary>
    public static TFLiteTargetSpec ForEdgeTpu()
    {
        return new TFLiteTargetSpec
        {
            TargetType = TFLiteTargetType.EdgeTPU,
            SupportEdgeTpu = true,
            SupportGpu = false
        };
    }

    /// <summary>
    /// Gets the default specification (CPU-only, wide compatibility).
    /// </summary>
    public static TFLiteTargetSpec Default => new TFLiteTargetSpec();
}
