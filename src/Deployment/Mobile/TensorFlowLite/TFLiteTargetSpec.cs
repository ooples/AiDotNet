namespace AiDotNet.Deployment.Mobile.TensorFlowLite;

/// <summary>
/// TensorFlow Lite target specification for compatibility.
/// </summary>
public enum TFLiteTargetSpec
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
