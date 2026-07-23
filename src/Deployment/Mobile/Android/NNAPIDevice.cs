namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// NNAPI acceleration devices.
/// </summary>
public enum NNAPIDevice
{
    /// <summary>Automatically select best device</summary>
    Auto,

    /// <summary>CPU</summary>
    CPU,

    /// <summary>GPU (if available)</summary>
    GPU,

    /// <summary>DSP (Digital Signal Processor)</summary>
    DSP,

    /// <summary>NPU (Neural Processing Unit)</summary>
    NPU
}
