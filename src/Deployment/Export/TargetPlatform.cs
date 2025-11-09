namespace AiDotNet.Deployment.Export;

/// <summary>
/// Target hardware platforms for model deployment.
/// </summary>
public enum TargetPlatform
{
    /// <summary>Generic CPU</summary>
    CPU,

    /// <summary>Generic GPU</summary>
    GPU,

    /// <summary>NVIDIA GPU with TensorRT</summary>
    TensorRT,

    /// <summary>Mobile devices (iOS/Android)</summary>
    Mobile,

    /// <summary>Edge devices</summary>
    Edge,

    /// <summary>iOS with CoreML</summary>
    CoreML,

    /// <summary>Android with NNAPI</summary>
    NNAPI,

    /// <summary>WebAssembly</summary>
    WebAssembly
}
