namespace AiDotNet.Enums;

/// <summary>
/// Target hardware platforms for model deployment and optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Different devices and platforms have different hardware capabilities.
/// This enum helps you specify where your AI model will run, allowing the library to optimize
/// the model specifically for that platform. For example:
/// - CPU: Traditional computer processors (slowest but most compatible)
/// - GPU: Graphics cards (much faster for AI workloads)
/// - TensorRT: NVIDIA's optimized AI inference engine (fastest for NVIDIA GPUs)
/// - Mobile: Smartphones and tablets (limited power, needs optimization)
/// - Edge: Small devices like Raspberry Pi or Arduino (very limited resources)
/// </remarks>
public enum TargetPlatform
{
    /// <summary>Generic CPU - most compatible but slower for AI workloads</summary>
    CPU,

    /// <summary>Generic GPU - faster than CPU for AI computations</summary>
    GPU,

    /// <summary>NVIDIA GPU with TensorRT - optimized for NVIDIA GPUs</summary>
    TensorRT,

    /// <summary>Mobile devices (iOS/Android) - requires size and power optimizations</summary>
    Mobile,

    /// <summary>Edge devices (Raspberry Pi, etc.) - very limited resources</summary>
    Edge,

    /// <summary>iOS with CoreML - Apple's machine learning framework</summary>
    CoreML,

    /// <summary>Android with NNAPI - Android's Neural Networks API</summary>
    NNAPI,

    /// <summary>TensorFlow Lite - lightweight models for mobile and edge devices</summary>
    TFLite,

    /// <summary>WebAssembly - run AI models in web browsers</summary>
    WebAssembly
}
