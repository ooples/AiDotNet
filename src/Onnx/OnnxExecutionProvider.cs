namespace AiDotNet.Onnx;

/// <summary>
/// Specifies the execution provider (hardware accelerator) for ONNX model inference.
/// </summary>
/// <remarks>
/// <para>
/// Execution providers allow ONNX models to run on different hardware accelerators.
/// The order of fallback is typically: CUDA/TensorRT → DirectML → CPU.
/// </para>
/// <para><b>For Beginners:</b> Think of execution providers as different engines:
/// <list type="bullet">
/// <item><b>CPU</b>: Works everywhere, slowest but most compatible</item>
/// <item><b>CUDA</b>: NVIDIA GPUs, much faster than CPU</item>
/// <item><b>TensorRT</b>: NVIDIA GPUs with extra optimizations, fastest for NVIDIA</item>
/// <item><b>DirectML</b>: Windows GPUs (AMD, Intel, NVIDIA), good cross-vendor support</item>
/// <item><b>CoreML</b>: Apple Silicon (M1/M2/M3), fastest on Mac</item>
/// </list>
/// </para>
/// </remarks>
public enum OnnxExecutionProvider
{
    /// <summary>
    /// CPU execution provider (default, always available).
    /// </summary>
    Cpu = 0,

    /// <summary>
    /// NVIDIA CUDA execution provider for GPU acceleration.
    /// Requires CUDA toolkit and compatible NVIDIA GPU.
    /// </summary>
    Cuda = 1,

    /// <summary>
    /// NVIDIA TensorRT execution provider for optimized GPU inference.
    /// Provides additional optimizations on top of CUDA.
    /// </summary>
    TensorRT = 2,

    /// <summary>
    /// DirectML execution provider for Windows GPU acceleration.
    /// Works with AMD, Intel, and NVIDIA GPUs on Windows.
    /// </summary>
    DirectML = 3,

    /// <summary>
    /// Apple CoreML execution provider for Apple Silicon.
    /// Optimized for M1/M2/M3 chips on macOS.
    /// </summary>
    CoreML = 4,

    /// <summary>
    /// OpenVINO execution provider for Intel hardware.
    /// Optimized for Intel CPUs and integrated graphics.
    /// </summary>
    OpenVINO = 5,

    /// <summary>
    /// ROCm execution provider for AMD GPUs.
    /// </summary>
    ROCm = 6,

    /// <summary>
    /// NNAPI execution provider for Android devices.
    /// </summary>
    NNAPI = 7,

    /// <summary>
    /// Automatically select the best available provider.
    /// Falls back through: TensorRT → CUDA → DirectML → CPU
    /// </summary>
    Auto = 100
}
