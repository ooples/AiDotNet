using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for exporting models to different formats and platforms.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> After training an AI model, you often need to export it to a specific
/// format depending on where it will run. Think of it like exporting a document to PDF, Word, or
/// plain text - same content, different format for different uses.
///
/// **Export Formats:**
/// - **ONNX**: Universal format that works everywhere (recommended for most cases)
/// - **TensorRT**: NVIDIA GPUs only, maximum performance on NVIDIA hardware
/// - **CoreML**: Apple devices (iPhone, iPad, Mac), optimized for Apple Silicon
/// - **TFLite**: Android devices and edge hardware, very efficient
/// - **WASM**: Run models in web browsers without plugins
///
/// **When to export:**
/// - Deploying to production servers (ONNX or TensorRT)
/// - Mobile apps (CoreML for iOS, TFLite for Android)
/// - Edge devices like Raspberry Pi (TFLite)
/// - Web applications (WASM)
///
/// **Optimization:**
/// Most export formats support optimization and quantization to make models smaller and faster.
/// </remarks>
public class ExportConfig
{
    /// <summary>
    /// Gets or sets the target platform for export (default: CPU).
    /// </summary>
    public TargetPlatform TargetPlatform { get; set; } = TargetPlatform.CPU;

    /// <summary>
    /// Gets or sets whether to optimize the exported model (default: true).
    /// Optimization can make models faster but may increase export time.
    /// </summary>
    public bool OptimizeModel { get; set; } = true;

    /// <summary>
    /// Gets or sets the quantization mode for export (default: None).
    /// Quantization makes models smaller and faster. See QuantizationConfig for details.
    /// </summary>
    public QuantizationMode Quantization { get; set; } = QuantizationMode.None;

    /// <summary>
    /// Gets or sets the batch size for static shapes (default: 1).
    /// Some platforms require fixed batch sizes, others support dynamic batching.
    /// </summary>
    public int BatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to include model metadata (name, version, description) (default: true).
    /// </summary>
    public bool IncludeMetadata { get; set; } = true;

    /// <summary>
    /// Gets or sets the model name to include in metadata (optional).
    /// </summary>
    public string? ModelName { get; set; }

    /// <summary>
    /// Gets or sets the model version to include in metadata (optional).
    /// </summary>
    public string? ModelVersion { get; set; }

    /// <summary>
    /// Gets or sets the model description to include in metadata (optional).
    /// </summary>
    public string? ModelDescription { get; set; }

    /// <summary>
    /// Gets or sets whether to validate the exported model (default: true).
    /// Validation ensures the exported model works correctly but adds export time.
    /// </summary>
    public bool ValidateAfterExport { get; set; } = true;

    /// <summary>
    /// Creates an export configuration for ONNX format (universal, CPU).
    /// </summary>
    public static ExportConfig ForONNX()
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.CPU,
            OptimizeModel = true,
            ValidateAfterExport = true
        };
    }

    /// <summary>
    /// Creates an export configuration for TensorRT (NVIDIA GPU, maximum performance).
    /// </summary>
    public static ExportConfig ForTensorRT(QuantizationMode quantization = QuantizationMode.Float16)
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.TensorRT,
            OptimizeModel = true,
            Quantization = quantization,
            ValidateAfterExport = true
        };
    }

    /// <summary>
    /// Creates an export configuration for CoreML (iOS devices).
    /// </summary>
    public static ExportConfig ForCoreML(QuantizationMode quantization = QuantizationMode.Float16)
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.CoreML,
            OptimizeModel = true,
            Quantization = quantization,
            BatchSize = 1 // iOS typically uses batch size 1
        };
    }

    /// <summary>
    /// Creates an export configuration for TensorFlow Lite (Android/edge devices).
    /// </summary>
    public static ExportConfig ForTFLite(QuantizationMode quantization = QuantizationMode.Int8)
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.NNAPI,
            OptimizeModel = true,
            Quantization = quantization,
            BatchSize = 1 // Mobile typically uses batch size 1
        };
    }

    /// <summary>
    /// Creates an export configuration for WebAssembly (browser).
    /// </summary>
    public static ExportConfig ForWebAssembly()
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.WebAssembly,
            OptimizeModel = true,
            Quantization = QuantizationMode.Float16, // Reduce download size
            BatchSize = 1
        };
    }
}
