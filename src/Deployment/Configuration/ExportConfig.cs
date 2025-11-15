using AiDotNet.Enums;

namespace AiDotNet.Deployment.Configuration;

/// <summary>
/// Configuration for exporting models to different formats and platforms.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After training an AI model, you often need to export it to a specific
/// format depending on where it will run. Think of it like exporting a document to PDF, Word, or
/// plain text - same content, different format for different uses.
///
/// Export Formats:
/// - ONNX: Universal format that works everywhere (recommended for most cases)
/// - TensorRT: NVIDIA GPUs only, maximum performance on NVIDIA hardware
/// - CoreML: Apple devices (iPhone, iPad, Mac), optimized for Apple Silicon
/// - TFLite: Android devices and edge hardware, very efficient
/// - WASM: Run models in web browsers without plugins
///
/// When to export:
/// - Deploying to production servers (ONNX or TensorRT)
/// - Mobile apps (CoreML for iOS, TFLite for Android)
/// - Edge devices like Raspberry Pi (TFLite)
/// - Web applications (WASM)
///
/// Optimization:
/// Most export formats support optimization and quantization to make models smaller and faster.
/// </para>
/// </remarks>
public class ExportConfig
{
    /// <summary>
    /// Gets or sets the target platform for export (default: CPU).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Choose where your model will run:
    /// CPU for general servers, GPU for graphics cards, TensorRT for NVIDIA GPUs,
    /// CoreML for Apple devices, NNAPI for Android, etc.
    /// </para>
    /// </remarks>
    public TargetPlatform TargetPlatform { get; set; } = TargetPlatform.CPU;

    /// <summary>
    /// Gets or sets whether to optimize the exported model (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Optimization makes models run faster.
    /// Recommended to keep true. May increase export time but improves inference speed.
    /// </para>
    /// </remarks>
    public bool OptimizeModel { get; set; } = true;

    /// <summary>
    /// Gets or sets the quantization mode for export (default: None).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Quantization compresses the model.
    /// None = full precision, Float16 = half precision, Int8 = maximum compression.
    /// See QuantizationConfig for more details.
    /// </para>
    /// </remarks>
    public QuantizationMode Quantization { get; set; } = QuantizationMode.None;

    /// <summary>
    /// Gets or sets the batch size for static shapes (default: 1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many inputs to process at once.
    /// 1 = process one at a time (typical for real-time inference).
    /// Higher values can be faster but use more memory.
    /// </para>
    /// </remarks>
    public int BatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets whether to include model metadata (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Include name, version, and description in the exported model.
    /// Helpful for documentation. Minimal file size impact.
    /// </para>
    /// </remarks>
    public bool IncludeMetadata { get; set; } = true;

    /// <summary>
    /// Gets or sets the model name to include in metadata (optional).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A friendly name for your model (e.g., "HousePricePredictor").
    /// Helps identify which model is which when you have many.
    /// </para>
    /// </remarks>
    public string? ModelName { get; set; }

    /// <summary>
    /// Gets or sets the model version to include in metadata (optional).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Version number (e.g., "1.2.3").
    /// Helps track which version of the model you're using.
    /// </para>
    /// </remarks>
    public string? ModelVersion { get; set; }

    /// <summary>
    /// Gets or sets the model description to include in metadata (optional).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Brief description of what the model does.
    /// Useful documentation for others (or your future self).
    /// </para>
    /// </remarks>
    public string? ModelDescription { get; set; }

    /// <summary>
    /// Gets or sets whether to validate the exported model (default: true).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Check that the exported model works correctly.
    /// Recommended to keep true - catches export errors before deployment.
    /// Adds export time but prevents broken models.
    /// </para>
    /// </remarks>
    public bool ValidateAfterExport { get; set; } = true;

    /// <summary>
    /// Creates an export configuration for ONNX format (universal, CPU).
    /// </summary>
    /// <returns>An ONNX export configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for general-purpose deployment.
    /// ONNX works everywhere - servers, cloud, edge devices. Good starting point.
    /// </para>
    /// </remarks>
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
    /// <param name="quantization">The quantization mode to use (default: Float16).</param>
    /// <returns>A TensorRT export configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for NVIDIA GPU deployment.
    /// Provides maximum performance on NVIDIA hardware. Float16 recommended for speed/accuracy balance.
    /// </para>
    /// </remarks>
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
    /// <param name="quantization">The quantization mode to use (default: Float16).</param>
    /// <returns>A CoreML export configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for iPhone/iPad apps.
    /// Optimized for Apple devices with Neural Engine. Float16 recommended.
    /// Batch size 1 is typical for mobile.
    /// </para>
    /// </remarks>
    public static ExportConfig ForCoreML(QuantizationMode quantization = QuantizationMode.Float16)
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.CoreML,
            OptimizeModel = true,
            Quantization = quantization,
            BatchSize = 1
        };
    }

    /// <summary>
    /// Creates an export configuration for TensorFlow Lite (Android/edge devices).
    /// </summary>
    /// <param name="quantization">The quantization mode to use (default: Int8).</param>
    /// <returns>A TFLite export configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this for Android apps or edge devices like Raspberry Pi.
    /// Int8 quantization recommended for smallest size. Batch size 1 typical for mobile.
    /// </para>
    /// </remarks>
    public static ExportConfig ForTFLite(QuantizationMode quantization = QuantizationMode.Int8)
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.NNAPI,
            OptimizeModel = true,
            Quantization = quantization,
            BatchSize = 1
        };
    }

    /// <summary>
    /// Creates an export configuration for WebAssembly (browser).
    /// </summary>
    /// <returns>A WebAssembly export configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to run models in web browsers.
    /// Float16 reduces download size. Works on any device with a web browser.
    /// </para>
    /// </remarks>
    public static ExportConfig ForWebAssembly()
    {
        return new ExportConfig
        {
            TargetPlatform = TargetPlatform.WebAssembly,
            OptimizeModel = true,
            Quantization = QuantizationMode.Float16,
            BatchSize = 1
        };
    }
}
