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
}
