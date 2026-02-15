using AiDotNet.Enums;


namespace AiDotNet.Deployment.Export;

/// <summary>
/// Configuration options for model export operations.
/// </summary>
public class ExportConfiguration
{
    /// <summary>
    /// Gets or sets the target ONNX opset version (default: 13).
    /// </summary>
    public int OpsetVersion { get; set; } = 13;

    /// <summary>
    /// Gets or sets whether to optimize the exported model (default: true).
    /// </summary>
    public bool OptimizeModel { get; set; } = true;

    /// <summary>
    /// Gets or sets the quantization mode for the exported model.
    /// </summary>
    public QuantizationMode QuantizationMode { get; set; } = QuantizationMode.None;

    /// <summary>
    /// Gets or sets whether to use dynamic input shapes (default: false).
    /// </summary>
    public bool UseDynamicShapes { get; set; } = false;

    /// <summary>
    /// Gets or sets the batch size for static shapes (default: 1).
    /// </summary>
    public int BatchSize { get; set; } = 1;

    /// <summary>
    /// Gets or sets the input shape dimensions (excluding batch dimension).
    /// </summary>
    public int[]? InputShape { get; set; }

    /// <summary>
    /// Gets or sets the output shape dimensions (excluding batch dimension).
    /// </summary>
    /// <remarks>
    /// <para>If not specified and the model implements <see cref="AiDotNet.Interfaces.ILayeredModel{T}"/>,
    /// the output shape will be inferred from the last layer's output shape.</para>
    /// </remarks>
    public int[]? OutputShape { get; set; }

    /// <summary>
    /// Gets or sets whether to include metadata in the exported model (default: true).
    /// </summary>
    public bool IncludeMetadata { get; set; } = true;

    /// <summary>
    /// Gets or sets the target hardware platform for optimization.
    /// </summary>
    public TargetPlatform TargetPlatform { get; set; } = TargetPlatform.CPU;

    /// <summary>
    /// Gets or sets custom operator mappings for unsupported operations.
    /// </summary>
    public Dictionary<string, string> CustomOperatorMappings { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to perform model validation after export (default: true).
    /// </summary>
    public bool ValidateAfterExport { get; set; } = true;

    /// <summary>
    /// Gets or sets additional platform-specific options.
    /// </summary>
    public Dictionary<string, object> PlatformSpecificOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the model name to include in metadata.
    /// </summary>
    public string? ModelName { get; set; }

    /// <summary>
    /// Gets or sets the model version to include in metadata.
    /// </summary>
    public string? ModelVersion { get; set; }

    /// <summary>
    /// Gets or sets the model description to include in metadata.
    /// </summary>
    public string? ModelDescription { get; set; }

    /// <summary>
    /// Creates a default configuration for TensorRT export.
    /// </summary>
    public static ExportConfiguration ForTensorRT(int batchSize = 1, bool useFp16 = true)
    {
        return new ExportConfiguration
        {
            TargetPlatform = TargetPlatform.TensorRT,
            BatchSize = batchSize,
            OptimizeModel = true,
            QuantizationMode = useFp16 ? QuantizationMode.Float16 : QuantizationMode.None,
            UseDynamicShapes = false
        };
    }

    /// <summary>
    /// Creates a default configuration for mobile export.
    /// </summary>
    public static ExportConfiguration ForMobile(QuantizationMode quantization = QuantizationMode.Int8)
    {
        return new ExportConfiguration
        {
            TargetPlatform = TargetPlatform.Mobile,
            OptimizeModel = true,
            QuantizationMode = quantization,
            BatchSize = 1
        };
    }

    /// <summary>
    /// Creates a default configuration for edge devices.
    /// </summary>
    public static ExportConfiguration ForEdge()
    {
        return new ExportConfiguration
        {
            TargetPlatform = TargetPlatform.Edge,
            OptimizeModel = true,
            QuantizationMode = QuantizationMode.Int8,
            BatchSize = 1,
            UseDynamicShapes = false
        };
    }
}
