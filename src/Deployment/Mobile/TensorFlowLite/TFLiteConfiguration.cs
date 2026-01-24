using AiDotNet.Deployment.Export;
using AiDotNet.Enums;

namespace AiDotNet.Deployment.Mobile.TensorFlowLite;

/// <summary>
/// Configuration for TensorFlow Lite model export.
/// </summary>
public class TFLiteConfiguration
{
    /// <summary>
    /// Gets or sets whether to use post-training quantization (default: true).
    /// </summary>
    public bool UseQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the quantization mode.
    /// </summary>
    public QuantizationMode QuantizationMode { get; set; } = Enums.QuantizationMode.Int8;

    /// <summary>
    /// Gets or sets whether to enable GPU delegate (default: false).
    /// </summary>
    public bool EnableGpuDelegate { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use NNAPI delegate for Android (default: false).
    /// </summary>
    public bool UseNnapiDelegate { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use XNNPACK delegate (default: true).
    /// </summary>
    public bool UseXnnpackDelegate { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of threads for CPU inference (default: 4).
    /// </summary>
    public int NumThreads { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to enable operator fusion (default: true).
    /// </summary>
    public bool EnableOperatorFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable constant folding (default: true).
    /// </summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string? ModelName { get; set; }

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string? ModelDescription { get; set; }

    /// <summary>
    /// Gets or sets whether to use dynamic range quantization (default: false).
    /// </summary>
    public bool UseDynamicRangeQuantization { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use integer-only quantization (default: false).
    /// </summary>
    public bool UseIntegerOnlyQuantization { get; set; } = false;

    /// <summary>
    /// Gets or sets the target specification for compatibility.
    /// </summary>
    public TFLiteTargetSpec TargetSpec { get; set; } = new TFLiteTargetSpec();

    /// <summary>
    /// Converts to ExportConfiguration.
    /// </summary>
    public ExportConfiguration ToExportConfiguration()
    {
        return new ExportConfiguration
        {
            TargetPlatform = TargetPlatform.Mobile,
            OptimizeModel = EnableOperatorFusion || EnableConstantFolding,
            QuantizationMode = UseQuantization ? QuantizationMode : Enums.QuantizationMode.None,
            ModelName = ModelName,
            ModelDescription = ModelDescription,
            BatchSize = 1
        };
    }

    /// <summary>
    /// Creates a configuration for Android deployment.
    /// </summary>
    public static TFLiteConfiguration ForAndroid()
    {
        return new TFLiteConfiguration
        {
            UseQuantization = true,
            QuantizationMode = Enums.QuantizationMode.Int8,
            UseNnapiDelegate = true,
            UseXnnpackDelegate = true,
            NumThreads = 4,
            EnableOperatorFusion = true,
            EnableConstantFolding = true
        };
    }

    /// <summary>
    /// Creates a configuration for iOS deployment.
    /// </summary>
    public static TFLiteConfiguration ForIOS()
    {
        return new TFLiteConfiguration
        {
            UseQuantization = true,
            QuantizationMode = Enums.QuantizationMode.Int8,
            EnableGpuDelegate = true,
            UseXnnpackDelegate = true,
            NumThreads = 4,
            EnableOperatorFusion = true,
            EnableConstantFolding = true
        };
    }

    /// <summary>
    /// Creates a configuration optimized for CPU inference.
    /// </summary>
    public static TFLiteConfiguration ForCPU(int numThreads = 4)
    {
        return new TFLiteConfiguration
        {
            UseQuantization = true,
            QuantizationMode = Enums.QuantizationMode.Int8,
            EnableGpuDelegate = false,
            UseNnapiDelegate = false,
            UseXnnpackDelegate = true,
            NumThreads = numThreads,
            EnableOperatorFusion = true,
            EnableConstantFolding = true
        };
    }

    /// <summary>
    /// Creates a configuration with full integer quantization.
    /// </summary>
    public static TFLiteConfiguration ForIntegerOnly()
    {
        return new TFLiteConfiguration
        {
            UseQuantization = true,
            QuantizationMode = Enums.QuantizationMode.Int8,
            UseIntegerOnlyQuantization = true,
            UseDynamicRangeQuantization = false,
            EnableOperatorFusion = true,
            EnableConstantFolding = true
        };
    }
}
