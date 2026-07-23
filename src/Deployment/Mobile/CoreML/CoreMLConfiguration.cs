using System.Collections.Generic;
using AiDotNet.Deployment.Export;

namespace AiDotNet.Deployment.Mobile.CoreML;

/// <summary>
/// Configuration for CoreML model export.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> CoreMLConfiguration provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class CoreMLConfiguration
{
    /// <summary>
    /// Gets or sets the CoreML specification version (default: 4).
    /// </summary>
    public int SpecVersion { get; set; } = 4;

    /// <summary>
    /// Gets or sets the compute units to use (CPU, GPU, Neural Engine).
    /// </summary>
    public CoreMLComputeUnits ComputeUnits { get; set; } = CoreMLComputeUnits.All;

    /// <summary>
    /// Gets or sets the minimum deployment target iOS version.
    /// </summary>
    public string MinimumDeploymentTarget { get; set; } = "iOS 13.0";

    /// <summary>
    /// Gets or sets whether to use quantization (default: true for mobile).
    /// </summary>
    public bool UseQuantization { get; set; } = true;

    /// <summary>
    /// Gets or sets the quantization bits (8 or 16, default: 8).
    /// </summary>
    public int QuantizationBits { get; set; } = 8;

    /// <summary>
    /// Gets or sets the model name.
    /// </summary>
    public string? ModelName { get; set; }

    /// <summary>
    /// Gets or sets the model author.
    /// </summary>
    public string? ModelAuthor { get; set; }

    /// <summary>
    /// Gets or sets the model license.
    /// </summary>
    public string? ModelLicense { get; set; }

    /// <summary>
    /// Gets or sets the model description.
    /// </summary>
    public string? ModelDescription { get; set; }

    /// <summary>
    /// Gets or sets input feature names and descriptions.
    /// </summary>
    public Dictionary<string, string> InputFeatures { get; set; } = new();

    /// <summary>
    /// Gets or sets output feature names and descriptions.
    /// </summary>
    public Dictionary<string, string> OutputFeatures { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to optimize for size (default: true).
    /// </summary>
    public bool OptimizeForSize { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable flexible input shapes (default: false).
    /// </summary>
    public bool FlexibleInputShapes { get; set; } = false;

    /// <summary>
    /// Converts to ExportConfiguration.
    /// </summary>
    public ExportConfiguration ToExportConfiguration()
    {
        return new ExportConfiguration
        {
            TargetPlatform = TargetPlatform.CoreML,
            OptimizeModel = true,
            QuantizationMode = UseQuantization
                ? (QuantizationBits == 8 ? QuantizationMode.Int8 : QuantizationMode.Float16)
                : QuantizationMode.None,
            ModelName = ModelName,
            ModelDescription = ModelDescription,
            BatchSize = 1
        };
    }

    /// <summary>
    /// Creates a configuration optimized for iPhone.
    /// </summary>
    public static CoreMLConfiguration ForIPhone()
    {
        return new CoreMLConfiguration
        {
            SpecVersion = 4,
            ComputeUnits = CoreMLComputeUnits.All,
            UseQuantization = true,
            QuantizationBits = 8,
            OptimizeForSize = true,
            MinimumDeploymentTarget = "iOS 13.0"
        };
    }

    /// <summary>
    /// Creates a configuration optimized for iPad.
    /// </summary>
    public static CoreMLConfiguration ForIPad()
    {
        return new CoreMLConfiguration
        {
            SpecVersion = 4,
            ComputeUnits = CoreMLComputeUnits.All,
            UseQuantization = true,
            QuantizationBits = 16, // Better quality on iPad
            OptimizeForSize = false,
            MinimumDeploymentTarget = "iOS 13.0"
        };
    }

    /// <summary>
    /// Creates a configuration for Neural Engine optimization.
    /// </summary>
    public static CoreMLConfiguration ForNeuralEngine()
    {
        return new CoreMLConfiguration
        {
            SpecVersion = 4,
            ComputeUnits = CoreMLComputeUnits.NeuralEngine,
            UseQuantization = true,
            QuantizationBits = 8,
            OptimizeForSize = true,
            MinimumDeploymentTarget = "iOS 14.0"
        };
    }
}
