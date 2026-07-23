namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Configuration for adaptive inference.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> AdaptiveInferenceConfig provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class AdaptiveInferenceConfig
{
    /// <summary>Gets or sets the quality level.</summary>
    public QualityLevel QualityLevel { get; set; }

    /// <summary>Gets or sets whether to use quantization.</summary>
    public bool UseQuantization { get; set; }

    /// <summary>Gets or sets the quantization bit width.</summary>
    public int QuantizationBits { get; set; }

    /// <summary>Gets or sets the layers to skip for speed.</summary>
    public List<string> SkipLayers { get; set; } = new();
}
