using AiDotNet.Deployment.Export;

namespace AiDotNet.Deployment.Optimization.Quantization;

/// <summary>
/// Per-layer quantization parameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> LayerQuantizationParams provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class LayerQuantizationParams
{
    /// <summary>Gets or sets the scale factor for this layer.</summary>
    public double ScaleFactor { get; set; } = 1.0;

    /// <summary>Gets or sets the zero point for this layer.</summary>
    public int ZeroPoint { get; set; } = 0;

    /// <summary>Gets or sets whether to skip quantization for this layer.</summary>
    public bool Skip { get; set; } = false;

    /// <summary>Gets or sets the bit width for this layer (if different from global).</summary>
    public int? BitWidth { get; set; }

    /// <summary>Gets or sets custom quantization mode for this layer.</summary>
    public QuantizationMode? Mode { get; set; }
}
