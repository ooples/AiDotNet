using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Foundational;

/// <summary>
/// Configuration options for UNITER (Universal Image-TExt Representation).
/// </summary>
/// <remarks>
/// <para>UNITER (Chen et al., ECCV 2020) uses conditional masking pre-training where either
/// image regions or text tokens are masked, forcing the model to learn cross-modal alignment.
/// It uses a single-stream transformer for joint image-text encoding.</para>
/// </remarks>
public class UNITEROptions : FoundationalVLMOptions
{
    /// <summary>Gets or sets the image region masking probability during training.</summary>
    public double ImageMaskProbability { get; set; } = 0.15;

    /// <summary>Gets or sets the text token masking probability during training.</summary>
    public double TextMaskProbability { get; set; } = 0.15;

    /// <summary>Gets or sets the maximum number of image regions.</summary>
    public int MaxImageRegions { get; set; } = 36;

    public UNITEROptions()
    {
        FusionType = FusionType.SingleStream;
        VisualFeatureType = VisualFeatureType.RegionFeatures;
        VisionDim = 2048;
        TextDim = 768;
        FusionDim = 768;
        NumFusionLayers = 12;
    }
}
