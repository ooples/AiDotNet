namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for DINO-X: strongest open-world perception model.
/// </summary>
public class DINOXOptions : GroundingVLMOptions
{
    public DINOXOptions()
    {
        VisionDim = 256;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        ImageSize = 800;
        MaxDetections = 300;
    }

    public int NumQueryPositions { get; set; } = 900;

    /// <summary>Gets or sets whether to enable universal perception mode.</summary>
    public bool EnableUniversalPerception { get; set; } = true;
}
