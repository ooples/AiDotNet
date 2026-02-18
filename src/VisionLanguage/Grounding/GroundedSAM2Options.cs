namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounded-SAM 2: combines Grounding DINO with SAM 2 for grounded segmentation and tracking.
/// </summary>
public class GroundedSAM2Options : GroundingVLMOptions
{
    public GroundedSAM2Options()
    {
        VisionDim = 256;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        ImageSize = 1024;
        MaxDetections = 300;
    }

    /// <summary>Gets or sets whether to produce segmentation masks.</summary>
    public bool EnableSegmentation { get; set; } = true;

    /// <summary>Gets or sets whether to enable video object tracking.</summary>
    public bool EnableTracking { get; set; } = true;
}
