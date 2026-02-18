namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounding DINO 1.5: enhanced open-set detection with improved architecture.
/// </summary>
public class GroundingDINO15Options : GroundingVLMOptions
{
    public GroundingDINO15Options()
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

    /// <summary>Gets or sets the visual backbone type.</summary>
    public string BackboneType { get; set; } = "Swin-L";
}
