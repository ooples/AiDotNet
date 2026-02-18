namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Grounding DINO: open-set detection combining DINO with grounded pre-training.
/// </summary>
public class GroundingDINOOptions : GroundingVLMOptions
{
    public GroundingDINOOptions()
    {
        VisionDim = 256;
        DecoderDim = 256;
        NumVisionLayers = 6;
        NumDecoderLayers = 6;
        NumHeads = 8;
        ImageSize = 800;
        MaxDetections = 300;
    }

    /// <summary>Gets or sets the number of object query positions.</summary>
    public int NumQueryPositions { get; set; } = 900;
}
