namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.
/// </summary>
public class OWLViTOptions : GroundingVLMOptions
{
    public OWLViTOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 6;
        NumHeads = 12;
        ImageSize = 768;
        MaxDetections = 100;
    }

    /// <summary>Gets or sets the class embedding dimension.</summary>
    public int NumClassEmbeddings { get; set; } = 512;
}
