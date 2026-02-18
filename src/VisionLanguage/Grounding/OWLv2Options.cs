namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for OWLv2: self-training for scaling open-vocabulary detection.
/// </summary>
public class OWLv2Options : GroundingVLMOptions
{
    public OWLv2Options()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 24;
        NumDecoderLayers = 6;
        NumHeads = 16;
        ImageSize = 960;
        MaxDetections = 100;
    }

    public int NumClassEmbeddings { get; set; } = 768;

    /// <summary>Gets or sets whether self-training augmentation is enabled.</summary>
    public bool EnableSelfTraining { get; set; } = true;
}
