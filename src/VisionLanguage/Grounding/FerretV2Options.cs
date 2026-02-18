namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Ferret-v2: improved referring and grounding with enhanced spatial understanding.
/// </summary>
public class FerretV2Options : GroundingVLMOptions
{
    public FerretV2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        MaxDetections = 100;
        VocabSize = 32000;
    }

    public bool EnableFreeFormRegions { get; set; } = true;

    /// <summary>Gets or sets whether to use high-resolution input processing.</summary>
    public bool EnableHighResolution { get; set; } = true;
}
