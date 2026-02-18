namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Ferret: spatial-aware visual sampler for free-form region referring and grounding.
/// </summary>
public class FerretOptions : GroundingVLMOptions
{
    public FerretOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        MaxDetections = 100;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to accept free-form region inputs (points, boxes, scribbles).</summary>
    public bool EnableFreeFormRegions { get; set; } = true;
}
