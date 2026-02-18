namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Shikra: referential dialogue model with coordinate output for grounding.
/// </summary>
public class ShikraOptions : GroundingVLMOptions
{
    public ShikraOptions()
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

    /// <summary>Gets or sets whether to produce coordinate outputs in dialogue.</summary>
    public bool EnableCoordinateOutput { get; set; } = true;
}
