namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Configuration options for Groma: localized visual tokenization for grounded understanding.
/// </summary>
public class GromaOptions : GroundingVLMOptions
{
    public GromaOptions()
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

    /// <summary>Gets or sets whether to use localized visual tokenization.</summary>
    public bool EnableLocalizedTokenization { get; set; } = true;
}
