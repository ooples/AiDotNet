namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for DocPedia: frequency-domain document understanding model.
/// </summary>
public class DocPediaOptions : DocumentVLMOptions
{
    public DocPediaOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use frequency-domain processing.</summary>
    public bool EnableFrequencyDomain { get; set; } = true;
}
