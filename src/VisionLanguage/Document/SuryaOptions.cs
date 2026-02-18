namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Surya: multi-language OCR with layout analysis support.
/// </summary>
public class SuryaOptions : DocumentVLMOptions
{
    public SuryaOptions()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 6;
        NumHeads = 12;
        ImageSize = 896;
    }

    /// <summary>Gets or sets the number of supported languages.</summary>
    public int NumLanguages { get; set; } = 90;

    /// <summary>Gets or sets whether to perform layout analysis.</summary>
    public bool EnableLayoutAnalysis { get; set; } = true;
}
