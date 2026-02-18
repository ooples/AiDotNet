namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for TextMonkey: OCR-free text understanding with shifted window attention.
/// </summary>
public class TextMonkeyOptions : DocumentVLMOptions
{
    public TextMonkeyOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 896;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use shifted window attention for text-heavy images.</summary>
    public bool EnableShiftedWindowAttention { get; set; } = true;
}
