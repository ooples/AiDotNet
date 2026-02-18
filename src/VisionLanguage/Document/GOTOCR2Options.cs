namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for GOT-OCR2: 580M unified OCR model for text, tables, charts, equations, and music scores.
/// </summary>
public class GOTOCR2Options : DocumentVLMOptions
{
    public GOTOCR2Options()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 1024;
    }

    /// <summary>Gets or sets whether to enable mathematical equation OCR.</summary>
    public bool EnableMathOCR { get; set; } = true;

    /// <summary>Gets or sets whether to enable music score OCR.</summary>
    public bool EnableMusicOCR { get; set; } = true;
}
