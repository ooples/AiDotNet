namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Nougat: neural OCR for academic documents converting PDF to Markdown.
/// </summary>
public class NougatOptions : DocumentVLMOptions
{
    public NougatOptions()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 12;
        NumDecoderLayers = 4;
        NumHeads = 16;
        ImageSize = 896;
    }

    /// <summary>Gets or sets the output format (Markdown, LaTeX, etc.).</summary>
    public string OutputFormat { get; set; } = "Markdown";
}
