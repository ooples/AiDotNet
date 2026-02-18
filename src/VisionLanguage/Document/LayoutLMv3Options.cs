namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for LayoutLMv3: unified text, image, and layout pre-training for document AI.
/// </summary>
public class LayoutLMv3Options : DocumentVLMOptions
{
    public LayoutLMv3Options()
    {
        VisionDim = 768;
        DecoderDim = 768;
        NumVisionLayers = 12;
        NumDecoderLayers = 12;
        NumHeads = 12;
        ImageSize = 224;
        IsOcrFree = false;
    }

    /// <summary>Gets or sets the maximum number of layout tokens.</summary>
    public int MaxLayoutTokens { get; set; } = 512;
}
