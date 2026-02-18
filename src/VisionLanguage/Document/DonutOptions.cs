namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for Donut: OCR-free document understanding transformer using Swin encoder + BART decoder.
/// </summary>
public class DonutOptions : DocumentVLMOptions
{
    public DonutOptions()
    {
        VisionDim = 1024;
        DecoderDim = 1024;
        NumVisionLayers = 12;
        NumDecoderLayers = 4;
        NumHeads = 16;
        ImageSize = 2560;
    }

    /// <summary>Gets or sets the visual encoder architecture type.</summary>
    public string EncoderType { get; set; } = "Swin";
}
