namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for UReader: universal OCR-free visually-situated language model.
/// </summary>
public class UReaderOptions : DocumentVLMOptions
{
    public UReaderOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets whether to use shape-adaptive cropping for varied document layouts.</summary>
    public bool EnableShapeAdaptiveCropping { get; set; } = true;
}
