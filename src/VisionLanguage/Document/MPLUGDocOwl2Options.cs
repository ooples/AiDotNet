namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for mPLUG-DocOwl 2: high-res compressing for multi-page document understanding.
/// </summary>
public class MPLUGDocOwl2Options : DocumentVLMOptions
{
    public MPLUGDocOwl2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        MaxPages = 20;
    }

    public int AbstractorDim { get; set; } = 1024;

    public int NumAbstractorLayers { get; set; } = 6;
}
