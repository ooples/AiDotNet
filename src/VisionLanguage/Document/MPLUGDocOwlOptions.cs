namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for mPLUG-DocOwl: modular MLLM for document understanding with visual abstractor.
/// </summary>
public class MPLUGDocOwlOptions : DocumentVLMOptions
{
    public MPLUGDocOwlOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    /// <summary>Gets or sets the visual abstractor dimension.</summary>
    public int AbstractorDim { get; set; } = 1024;

    /// <summary>Gets or sets the number of abstractor layers.</summary>
    public int NumAbstractorLayers { get; set; } = 6;
}
