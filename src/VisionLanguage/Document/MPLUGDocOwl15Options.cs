namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// Configuration options for mPLUG-DocOwl 1.5: unified structure learning achieving SOTA on 10 document benchmarks.
/// </summary>
public class MPLUGDocOwl15Options : DocumentVLMOptions
{
    public MPLUGDocOwl15Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
    }

    public int AbstractorDim { get; set; } = 1024;

    public int NumAbstractorLayers { get; set; } = 6;

    /// <summary>Gets or sets whether to use unified structure learning.</summary>
    public bool EnableUnifiedStructureLearning { get; set; } = true;
}
