namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for LongVILA: long-context visual language model for 1hr+ videos.
/// </summary>
public class LongVILAOptions : VideoLanguageOptions
{
    public LongVILAOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MaxFrames = 256;
    }

    /// <summary>Gets or sets the maximum video duration in minutes.</summary>
    public int MaxVideoMinutes { get; set; } = 60;
}
