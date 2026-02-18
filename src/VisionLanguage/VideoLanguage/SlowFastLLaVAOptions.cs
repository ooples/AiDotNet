namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// Configuration options for SlowFast-LLaVA: token-efficient slow/fast pathways for long video.
/// </summary>
public class SlowFastLLaVAOptions : VideoLanguageOptions
{
    public SlowFastLLaVAOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        MaxFrames = 64;
    }

    /// <summary>Gets or sets the number of slow pathway frames (high-detail).</summary>
    public int SlowFrames { get; set; } = 8;

    /// <summary>Gets or sets the number of fast pathway frames (low-detail).</summary>
    public int FastFrames { get; set; } = 64;
}
