using AiDotNet.VisionLanguage.Proprietary;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Configuration options for Grok Vision.
/// </summary>
public class GrokVisionOptions : ProprietaryVLMOptions
{
    public GrokVisionOptions()
    {
        VisionDim = 1024;
        DecoderDim = 8192;
        NumVisionLayers = 32;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 131072;
        Provider = "xAI";
        LanguageModelName = "Grok";
        MaxContextLength = 128000;
    }

    /// <summary>Gets or sets whether real-time data access is enabled.</summary>
    public bool RealTimeAccess { get; set; } = true;
}
