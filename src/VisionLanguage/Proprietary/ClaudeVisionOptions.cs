using AiDotNet.VisionLanguage.Proprietary;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Configuration options for Claude Vision.
/// </summary>
public class ClaudeVisionOptions : ProprietaryVLMOptions
{
    public ClaudeVisionOptions()
    {
        VisionDim = 1024;
        DecoderDim = 8192;
        NumVisionLayers = 32;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 100000;
        Provider = "Anthropic";
        LanguageModelName = "Claude";
        MaxContextLength = 200000;
    }

    /// <summary>Gets or sets whether extended thinking mode is enabled.</summary>
    public bool ExtendedThinking { get; set; } = true;
}
