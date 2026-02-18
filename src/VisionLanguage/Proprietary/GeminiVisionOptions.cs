using AiDotNet.VisionLanguage.Proprietary;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Configuration options for Gemini Vision.
/// </summary>
public class GeminiVisionOptions : ProprietaryVLMOptions
{
    public GeminiVisionOptions()
    {
        VisionDim = 1024;
        DecoderDim = 8192;
        NumVisionLayers = 32;
        NumDecoderLayers = 64;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 256000;
        Provider = "Google";
        LanguageModelName = "Gemini-MoE";
        MaxContextLength = 2000000;
    }

    /// <summary>Gets or sets the maximum context length in tokens.</summary>
    public int MaxContextTokens { get; set; } = 2000000;

    /// <summary>Gets or sets the number of MoE experts.</summary>
    public int NumExperts { get; set; } = 16;
}
