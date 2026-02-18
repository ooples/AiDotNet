namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Janus-Pro: scaled data and model with optimized training strategy.
/// </summary>
public class JanusProOptions : UnifiedVisionOptions
{
    public JanusProOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 384;
        VocabSize = 32000;
        LanguageModelName = "DeepSeek-LLM";
        NumVisualTokens = 16384;
    }

    public bool EnableDecoupledEncoding { get; set; } = true;
}
