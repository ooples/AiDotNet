namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Janus: decoupled visual encoding for understanding vs generation.
/// </summary>
public class JanusOptions : UnifiedVisionOptions
{
    public JanusOptions()
    {
        VisionDim = 1024;
        DecoderDim = 2048;
        NumVisionLayers = 24;
        NumDecoderLayers = 24;
        NumHeads = 16;
        ImageSize = 384;
        VocabSize = 32000;
        LanguageModelName = "DeepSeek-LLM";
        NumVisualTokens = 8192;
    }

    /// <summary>Gets or sets whether to use decoupled visual encoding paths.</summary>
    public bool EnableDecoupledEncoding { get; set; } = true;
}
