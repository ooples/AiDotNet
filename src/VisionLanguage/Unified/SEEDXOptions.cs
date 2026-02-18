namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for SEED-X: multi-granularity comprehension and generation model.
/// </summary>
public class SEEDXOptions : UnifiedVisionOptions
{
    public SEEDXOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-2";
        NumVisualTokens = 8192;
    }

    /// <summary>Gets or sets whether to use multi-granularity visual encoding.</summary>
    public bool EnableMultiGranularity { get; set; } = true;
}
