namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for OmniGen2: dual-path architecture with parameter decoupling.
/// </summary>
public class OmniGen2Options : UnifiedVisionOptions
{
    public OmniGen2Options()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 512;
        VocabSize = 32000;
        LanguageModelName = "Phi-3";
        NumVisualTokens = 16384;
    }

    /// <summary>Gets or sets whether to use dual-path architecture.</summary>
    public bool EnableDualPath { get; set; } = true;
}
