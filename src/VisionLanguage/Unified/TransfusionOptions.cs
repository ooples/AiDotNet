namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Configuration options for Transfusion: combined autoregressive and diffusion loss in single transformer.
/// </summary>
public class TransfusionOptions : UnifiedVisionOptions
{
    public TransfusionOptions()
    {
        VisionDim = 4096;
        DecoderDim = 4096;
        NumVisionLayers = 0;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 256;
        VocabSize = 32000;
        LanguageModelName = "Transfusion";
        NumVisualTokens = 256;
    }

    /// <summary>Gets or sets whether to use diffusion loss for image generation.</summary>
    public bool EnableDiffusionLoss { get; set; } = true;
}
