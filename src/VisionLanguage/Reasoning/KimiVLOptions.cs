namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Kimi-VL: MoE VLM with MoonViT and long-context processing.
/// </summary>
public class KimiVLOptions : ReasoningVLMOptions
{
    public KimiVLOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 28;
        NumHeads = 32;
        ImageSize = 384;
        VocabSize = 128256;
        LanguageModelName = "MoonshotMoE";
        ReasoningApproach = "MoE-Reasoning";
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 16;

    /// <summary>Gets or sets the active parameter count in billions (MoE routing).</summary>
    public int ActiveParameters { get; set; } = 2;

    /// <summary>Gets or sets whether to enable 128K long-context mode.</summary>
    public bool EnableLongContext { get; set; } = true;
}
