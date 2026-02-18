namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Kimi-VL-Thinking: long chain-of-thought reasoning with RL alignment.
/// </summary>
public class KimiVLThinkingOptions : ReasoningVLMOptions
{
    public KimiVLThinkingOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 28;
        NumHeads = 32;
        ImageSize = 384;
        VocabSize = 128256;
        LanguageModelName = "MoonshotMoE";
        ReasoningApproach = "RL-CoT";
        MaxReasoningTokens = 4096;
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 16;

    /// <summary>Gets or sets the active parameter count in billions.</summary>
    public int ActiveParameters { get; set; } = 2;

    /// <summary>Gets or sets whether to enable extended thinking chains.</summary>
    public bool EnableLongThinking { get; set; } = true;
}
