namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Skywork R1V2: hybrid RL (MPO + GRPO) for multimodal reasoning SOTA.
/// </summary>
public class SkyworkR1V2Options : ReasoningVLMOptions
{
    public SkyworkR1V2Options()
    {
        VisionDim = 1152;
        DecoderDim = 8192;
        NumVisionLayers = 27;
        NumDecoderLayers = 80;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 152064;
        LanguageModelName = "Qwen2.5";
        ReasoningApproach = "HybridRL";
        MaxReasoningTokens = 4096;
    }

    /// <summary>Gets or sets whether to use hybrid RL alignment (MPO + GRPO).</summary>
    public bool EnableHybridRL { get; set; } = true;
}
