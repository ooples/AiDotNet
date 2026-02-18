namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for QVQ-72B: first open-source multimodal reasoning model from Qwen.
/// </summary>
public class QVQ72BOptions : ReasoningVLMOptions
{
    public QVQ72BOptions()
    {
        VisionDim = 1152;
        DecoderDim = 8192;
        NumVisionLayers = 27;
        NumDecoderLayers = 80;
        NumHeads = 64;
        ImageSize = 448;
        VocabSize = 152064;
        LanguageModelName = "Qwen2.5";
        MaxReasoningTokens = 2048;
        ReasoningApproach = "RL-Aligned";
    }

    /// <summary>Gets or sets the total parameter count in billions.</summary>
    public int TotalParameters { get; set; } = 72;
}
