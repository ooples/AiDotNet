namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Skywork R1V: cross-modal transfer of reasoning LLMs to vision.
/// </summary>
public class SkyworkR1VOptions : ReasoningVLMOptions
{
    public SkyworkR1VOptions()
    {
        VisionDim = 1152;
        DecoderDim = 4096;
        NumVisionLayers = 27;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 448;
        VocabSize = 152064;
        LanguageModelName = "Qwen2.5";
        ReasoningApproach = "CrossModal-Transfer";
    }

    /// <summary>Gets or sets whether to enable cross-modal reasoning transfer.</summary>
    public bool EnableCrossModalTransfer { get; set; } = true;
}
