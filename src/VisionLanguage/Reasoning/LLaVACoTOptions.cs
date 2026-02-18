namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for LLaVA-CoT: chain-of-thought visual reasoning with structured output.
/// </summary>
public class LLaVACoTOptions : ReasoningVLMOptions
{
    public LLaVACoTOptions()
    {
        VisionDim = 1024;
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 336;
        VocabSize = 32000;
        LanguageModelName = "LLaMA-3";
        ReasoningApproach = "CoT";
    }

    /// <summary>Gets or sets whether to produce structured reasoning steps.</summary>
    public bool EnableStructuredReasoning { get; set; } = true;
}
