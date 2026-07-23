namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for LLaVA-CoT: chain-of-thought visual reasoning with structured output.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the LLaVACoT model. Default values follow the original paper settings.</para>
/// </remarks>
public class LLaVACoTOptions : ReasoningVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LLaVACoTOptions(LLaVACoTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        ImageSize = other.ImageSize;
        VisionDim = other.VisionDim;
        DecoderDim = other.DecoderDim;
        NumVisionLayers = other.NumVisionLayers;
        NumDecoderLayers = other.NumDecoderLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        MaxSequenceLength = other.MaxSequenceLength;
        MaxGenerationLength = other.MaxGenerationLength;
        DropoutRate = other.DropoutRate;
        ArchitectureType = other.ArchitectureType;
        ImageMean = other.ImageMean;
        ImageStd = other.ImageStd;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        InstructionArchitectureType = other.InstructionArchitectureType;
        ProjectionDim = other.ProjectionDim;
        LanguageModelName = other.LanguageModelName;
        MaxVisualTokens = other.MaxVisualTokens;
        SystemPrompt = other.SystemPrompt;
        ReasoningApproach = other.ReasoningApproach;
        MaxReasoningTokens = other.MaxReasoningTokens;
        EnableThinkingSteps = other.EnableThinkingSteps;
        EnableStructuredReasoning = other.EnableStructuredReasoning;
    }

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
