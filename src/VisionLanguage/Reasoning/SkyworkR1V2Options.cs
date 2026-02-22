namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Skywork R1V2: hybrid RL (MPO + GRPO) for multimodal reasoning SOTA.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SkyworkR1V2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class SkyworkR1V2Options : ReasoningVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SkyworkR1V2Options(SkyworkR1V2Options other)
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
        EnableHybridRL = other.EnableHybridRL;
    }

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
