namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Configuration options for Skywork R1V: cross-modal transfer of reasoning LLMs to vision.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SkyworkR1V model. Default values follow the original paper settings.</para>
/// </remarks>
public class SkyworkR1VOptions : ReasoningVLMOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SkyworkR1VOptions(SkyworkR1VOptions other)
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
        EnableCrossModalTransfer = other.EnableCrossModalTransfer;
    }

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
