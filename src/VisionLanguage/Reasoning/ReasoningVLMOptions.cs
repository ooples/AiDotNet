using AiDotNet.VisionLanguage.InstructionTuned;

namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Base configuration options for reasoning vision-language models with chain-of-thought capabilities.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Reasoning model. Default values follow the original paper settings.</para>
/// </remarks>
public class ReasoningVLMOptions : InstructionTunedVLMOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ReasoningVLMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ReasoningVLMOptions(ReasoningVLMOptions other)
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
    }

    /// <summary>Gets or sets the reasoning approach (e.g., "CoT", "RL-Aligned", "MoE-Reasoning").</summary>
    public string ReasoningApproach { get; set; } = "CoT";

    /// <summary>Gets or sets the maximum number of reasoning tokens before the final answer.</summary>
    public int MaxReasoningTokens { get; set; } = 1024;

    /// <summary>Gets or sets whether to include explicit thinking steps in the output.</summary>
    public bool EnableThinkingSteps { get; set; } = true;
}
