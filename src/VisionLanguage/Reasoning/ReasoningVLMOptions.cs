using AiDotNet.VisionLanguage.InstructionTuned;

namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Base configuration options for reasoning vision-language models with chain-of-thought capabilities.
/// </summary>
public class ReasoningVLMOptions : InstructionTunedVLMOptions
{
    /// <summary>Gets or sets the reasoning approach (e.g., "CoT", "RL-Aligned", "MoE-Reasoning").</summary>
    public string ReasoningApproach { get; set; } = "CoT";

    /// <summary>Gets or sets the maximum number of reasoning tokens before the final answer.</summary>
    public int MaxReasoningTokens { get; set; } = 1024;

    /// <summary>Gets or sets whether to include explicit thinking steps in the output.</summary>
    public bool EnableThinkingSteps { get; set; } = true;
}
