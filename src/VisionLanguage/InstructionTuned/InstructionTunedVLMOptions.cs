using AiDotNet.VisionLanguage.Encoders;
using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.InstructionTuned;

/// <summary>
/// Base configuration options for instruction-tuned vision-language models.
/// </summary>
public class InstructionTunedVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the instruction-tuned architecture type.</summary>
    public InstructionTunedArchitectureType InstructionArchitectureType { get; set; } = InstructionTunedArchitectureType.MLPProjection;

    /// <summary>Gets or sets the MLP projection hidden dimension (for MLP connector models).</summary>
    public int ProjectionDim { get; set; } = 4096;

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the maximum number of visual tokens per image.</summary>
    public int MaxVisualTokens { get; set; } = 576;

    /// <summary>Gets or sets the system prompt for chat mode.</summary>
    public string SystemPrompt { get; set; } = "You are a helpful assistant.";
}
