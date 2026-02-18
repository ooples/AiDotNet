using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Proprietary;

/// <summary>
/// Base configuration options for proprietary VLM reference implementations.
/// </summary>
public class ProprietaryVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the proprietary model provider name.</summary>
    public string Provider { get; set; } = "Unknown";

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "Proprietary";

    /// <summary>Gets or sets the maximum context window length in tokens.</summary>
    public int MaxContextLength { get; set; } = 128000;
}
