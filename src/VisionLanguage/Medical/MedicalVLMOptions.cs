using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Medical;

/// <summary>
/// Base configuration options for medical domain vision-language models.
/// </summary>
public class MedicalVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the medical domain specialization.</summary>
    public string MedicalDomain { get; set; } = "General";

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the maximum output token length for report generation.</summary>
    public int MaxOutputTokens { get; set; } = 512;
}
