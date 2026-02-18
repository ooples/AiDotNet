using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Base configuration options for unified understanding + generation vision models.
/// </summary>
public class UnifiedVisionOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets whether the model supports image generation.</summary>
    public bool SupportsGeneration { get; set; } = true;

    /// <summary>Gets or sets the generated image resolution.</summary>
    public int OutputImageSize { get; set; } = 512;

    /// <summary>Gets or sets the number of discrete visual tokens in the vocabulary.</summary>
    public int NumVisualTokens { get; set; } = 8192;
}
