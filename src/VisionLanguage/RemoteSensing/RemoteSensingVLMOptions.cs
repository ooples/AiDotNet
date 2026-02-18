using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.RemoteSensing;

/// <summary>
/// Base configuration options for remote sensing vision-language models.
/// </summary>
public class RemoteSensingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the supported image bands (e.g., "RGB", "Multispectral").</summary>
    public string SupportedBands { get; set; } = "RGB";

    /// <summary>Gets or sets the language model backbone name.</summary>
    public string LanguageModelName { get; set; } = "LLaMA";

    /// <summary>Gets or sets the ground sample distance in meters.</summary>
    public double GroundSampleDistance { get; set; } = 0.5;
}
