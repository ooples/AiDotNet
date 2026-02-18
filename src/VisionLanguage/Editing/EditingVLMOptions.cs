using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Editing;

/// <summary>
/// Base configuration options for image editing VLMs.
/// </summary>
public class EditingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the output image resolution.</summary>
    public int OutputImageSize { get; set; } = 512;

    /// <summary>Gets or sets the number of diffusion denoising steps.</summary>
    public int NumDiffusionSteps { get; set; } = 50;

    /// <summary>Gets or sets the guidance scale for classifier-free guidance.</summary>
    public double GuidanceScale { get; set; } = 7.5;
}
