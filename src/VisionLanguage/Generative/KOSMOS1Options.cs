using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// Options for KOSMOS-1 (Huang et al. 2023, "Language Is Not All You Need: Aligning Perception with Language
/// Models").
/// </summary>
/// <remarks>
/// KOSMOS-1 has no released code. The defaults follow the paper:
/// <list type="bullet">
/// <item>A MAGNETO decoder with xPos relative positions and a 64K vocabulary.</item>
/// <item>A CLIP ViT-L/14 image encoder.</item>
/// <item>A Flamingo-style perceiver resampler with 64 latents. Its depth follows Flamingo (6), because the
/// paper does not state one.</item>
/// </list>
/// </remarks>
public class KOSMOS1Options : KosmosOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KOSMOS1Options(KOSMOS1Options other)
        : base(other)
    {
        ResamplerDepth = other.ResamplerDepth;
    }

    /// <summary>Initializes the options with the paper's values.</summary>
    public KOSMOS1Options()
    {
        VocabSize = 64000;
        ResamplerDepth = 6;
    }

    /// <summary>Gets or sets the perceiver resampler depth. Flamingo: 6.</summary>
    public int ResamplerDepth { get; set; }
}
