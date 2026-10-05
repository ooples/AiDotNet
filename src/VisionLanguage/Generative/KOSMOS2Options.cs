using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.VisionLanguage.Generative;

/// <summary>
/// Options for KOSMOS-2 (grounded multimodal with text spans linked to bounding boxes; Peng et al. 2023).
/// </summary>
/// <remarks>
/// The defaults follow the released model (microsoft/kosmos-2-patch14-224):
/// <list type="bullet">
/// <item>A vocabulary of 65037, including the grounding tokens.</item>
/// <item>Sinusoidal positions.</item>
/// <item>64 latent queries.</item>
/// <item>A 32 x 32 grid of location bins.</item>
/// </list>
/// </remarks>
public class KOSMOS2Options : KosmosOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KOSMOS2Options(KOSMOS2Options other)
        : base(other)
    {
        EnableGroundingTokens = other.EnableGroundingTokens;
        NumLocationBins = other.NumLocationBins;
    }

    /// <summary>Initializes the options with the released model's values.</summary>
    public KOSMOS2Options()
    {
        VocabSize = 65037;
        EnableGroundingTokens = true;
        NumLocationBins = 1024;
    }

    /// <summary>Gets or sets whether location tokens for grounding are enabled.</summary>
    public bool EnableGroundingTokens { get; set; }

    /// <summary>
    /// Gets or sets the number of location token bins. KOSMOS-2 splits the image into a 32 x 32 grid, so
    /// 1024 <c>&lt;patch_index_*&gt;</c> tokens.
    /// </summary>
    public int NumLocationBins { get; set; }
}
