namespace AiDotNet.Models.Options;

/// <summary>
/// Options shared by the referring segmentation models (LISA, VideoLISA, OMG-LLaVA, PixelLM), which
/// turn a referring expression into one text embedding and decode a binary mask from it.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A referring model is given a sentence such as "the dog on the left" and returns the
/// pixels it describes. All four models share the same last step: the sentence becomes one text embedding, each
/// pixel's embedding is compared with it, and the resulting score decides whether the pixel is in the mask. These
/// options control that shared step and the text encoding (inherited from
/// <see cref="TextQueriedSegmentationOptions"/>). Network sizes, such as encoder widths and depths, are configured on
/// each model's own options class.</para>
/// <para><b>Reference:</b> Lai et al., "LISA: Reasoning Segmentation via Large Language Model", CVPR 2024.</para>
/// </remarks>
public abstract class ReferringSegmentationOptions : TextQueriedSegmentationOptions
{
    /// <summary>
    /// Gets or sets the probability a pixel's sigmoid mask score must exceed to be in the mask.
    /// </summary>
    /// <value>
    /// Defaults to 0.5: LISA (Lai et al. 2024) keeps a pixel when its mask logit is above 0, which is a
    /// sigmoid probability above 0.5. Must lie in (0, 1).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every pixel gets a score between 0 and 1 for how well it matches the sentence. Pixels
    /// scoring above this value form the mask, so a higher threshold gives a smaller, more certain mask.</para>
    /// </remarks>
    public double MaskThreshold { get; set; } = 0.5;

    /// <summary>
    /// Copies the text-query and mask-threshold settings from <paramref name="other"/>.
    /// </summary>
    protected void CopyReferringSettings(ReferringSegmentationOptions other)
    {
        CopyTextQuerySettings(other);
        MaskThreshold = other.MaskThreshold;
    }

    /// <summary>
    /// Throws when <see cref="MaskThreshold"/> is not in (0, 1).
    /// </summary>
    internal void ValidateMaskThreshold()
    {
        if (double.IsNaN(MaskThreshold) || MaskThreshold <= 0.0 || MaskThreshold >= 1.0)
            throw new ArgumentException(
                $"{GetType().Name}.{nameof(MaskThreshold)} is {MaskThreshold.ToString(CultureInfo.InvariantCulture)}, " +
                "but it must lie strictly between 0 and 1.",
                OptionsParameterName);
    }
}
