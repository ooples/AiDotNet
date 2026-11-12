namespace AiDotNet.Models.Options;

/// <summary>
/// Options shared by the referring segmentation models (LISA, VideoLISA, OMG-LLaVA, PixelLM), which
/// turn a referring expression into one text embedding and decode a binary mask from it.
/// </summary>
public abstract class ReferringSegmentationOptions : TextQueriedSegmentationOptions
{
    /// <summary>
    /// Gets or sets the probability a pixel's sigmoid mask score must exceed to be in the mask.
    /// </summary>
    /// <value>
    /// Defaults to 0.5: LISA (Lai et al. 2024) keeps a pixel when its mask logit is above 0, which is a
    /// sigmoid probability above 0.5. Must lie in (0, 1).
    /// </value>
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
