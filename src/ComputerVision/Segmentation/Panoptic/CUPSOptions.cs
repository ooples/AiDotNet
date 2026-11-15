using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// Configuration options for CUPS unsupervised panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CUPS model. Default values follow the original paper settings.</para>
/// </remarks>
public class CUPSOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CUPSOptions()
    {
        // CUPS clusters 27 pseudo classes (Hahn et al. 2025, Sec. 4: "By default, CUPS uses 27 pseudo classes").
        NumClasses = 27;
        DropRate = 0.1;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CUPSOptions(CUPSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ChannelDimensions = (int[])other.ChannelDimensions.Clone();
        StageDepths = (int[])other.StageDepths.Clone();
        DecoderDimension = other.DecoderDimension;
        ThingStuffThreshold = other.ThingStuffThreshold;
        InstanceScoreThreshold = other.InstanceScoreThreshold;
        OverlapThreshold = other.OverlapThreshold;
        StuffAreaThreshold = other.StuffAreaThreshold;
    }

    /// <summary>Feature widths for the four hierarchical encoder stages.</summary>
    public int[] ChannelDimensions { get; set; } = [96, 192, 384, 768];

    /// <summary>Block counts for the four hierarchical encoder stages.</summary>
    public int[] StageDepths { get; set; } = [2, 2, 6, 2];

    /// <summary>Feature width used by the panoptic decoder.</summary>
    public int DecoderDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets <c>psi_ts</c>: a pseudo class is a thing when the share of its pixels inside instance pseudo
    /// masks exceeds this value, and stuff otherwise. Default: 0.08 (Hahn et al. 2025).
    /// </summary>
    /// <value>A fraction in [0, 1]. The default is 0.08.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> CUPS learns which of its pseudo classes are countable objects ("things", such as
    /// cars) and which are regions ("stuff", such as sky). A class counts as a thing when more than this fraction of
    /// its pixels fall inside instance masks. A larger value makes fewer classes things.</para>
    /// </remarks>
    public double ThingStuffThreshold { get; set; } = 0.08;

    /// <summary>
    /// Gets or sets the minimum score of an instance pasted by the panoptic merge. Default: 0.4 (the CUPS
    /// Panoptic Cascade Mask R-CNN's <c>confidence_threshold</c>).
    /// </summary>
    /// <value>A score in [0, 1]. The default is 0.4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each detected object has a confidence score. Objects scoring below this value are
    /// left out of the final panoptic map; a larger value keeps only the more certain detections.</para>
    /// </remarks>
    public double InstanceScoreThreshold { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets the largest share of an instance that may already be claimed when it is pasted. Default:
    /// 0.5 (Panoptic FPN).
    /// </summary>
    /// <value>A fraction in [0, 1]. The default is 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Objects are pasted onto the map from the most to the least confident. An object is
    /// skipped when more than this fraction of its pixels already belong to an object pasted before it, so a smaller
    /// value removes more overlapping duplicates.</para>
    /// </remarks>
    public double OverlapThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the smallest stuff region, in input-image pixels, kept by the panoptic merge. Default: 4096
    /// (Panoptic FPN).
    /// </summary>
    /// <value>A non-negative pixel count, measured at the input resolution. The default is 4096.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> After the objects are placed, each stuff class keeps its remaining pixels only if
    /// they cover at least this many pixels, which removes small, noisy regions. Zero keeps every region.</para>
    /// </remarks>
    public double StuffAreaThreshold { get; set; } = 4096;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1), when
    /// <see cref="ThingStuffThreshold"/>, <see cref="InstanceScoreThreshold"/> or <see cref="OverlapThreshold"/> is not
    /// a fraction in [0, 1], or when <see cref="StuffAreaThreshold"/> is negative or not finite.
    /// </exception>
    public void Validate()
    {
        ValidateSegmentationCore();
        RequireFraction(ThingStuffThreshold, nameof(ThingStuffThreshold));
        RequireFraction(InstanceScoreThreshold, nameof(InstanceScoreThreshold));
        RequireFraction(OverlapThreshold, nameof(OverlapThreshold));
        if (double.IsNaN(StuffAreaThreshold) || double.IsInfinity(StuffAreaThreshold) || StuffAreaThreshold < 0)
            throw new ArgumentOutOfRangeException(nameof(StuffAreaThreshold), StuffAreaThreshold,
                "The stuff area threshold must be a finite, non-negative pixel count.");
    }

    // A NaN fails both comparisons, so it is rejected too.
    private static void RequireFraction(double value, string name)
    {
        if (!(value >= 0 && value <= 1))
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be a fraction in [0, 1].");
    }
}
