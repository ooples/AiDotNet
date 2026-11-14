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
    public double ThingStuffThreshold { get; set; } = 0.08;

    /// <summary>
    /// Gets or sets the minimum score of an instance pasted by the panoptic merge. Default: 0.4 (the CUPS
    /// Panoptic Cascade Mask R-CNN's <c>confidence_threshold</c>).
    /// </summary>
    public double InstanceScoreThreshold { get; set; } = 0.4;

    /// <summary>
    /// Gets or sets the largest share of an instance that may already be claimed when it is pasted. Default:
    /// 0.5 (Panoptic FPN).
    /// </summary>
    public double OverlapThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the smallest stuff region, in input-image pixels, kept by the panoptic merge. Default: 4096
    /// (Panoptic FPN).
    /// </summary>
    public double StuffAreaThreshold { get; set; } = 4096;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
