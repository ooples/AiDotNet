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
        NumClasses = 133;
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
    }

    /// <summary>Feature widths for the four hierarchical encoder stages.</summary>
    public int[] ChannelDimensions { get; set; } = [96, 192, 384, 768];

    /// <summary>Block counts for the four hierarchical encoder stages.</summary>
    public int[] StageDepths { get; set; } = [2, 2, 6, 2];

    /// <summary>Feature width used by the panoptic decoder.</summary>
    public int DecoderDimension { get; set; } = 256;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
