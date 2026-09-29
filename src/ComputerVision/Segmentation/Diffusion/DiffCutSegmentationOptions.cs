using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Diffusion;

/// <summary>
/// Configuration options for DiffCut diffusion-based zero-shot segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DiffCut model. Default values follow the original paper settings.</para>
/// </remarks>
public class DiffCutSegmentationOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DiffCutSegmentationOptions()
    {
        NumClasses = 1;
        DropRate = 0;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffCutSegmentationOptions(DiffCutSegmentationOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ChannelDimensions = other.ChannelDimensions.ToArray();
        StageDepths = other.StageDepths.ToArray();
        DecoderDimension = other.DecoderDimension;
    }

    /// <summary>
    /// Gets or sets the Stable-Diffusion U-Net encoder widths used by the native approximation.
    /// </summary>
    public int[] ChannelDimensions { get; set; } = [320, 640, 1280, 1280];

    /// <summary>
    /// Gets or sets the number of residual blocks in each native encoder stage.
    /// </summary>
    public int[] StageDepths { get; set; } = [2, 2, 2, 2];

    /// <summary>
    /// Gets or sets the feature width of the native segmentation decoder.
    /// </summary>
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
