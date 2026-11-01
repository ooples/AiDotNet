using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// Configuration options for SlimSAM (pruned SAM with 1.4% params).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SlimSAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class SlimSAMOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SlimSAMOptions()
    {
        NumClasses = 1;
        DropRate = 0;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SlimSAMOptions(SlimSAMOptions other)
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
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>
    /// Gets or sets the channel width of each of the image encoder's four stages, from the highest-resolution stage
    /// to the lowest.
    /// </summary>
    /// <value>Exactly four positive widths. Default: { 64, 128, 320, 768 }.</value>
    public int[] ChannelDimensions { get; set; } = [64, 128, 320, 768];

    /// <summary>
    /// Gets or sets the number of blocks in each of the image encoder's four stages, in the same order as
    /// <see cref="ChannelDimensions"/>.
    /// </summary>
    /// <value>Exactly four positive block counts. Default: { 2, 2, 4, 12 }.</value>
    public int[] StageDepths { get; set; } = [2, 2, 4, 12];

    /// <summary>Gets or sets the channel width of the mask decoder.</summary>
    /// <value>A positive width. Default: 256.</value>
    public int DecoderDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the learning rate of the default AdamW optimizer, used when no optimizer is supplied to the model.
    /// </summary>
    /// <value>Default: 1e-4.</value>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the decoupled weight decay of the default AdamW optimizer, used when no optimizer is supplied to
    /// the model.
    /// </summary>
    /// <value>Default: 0.01.</value>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
