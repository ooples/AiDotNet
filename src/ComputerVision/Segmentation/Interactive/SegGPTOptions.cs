using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// Configuration options for SegGPT in-context segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SegGPT model. Default values follow the original paper settings.</para>
/// </remarks>
public class SegGPTOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SegGPTOptions()
    {
        NumClasses = 1;
        DropRate = 0.1;
        ModelSize = SegGPTModelSize.ViTLarge;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SegGPTOptions(SegGPTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        ChannelDimensions = other.ChannelDimensions.ToArray();
        StageDepths = other.StageDepths.ToArray();
        DecoderDimension = other.DecoderDimension;
    }

    /// <summary>
    /// Gets or sets the four hierarchical encoder widths. The defaults reproduce
    /// the ViT-Large configuration used by the original SegGPT implementation.
    /// </summary>
    public int[] ChannelDimensions { get; set; } = [64, 128, 320, 1024];

    /// <summary>
    /// Gets or sets the number of blocks in each encoder stage.
    /// </summary>
    public int[] StageDepths { get; set; } = [2, 2, 4, 24];

    /// <summary>
    /// Gets or sets the mask decoder width.
    /// </summary>
    public int DecoderDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>SegGPTModelSize.ViTLarge</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Most of these models were published in several sizes that
    /// trade accuracy against speed and memory. Picking a variant selects the widths and
    /// depths the paper reports for it; it is not a hint, it changes the network that gets
    /// built.
    /// </para>
    /// <para>
    /// Declared here rather than on <see cref="SegmentationModelOptions"/> because each
    /// model names its own variants with its own enum, so there is no shared type to
    /// declare.
    /// </para>
    /// </remarks>
    public SegGPTModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
