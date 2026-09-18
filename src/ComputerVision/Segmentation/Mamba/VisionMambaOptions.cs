using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>
/// Configuration options for Vision Mamba (Vim) bidirectional SSM segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the VisionMamba model. Default values follow the original paper settings.</para>
/// </remarks>
public class VisionMambaOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public VisionMambaOptions()
    {
        NumClasses = 150;
        DropRate = 0.1;
        ModelSize = VisionMambaModelSize.Tiny;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public VisionMambaOptions(VisionMambaOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the AdamW learning rate used by the ADE20K recipe.</summary>
    /// <value>Defaults to <c>6e-5</c>, as specified for Vision Mamba ADE20K training.</value>
    /// <remarks><para><b>For Beginners:</b> This controls the size of each optimizer update.</para></remarks>
    public double LearningRate { get; set; } = 6e-5;

    /// <summary>Gets or sets the AdamW decoupled weight decay used by the ADE20K recipe.</summary>
    /// <value>Defaults to <c>0.01</c>, as specified for Vision Mamba ADE20K training.</value>
    /// <remarks><para><b>For Beginners:</b> Weight decay regularizes the learned weights.</para></remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>VisionMambaModelSize.Tiny</c>.
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
    public VisionMambaModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
