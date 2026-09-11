using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// Configuration options for SEEM interactive segmentation.
/// </summary>
/// <remarks>
/// <para>
/// By default, the model-size selector uses the released Focal-T configuration (96-channel stem,
/// stage depths 2/2/6/2, and a 512-wide decoder). Set the optional topology properties to load a
/// custom or reduced checkpoint without changing the production defaults.
/// </para>
/// <para><b>For Beginners:</b> These options configure SEEM and allow every native stage size to be customized.</para>
/// </remarks>
public class SEEMOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SEEMOptions()
    {
        NumClasses = 133;
        DropRate = 0.0;
        ModelSize = SEEMModelSize.Tiny;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SEEMOptions(SEEMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        ChannelDimensions = other.ChannelDimensions?.ToArray();
        StageDepths = other.StageDepths?.ToArray();
        DecoderDimension = other.DecoderDimension;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets custom four-stage Focal channel widths, or null to use the selected released variant.</summary>
    public int[]? ChannelDimensions { get; set; }

    /// <summary>Gets or sets custom four-stage Focal block depths, or null to use the selected released variant.</summary>
    public int[]? StageDepths { get; set; }

    /// <summary>Gets or sets a custom mask-decoder width, or null to use the released 512-wide decoder.</summary>
    public int? DecoderDimension { get; set; }

    /// <summary>Gets or sets the AdamW learning rate used by native training.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>Gets or sets the AdamW weight decay used by native training.</summary>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>SEEMModelSize.Tiny</c>.
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
    public SEEMModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
