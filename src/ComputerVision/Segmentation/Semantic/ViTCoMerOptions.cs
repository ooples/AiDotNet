using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the ViT-CoMer semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-CoMer options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. ViT-CoMer is a hybrid model that runs CNN and transformer
/// branches in parallel and fuses them to get excellent boundary quality in segmentation.
/// </para>
/// </remarks>
public class ViTCoMerOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ViTCoMerOptions()
    {
        NumClasses = 150;
        DropRate = 0.1;
        ModelSize = ViTCoMerModelSize.Small;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ViTCoMerOptions(ViTCoMerOptions other)
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
    }

    /// <summary>
    /// Gets or sets the initial AdamW learning rate. The default follows the ViT-CoMer
    /// ADE20K training configuration from the original paper.
    /// </summary>
    public double LearningRate { get; set; } = 6e-5;

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>ViTCoMerModelSize.Small</c>.
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
    public ViTCoMerModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
