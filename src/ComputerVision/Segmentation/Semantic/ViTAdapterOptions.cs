using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// Configuration options for the ViT-Adapter semantic segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-Adapter options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. ViT-Adapter enables plain Vision Transformers to handle
/// dense prediction tasks by adding lightweight spatial prior modules, without requiring any
/// vision-specific architectural changes to the base ViT.
/// </para>
/// </remarks>
public class ViTAdapterOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ViTAdapterOptions()
    {
        NumClasses = 150;
        DropRate = 0.1;
        ModelSize = ViTAdapterModelSize.Base;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ViTAdapterOptions(ViTAdapterOptions other)
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

    /// <summary>Gets or sets the AdamW backbone learning rate used by the ADE20K recipe.</summary>
    /// <value>Defaults to <c>2e-5</c>, following the ViT-Adapter ADE20K recipe.</value>
    /// <remarks><para><b>For Beginners:</b> This is the update size used for the pretrained backbone.</para></remarks>
    public double LearningRate { get; set; } = 2e-5;

    /// <summary>Gets or sets the AdamW decoupled weight decay used by the ADE20K recipe.</summary>
    /// <value>Defaults to <c>0.01</c>, following the ViT-Adapter ADE20K recipe.</value>
    /// <remarks><para><b>For Beginners:</b> Weight decay helps prevent overfitting during fine-tuning.</para></remarks>
    public double WeightDecay { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>ViTAdapterModelSize.Base</c>.
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
    public ViTAdapterModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
