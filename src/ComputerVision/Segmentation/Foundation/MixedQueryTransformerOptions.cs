using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the MixedQueryTransformer (MQ-Former) model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MixedQueryTransformer dynamically melds instance and stuff queries via
/// cross-attention to scale across diverse datasets. Options inherit from NeuralNetworkOptions.
/// </para>
/// </remarks>
public class MixedQueryTransformerOptions : PanopticSegmentationOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MixedQueryTransformerOptions()
    {
        NumClasses = 133;   // COCO panoptic
        NumQueries = 200;
        DropRate = 0.1;
        ModelSize = MixedQueryTransformerModelSize.R50;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MixedQueryTransformerOptions(MixedQueryTransformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        NumClasses = other.NumClasses;
        NumQueries = other.NumQueries;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
        LearningRate = other.LearningRate;
    }

    /// <summary>
    /// Gets or sets the AdamW learning rate used when the model builds its own optimizer.
    /// </summary>
    /// <value>
    /// Defaults to 1e-4, the AdamW rate of the Mask2Former recipe this architecture builds on
    /// (arXiv:2404.04469).
    /// </value>
    /// <remarks>
    /// <para>
    /// The model previously constructed <c>AdamWOptimizer</c> with no options, so it trained at the
    /// library-wide AdamW default of 1e-3 -- an order of magnitude above the published rate, and not
    /// reachable by a caller short of building the whole optimizer. Supplying your own optimizer
    /// still wins; this is consulted only when the model has to build one.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the backbone size variant. Default: <see cref="MixedQueryTransformerModelSize.R50"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which size of backbone to use. A larger one is more
    /// accurate and slower.</para>
    /// </remarks>
    public MixedQueryTransformerModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate() => ValidateCore();
}
