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
public class MixedQueryTransformerOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MixedQueryTransformerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MixedQueryTransformerOptions(MixedQueryTransformerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
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

}
