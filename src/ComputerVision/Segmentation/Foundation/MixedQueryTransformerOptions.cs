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
    /// Optimizer learning rate. Default 1e-4, the Mask2Former training recipe this architecture
    /// inherits (AdamW, lr 1e-4, weight decay 0.05).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why this exists.</b> The constructor previously built its optimizer bare, so training ran
    /// on AdamW's own generic default of 0.001 -- ten times this value -- and no caller could
    /// correct it, because there was no property to set. Both halves of the rule were broken: the
    /// default was not the paper's, and it was not overridable.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate controls how large a step training takes on each
    /// update. Too large and training diverges; too small and it crawls. The value here is the one
    /// this architecture's authors trained with, so prefer it unless you have measured a reason to
    /// change it.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

}
