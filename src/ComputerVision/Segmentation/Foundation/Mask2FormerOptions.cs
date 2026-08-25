using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Mask2Former universal segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask2Former options inherit from NeuralNetworkOptions, which provides
/// a Seed property for reproducibility. Mask2Former is a universal model that can perform
/// semantic, instance, and panoptic segmentation with a single architecture by using
/// masked cross-attention in its transformer decoder.
/// </para>
/// </remarks>
public class Mask2FormerOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public Mask2FormerOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Mask2FormerOptions(Mask2FormerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>
    /// Gets or sets the optimizer learning rate. Default is 1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Cheng et al. 2022 ("Masked-attention Mask Transformer for Universal Image Segmentation",
    /// section 4) train with AdamW at 1e-4. The framework default of 1e-3 is an order of magnitude
    /// too aggressive for a 41M-parameter transformer with a Hungarian-matched loss, and measured on
    /// the model-family harness it diverged in a SINGLE step: the loss went from 1.872619 to
    /// 208.665359, with no non-finite parameters -- an overshoot, not a broken gradient.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step training takes each time it learns something.
    /// Too large and each correction overshoots, so the model gets worse instead of better.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the AdamW weight decay. Default is 0.05.
    /// </summary>
    /// <remarks>
    /// The paper's value, and the one this class's own constructor documentation has always claimed
    /// -- it simply had no property to put it in and no override to apply it.
    /// </remarks>
    public double WeightDecay { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the gradient-norm clipping threshold; 0 disables clipping. Default is 0.01.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper clips at 0.01, and for this architecture that is not a refinement. The Hungarian
    /// matching in the loss is a DISCRETE assignment, so a small parameter change can flip which
    /// prediction is matched to which target and produce a large, abrupt gradient. Clipping bounds
    /// what a single such flip can do to the weights.
    /// </para>
    /// </remarks>
    public double MaxGradientNorm { get; set; } = 0.01;
}
