using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Configuration options for the Segment Anything Model (SAM).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM is Meta AI's foundation model for image segmentation.
/// Options inherit from NeuralNetworkOptions and can be extended with SAM-specific settings.
/// </para>
/// </remarks>
public class SAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SAMOptions(SAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        AdamBeta1 = other.AdamBeta1;
        AdamBeta2 = other.AdamBeta2;
        AdamEpsilon = other.AdamEpsilon;
        WarmupSteps = other.WarmupSteps;
        MaskFocalWeight = other.MaskFocalWeight;
        MaskDiceWeight = other.MaskDiceWeight;
        FocalAlpha = other.FocalAlpha;
        FocalGamma = other.FocalGamma;
    }

    /// <summary>
    /// Gets or sets the initial AdamW learning rate. The default, 8e-4, is the
    /// Segment Anything paper's training value.
    /// </summary>
    public double LearningRate { get; set; } = 8e-4;

    /// <summary>
    /// Gets or sets the decoupled AdamW weight decay. The paper default is 0.1.
    /// </summary>
    public double WeightDecay { get; set; } = 0.1;

    /// <summary>Gets or sets AdamW's first-moment decay. The paper default is 0.9.</summary>
    public double AdamBeta1 { get; set; } = 0.9;

    /// <summary>Gets or sets AdamW's second-moment decay. The paper default is 0.999.</summary>
    public double AdamBeta2 { get; set; } = 0.999;

    /// <summary>Gets or sets AdamW's numerical-stability epsilon.</summary>
    public double AdamEpsilon { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the linear-warmup length in optimizer steps. The default, 250, is the Segment
    /// Anything paper's value (Kirillov et al. 2023, §A Training algorithm): the learning rate above
    /// is the PEAK reached only after warming up over the first 250 iterations.
    /// </summary>
    public int WarmupSteps { get; set; } = 250;

    /// <summary>
    /// Gets or sets the focal-loss weight in the mask objective. The default, 20, is the paper's
    /// focal:dice ratio of 20:1 (Kirillov et al. 2023, §3).
    /// </summary>
    public double MaskFocalWeight { get; set; } = 20.0;

    /// <summary>
    /// Gets or sets the dice-loss weight in the mask objective. The default, 1, is the paper's
    /// focal:dice ratio of 20:1.
    /// </summary>
    public double MaskDiceWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets focal loss's focusing parameter gamma. The default, 2, is the RetinaNet value
    /// the Segment Anything paper cites.
    /// </summary>
    public double FocalGamma { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets focal loss's class-balance parameter alpha. The default, 0.25, is the RetinaNet
    /// value the Segment Anything paper cites.
    /// </summary>
    public double FocalAlpha { get; set; } = 0.25;
}
