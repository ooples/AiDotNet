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
}
