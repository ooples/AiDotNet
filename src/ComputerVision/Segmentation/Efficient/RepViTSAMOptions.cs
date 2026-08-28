using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// Configuration options for RepViT-SAM real-time mobile SAM.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the RepViTSAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class RepViTSAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public RepViTSAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RepViTSAMOptions(RepViTSAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>
    /// The AdamW learning rate used when the caller supplies no optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The RepViT reference training recipe uses AdamW with a 1e-3 learning rate. Keeping the
    /// value in the options preserves caller configurability while making the default explicit.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate controls how large each training update is. This
    /// default follows the reference recipe and can still be changed for a specific data set.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// The decoupled AdamW weight decay used when the caller supplies no optimizer.
    /// </summary>
    public double WeightDecay { get; set; } = 0.025;
}
