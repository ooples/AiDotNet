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
    /// 1e-4 is the SAM-family fine-tuning rate, the same value SlimSAM carries in this repo.
    /// RepViTSAM was silently inheriting the segmentation base's GENERIC 1e-3, ten times higher,
    /// and at that rate its training invariants moved the wrong way: measured at runner parity, the
    /// loss rose from 0.644082 to 0.650058 over ten iterations, worse than the untrained baseline,
    /// with all 7,329,457 parameters finite. So the gradients were fine and the step size was not.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The learning rate is how big a step training takes each time it
    /// corrects the model. Too small and it learns slowly; too large and it steps past the answer
    /// and can end up worse than where it started, which is exactly what was happening here.
    /// </para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>
    /// The decoupled AdamW weight decay used when the caller supplies no optimizer.
    /// </summary>
    public double WeightDecay { get; set; } = 0.01;
}
