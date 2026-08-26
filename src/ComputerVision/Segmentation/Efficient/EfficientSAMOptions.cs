using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// Configuration options for EfficientSAM (SAMI-pretrained fast SAM).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EfficientSAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class EfficientSAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EfficientSAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EfficientSAMOptions(EfficientSAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>Gets or sets the AdamW learning rate used for dense segmentation fine-tuning.</summary>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the AdamW decoupled weight decay used for dense segmentation fine-tuning.</summary>
    public double WeightDecay { get; set; } = 0.05;
}
