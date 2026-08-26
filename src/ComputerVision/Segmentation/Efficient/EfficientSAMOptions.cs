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
    /// <value>Defaults to <c>2e-4</c>, following the EfficientSAM training recipe.</value>
    /// <remarks><para><b>For Beginners:</b> This controls the size of each optimizer update.</para></remarks>
    public double LearningRate { get; set; } = 2e-4;

    /// <summary>Gets or sets the AdamW decoupled weight decay used for dense segmentation fine-tuning.</summary>
    /// <value>Defaults to <c>0.05</c>, following the EfficientSAM training recipe.</value>
    /// <remarks><para><b>For Beginners:</b> Weight decay discourages overly large model weights.</para></remarks>
    public double WeightDecay { get; set; } = 0.05;
}
