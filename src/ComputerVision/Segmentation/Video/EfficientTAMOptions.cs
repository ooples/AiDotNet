using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for EfficientTAM lightweight video segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the EfficientTAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class EfficientTAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public EfficientTAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public EfficientTAMOptions(EfficientTAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
    }

    /// <summary>
    /// Gets or sets the AdamW image-encoder learning rate used by the full video training recipe.
    /// </summary>
    /// <value>Defaults to <c>6e-5</c>, following the EfficientTAM full-training recipe.</value>
    /// <remarks><para><b>For Beginners:</b> This controls how quickly the image encoder learns.</para></remarks>
    public double LearningRate { get; set; } = 6e-5;

    /// <summary>Gets or sets the AdamW decoupled weight decay used by the full training recipe.</summary>
    /// <value>Defaults to <c>0.1</c>, following the EfficientTAM full-training recipe.</value>
    /// <remarks><para><b>For Beginners:</b> Weight decay regularizes the video segmentation model.</para></remarks>
    public double WeightDecay { get; set; } = 0.1;
}
