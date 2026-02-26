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
    }

}
