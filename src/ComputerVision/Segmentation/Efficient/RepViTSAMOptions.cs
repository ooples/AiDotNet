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
    }

}
