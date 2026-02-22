using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for Grounded SAM 2 text-prompted detection and tracking.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the GroundedSAM2 model. Default values follow the original paper settings.</para>
/// </remarks>
public class GroundedSAM2Options : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public GroundedSAM2Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GroundedSAM2Options(GroundedSAM2Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
