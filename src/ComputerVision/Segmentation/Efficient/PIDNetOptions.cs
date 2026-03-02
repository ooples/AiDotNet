using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// Configuration options for PIDNet real-time segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PIDNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class PIDNetOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public PIDNetOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PIDNetOptions(PIDNetOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
