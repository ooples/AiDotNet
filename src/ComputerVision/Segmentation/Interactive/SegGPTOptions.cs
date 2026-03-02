using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// Configuration options for SegGPT in-context segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SegGPT model. Default values follow the original paper settings.</para>
/// </remarks>
public class SegGPTOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SegGPTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SegGPTOptions(SegGPTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
