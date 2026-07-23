using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for UniVS universal video segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UniVS model. Default values follow the original paper settings.</para>
/// </remarks>
public class UniVSOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UniVSOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UniVSOptions(UniVSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
