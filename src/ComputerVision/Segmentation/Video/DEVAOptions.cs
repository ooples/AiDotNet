using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Video;

/// <summary>
/// Configuration options for DEVA decoupled video segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the DEVA model. Default values follow the original paper settings.</para>
/// </remarks>
public class DEVAOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public DEVAOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DEVAOptions(DEVAOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
