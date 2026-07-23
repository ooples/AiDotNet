using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Referring;

/// <summary>
/// Configuration options for LISA reasoning segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the LISA model. Default values follow the original paper settings.</para>
/// </remarks>
public class LISAOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public LISAOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public LISAOptions(LISAOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
