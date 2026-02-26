using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for UniverSeg few-shot medical segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the UniverSeg model. Default values follow the original paper settings.</para>
/// </remarks>
public class UniverSegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public UniverSegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public UniverSegOptions(UniverSegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
