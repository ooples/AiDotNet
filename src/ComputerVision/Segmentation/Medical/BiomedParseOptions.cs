using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for BiomedParse biomedical foundation segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the BiomedParse model. Default values follow the original paper settings.</para>
/// </remarks>
public class BiomedParseOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public BiomedParseOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public BiomedParseOptions(BiomedParseOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
