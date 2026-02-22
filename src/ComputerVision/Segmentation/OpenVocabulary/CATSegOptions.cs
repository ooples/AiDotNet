using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for CAT-Seg cost aggregation open-vocabulary segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CATSeg model. Default values follow the original paper settings.</para>
/// </remarks>
public class CATSegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CATSegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CATSegOptions(CATSegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
