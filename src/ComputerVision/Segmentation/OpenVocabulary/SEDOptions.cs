using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for SED simple encoder-decoder open-vocabulary segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SED model. Default values follow the original paper settings.</para>
/// </remarks>
public class SEDOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SEDOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SEDOptions(SEDOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
