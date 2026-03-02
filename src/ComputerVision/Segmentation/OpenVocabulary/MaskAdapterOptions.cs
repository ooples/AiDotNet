using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.OpenVocabulary;

/// <summary>
/// Configuration options for Mask-Adapter open-vocabulary segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the MaskAdapter model. Default values follow the original paper settings.</para>
/// </remarks>
public class MaskAdapterOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public MaskAdapterOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MaskAdapterOptions(MaskAdapterOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
