using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// Configuration options for CUPS unsupervised panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CUPS model. Default values follow the original paper settings.</para>
/// </remarks>
public class CUPSOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CUPSOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CUPSOptions(CUPSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
