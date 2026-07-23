using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// Configuration options for kMaX-DeepLab panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the KMaXDeepLab model. Default values follow the original paper settings.</para>
/// </remarks>
public class KMaXDeepLabOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public KMaXDeepLabOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public KMaXDeepLabOptions(KMaXDeepLabOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
