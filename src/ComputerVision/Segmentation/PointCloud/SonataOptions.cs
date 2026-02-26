using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Configuration options for Sonata 3D segmentation with self-distillation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Sonata model. Default values follow the original paper settings.</para>
/// </remarks>
public class SonataOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SonataOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SonataOptions(SonataOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
