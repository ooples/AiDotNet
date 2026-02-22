using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Configuration options for Concerto joint 2D-3D segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the Concerto model. Default values follow the original paper settings.</para>
/// </remarks>
public class ConcertoOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ConcertoOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ConcertoOptions(ConcertoOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
