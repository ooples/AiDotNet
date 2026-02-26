using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Configuration options for Point Transformer V3 3D segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the PointTransformerV3 model. Default values follow the original paper settings.</para>
/// </remarks>
public class PointTransformerV3Options : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public PointTransformerV3Options() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public PointTransformerV3Options(PointTransformerV3Options other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
