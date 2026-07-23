using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// Configuration options for YOLO26-Seg instance segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the YOLO26Seg model. Default values follow the original paper settings.</para>
/// </remarks>
public class YOLO26SegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public YOLO26SegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YOLO26SegOptions(YOLO26SegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
