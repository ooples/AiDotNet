using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// Configuration options for YOLOv12-Seg instance segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the YOLOv12Seg model. Default values follow the original paper settings.</para>
/// </remarks>
public class YOLOv12SegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public YOLOv12SegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YOLOv12SegOptions(YOLOv12SegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
