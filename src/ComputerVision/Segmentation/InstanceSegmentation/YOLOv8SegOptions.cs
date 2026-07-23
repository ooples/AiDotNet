using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// Configuration options for the YOLOv8-Seg instance segmentation model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLOv8-Seg is a real-time instance segmentation model from Ultralytics.
/// Options inherit from NeuralNetworkOptions and can be extended with YOLO-specific settings.
/// </para>
/// </remarks>
public class YOLOv8SegOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public YOLOv8SegOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YOLOv8SegOptions(YOLOv8SegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
    }

}
