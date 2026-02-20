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
}
