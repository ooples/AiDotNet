using AiDotNet.Enums;
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
public class YOLOv8SegOptions : SegmentationModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public YOLOv8SegOptions()
    {
        NumClasses = 80;
        DropRate = 0;
        ModelSize = YOLOv8SegModelSize.N;
    }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public YOLOv8SegOptions(YOLOv8SegOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        MaxGradNorm = other.MaxGradNorm;
        NumClasses = other.NumClasses;
        DropRate = other.DropRate;
        ModelSize = other.ModelSize;
    }

    /// <summary>
    /// Gets or sets which published size variant of the model to build.
    /// Default: <c>YOLOv8SegModelSize.N</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Most of these models were published in several sizes that
    /// trade accuracy against speed and memory. Picking a variant selects the widths and
    /// depths the paper reports for it; it is not a hint, it changes the network that gets
    /// built.
    /// </para>
    /// <para>
    /// Declared here rather than on <see cref="SegmentationModelOptions"/> because each
    /// model names its own variants with its own enum, so there is no shared type to
    /// declare.
    /// </para>
    /// </remarks>
    public YOLOv8SegModelSize ModelSize { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="SegmentationModelOptions.NumClasses"/> is not positive, or when
    /// <see cref="SegmentationModelOptions.DropRate"/> is not a fraction in [0, 1).
    /// </exception>
    public void Validate() => ValidateSegmentationCore();
}
