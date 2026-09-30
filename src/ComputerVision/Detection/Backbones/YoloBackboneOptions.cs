using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Configuration options for the YOLOv8, YOLOv9 and YOLO11 detection backbones.
/// </summary>
/// <remarks>
/// <para>
/// These values were constructor parameters (<c>size</c>, <c>inChannels</c>), so the backbones
/// advertised no configuration surface and nothing could be set through an options object. The three
/// backbones take the same two knobs - one scale and the input channels - so they share one options
/// type rather than three identical ones.
/// </para>
/// <para><b>For Beginners:</b> A YOLO backbone is the feature extractor a YOLO detector sits on. The
/// same architecture ships in several sizes; <see cref="Size"/> picks one.</para>
/// </remarks>
public class YoloBackboneOptions : DetectionBackboneOptions
{
    /// <summary>
    /// The model scale (Nano, Small, Medium, Large, XLarge). Default: <see cref="ModelSize.Nano"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bigger sizes have more and wider layers: more accurate, slower. Each
    /// YOLO generation maps these sizes to its own published width and depth multipliers.</para>
    /// </remarks>
    public ModelSize Size { get; set; } = ModelSize.Nano;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working backbone.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when <see cref="DetectionBackboneOptions.InChannels"/> is not positive.
    /// </exception>
    public void Validate() => ValidateBackboneCore();
}
