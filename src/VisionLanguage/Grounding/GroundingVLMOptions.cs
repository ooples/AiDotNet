using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Base configuration options for visual grounding models.
/// </summary>
public class GroundingVLMOptions : GenerativeVLMOptions
{
    /// <summary>Gets or sets the maximum number of detections per image.</summary>
    public int MaxDetections { get; set; } = 300;

    /// <summary>Gets or sets the confidence threshold for detection filtering.</summary>
    public double ConfidenceThreshold { get; set; } = 0.25;

    /// <summary>Gets or sets the IoU threshold for non-maximum suppression.</summary>
    public double NmsThreshold { get; set; } = 0.5;

    /// <summary>Gets or sets the number of output coordinates per box (typically 4 for [x1,y1,x2,y2]).</summary>
    public int BoxDimension { get; set; } = 4;
}
