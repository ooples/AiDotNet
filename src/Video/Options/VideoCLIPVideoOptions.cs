using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the VideoCLIP video understanding model.
/// </summary>
public class VideoCLIPVideoOptions : NeuralNetworkOptions
{
    /// <summary>Gets or sets the learning rate used by the default AdamW optimizer.</summary>
    /// <value>Defaults to 0.0001.</value>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the hidden width of the video and text encoders.</summary>
    /// <value>Defaults to 768, matching the paper-scale configuration.</value>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Gets or sets the number of spatial video encoder blocks.</summary>
    /// <value>Defaults to 12.</value>
    public int NumSpatialBlocks { get; set; } = 12;

    /// <summary>Gets or sets the number of temporal video encoder blocks.</summary>
    /// <value>Defaults to 4.</value>
    public int NumTemporalBlocks { get; set; } = 4;

    /// <summary>Gets or sets the number of text transformer blocks.</summary>
    /// <value>Defaults to 12.</value>
    public int NumTextBlocks { get; set; } = 12;
}
