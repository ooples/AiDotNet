using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// Configuration options for FastSAM (CNN-based fast Segment Anything).
/// </summary>
public class FastSAMOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Channel dimensions for each backbone stage. Default: [80, 160, 320, 640] (YOLOv8-Seg).
    /// </summary>
    public int[]? ChannelDims { get; set; }

    /// <summary>
    /// Number of blocks per backbone stage. Default: [3, 6, 6, 3] (YOLOv8-Seg).
    /// </summary>
    public int[]? Depths { get; set; }

    /// <summary>
    /// Decoder hidden dimension. Default: 256.
    /// </summary>
    public int? DecoderDim { get; set; }
}
