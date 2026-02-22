using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Efficient;

/// <summary>
/// Configuration options for FastSAM (CNN-based fast Segment Anything).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FastSAM model. Default values follow the original paper settings.</para>
/// </remarks>
public class FastSAMOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public FastSAMOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public FastSAMOptions(FastSAMOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDims = other.ChannelDims;
        Depths = other.Depths;
        DecoderDim = other.DecoderDim;
    }

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
