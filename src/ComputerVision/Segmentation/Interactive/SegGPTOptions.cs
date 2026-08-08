using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Interactive;

/// <summary>
/// Configuration options for SegGPT in-context segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SegGPT model. Default values follow the original paper settings.</para>
/// </remarks>
public class SegGPTOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SegGPTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SegGPTOptions(SegGPTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = other.ChannelDimensions.ToArray();
        StageDepths = other.StageDepths.ToArray();
        DecoderDimension = other.DecoderDimension;
    }

    /// <summary>
    /// Gets or sets the four hierarchical encoder widths. The defaults reproduce
    /// the ViT-Large configuration used by the original SegGPT implementation.
    /// </summary>
    public int[] ChannelDimensions { get; set; } = [64, 128, 320, 1024];

    /// <summary>
    /// Gets or sets the number of blocks in each encoder stage.
    /// </summary>
    public int[] StageDepths { get; set; } = [2, 2, 4, 24];

    /// <summary>
    /// Gets or sets the mask decoder width.
    /// </summary>
    public int DecoderDimension { get; set; } = 256;
}
