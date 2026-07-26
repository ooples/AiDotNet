using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Diffusion;

/// <summary>
/// Configuration options for ODISE diffusion-based panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the ODISE model. Default values follow the original paper settings.</para>
/// </remarks>
public class ODISESegmentationOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public ODISESegmentationOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public ODISESegmentationOptions(ODISESegmentationOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = (int[])other.ChannelDimensions.Clone();
        StageDepths = (int[])other.StageDepths.Clone();
        DecoderDimension = other.DecoderDimension;
    }

    /// <summary>
    /// Gets or sets the four diffusion-backbone stage widths. The defaults preserve
    /// ODISE's paper-scale feature hierarchy.
    /// </summary>
    public int[] ChannelDimensions { get; set; } = [320, 640, 1280, 1280];

    /// <summary>Gets or sets the block count in each diffusion-backbone stage.</summary>
    public int[] StageDepths { get; set; } = [2, 2, 2, 2];

    /// <summary>Gets or sets the panoptic decoder width.</summary>
    public int DecoderDimension { get; set; } = 256;

}
