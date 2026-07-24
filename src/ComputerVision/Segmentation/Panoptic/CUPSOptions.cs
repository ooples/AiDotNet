using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Panoptic;

/// <summary>
/// Configuration options for CUPS unsupervised panoptic segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the CUPS model. Default values follow the original paper settings.</para>
/// </remarks>
public class CUPSOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public CUPSOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public CUPSOptions(CUPSOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = (int[])other.ChannelDimensions.Clone();
        StageDepths = (int[])other.StageDepths.Clone();
        DecoderDimension = other.DecoderDimension;
    }

    /// <summary>Feature widths for the four hierarchical encoder stages.</summary>
    public int[] ChannelDimensions { get; set; } = [96, 192, 384, 768];

    /// <summary>Block counts for the four hierarchical encoder stages.</summary>
    public int[] StageDepths { get; set; } = [2, 2, 6, 2];

    /// <summary>Feature width used by the panoptic decoder.</summary>
    public int DecoderDimension { get; set; } = 256;
}
