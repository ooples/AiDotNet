using AiDotNet.Models.Options;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// Configuration options for SegMamba 3D volumetric Mamba segmentation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the SegMamba model. Default values follow the original paper settings.</para>
/// </remarks>
public class SegMambaOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public SegMambaOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public SegMambaOptions(SegMambaOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ChannelDimensions = other.ChannelDimensions.ToArray();
        StageDepths = other.StageDepths.ToArray();
        StateDimension = other.StateDimension;
        InputChannels = other.InputChannels;
    }

    /// <summary>
    /// Gets or sets how many channels each input volume carries. Default is 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A three-dimensional architecture describes its input as <c>[InputDepth, InputHeight,
    /// InputWidth]</c> — three SPATIAL axes and no channel axis at all — so the channel count of a
    /// volume cannot be read off it. One is the right default for the medical scans SegMamba targets:
    /// CT and single-sequence MRI are single-channel. Raise it for multi-modal input, such as the four
    /// co-registered MRI sequences of the BraTS benchmark the paper evaluates on.
    /// </para>
    /// <para><b>For Beginners:</b> A colour photo has three channels — red, green and blue — stacked
    /// at every pixel. A CT scan has one number per voxel, so one channel. This is that count, and it
    /// is separate from the DEPTH of the volume, which is how many slices the scan contains.
    /// </para>
    /// </remarks>
    public int InputChannels { get; set; } = 1;

    /// <summary>
    /// Gets or sets the feature width of each encoder stage.
    /// </summary>
    /// <remarks>The default <c>[48, 96, 192, 384]</c> follows the original SegMamba paper.</remarks>
    public int[] ChannelDimensions { get; set; } = [48, 96, 192, 384];

    /// <summary>
    /// Gets or sets the number of tri-orientated Mamba blocks in each encoder stage.
    /// </summary>
    /// <remarks>The default <c>[2, 2, 2, 2]</c> follows the original SegMamba paper.</remarks>
    public int[] StageDepths { get; set; } = [2, 2, 2, 2];

    /// <summary>
    /// Gets or sets the selective state-space dimension used by every Mamba block.
    /// </summary>
    public int StateDimension { get; set; } = 16;
}
