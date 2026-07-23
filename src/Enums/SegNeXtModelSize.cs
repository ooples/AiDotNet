namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for SegNeXt (Multi-Scale Convolutional Attention backbone).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegNeXt comes in four sizes (Tiny through Large). Smaller sizes (Tiny)
/// are faster and use less memory, while larger sizes (Large) are more accurate but require
/// more compute. Tiny is great for real-time applications, while Base and Large offer
/// excellent accuracy for production deployments.
/// </para>
/// <para>
/// <b>Technical Details:</b> Each size uses a different MSCAN (Multi-Scale Convolutional Attention
/// Network) backbone with varying channel widths and encoder depths. All variants use the
/// Hamburger decoder for semantic segmentation.
/// </para>
/// <para>
/// <b>Reference:</b> Guo et al., "SegNeXt: Rethinking Convolutional Attention Design for
/// Semantic Segmentation", NeurIPS 2022.
/// </para>
/// </remarks>
public enum SegNeXtModelSize
{
    /// <summary>
    /// MSCAN-T: Tiny variant (4.3M params). Fastest inference, suitable for edge devices.
    /// </summary>
    /// <remarks>
    /// Channel dims: [32, 64, 160, 256], Depths: [3, 3, 5, 2], Hamburger decoder dim: 256.
    /// </remarks>
    Tiny,

    /// <summary>
    /// MSCAN-S: Small variant (13.9M params). Good speed/accuracy trade-off.
    /// </summary>
    /// <remarks>
    /// Channel dims: [64, 128, 320, 512], Depths: [2, 2, 4, 2], Hamburger decoder dim: 256.
    /// </remarks>
    Small,

    /// <summary>
    /// MSCAN-B: Base variant (27.6M params). Balanced for production use.
    /// </summary>
    /// <remarks>
    /// Channel dims: [64, 128, 320, 512], Depths: [3, 3, 12, 3], Hamburger decoder dim: 512.
    /// </remarks>
    Base,

    /// <summary>
    /// MSCAN-L: Large variant (48.9M params). Maximum accuracy.
    /// </summary>
    /// <remarks>
    /// Channel dims: [64, 128, 320, 512], Depths: [3, 5, 27, 3], Hamburger decoder dim: 1024.
    /// </remarks>
    Large
}
