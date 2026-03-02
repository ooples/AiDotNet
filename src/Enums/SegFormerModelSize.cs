namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for SegFormer (Mix Transformer backbone).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegFormer comes in six sizes (B0 through B5). Smaller sizes (B0)
/// are faster and use less memory, while larger sizes (B5) are more accurate but require
/// more compute. B0 is a great starting point for experimentation, while B2-B3 offer
/// a good balance for production use.
/// </para>
/// <para>
/// <b>Technical Details:</b> Each size uses a different Mix Transformer (MiT) backbone
/// with varying embedding dimensions, transformer depths, and attention heads.
/// </para>
/// <para>
/// <b>Reference:</b> Xie et al., "SegFormer: Simple and Efficient Design for Semantic
/// Segmentation with Transformers", NeurIPS 2021.
/// </para>
/// </remarks>
public enum SegFormerModelSize
{
    /// <summary>
    /// MiT-B0: Smallest variant (3.8M params). Fastest inference, lowest memory.
    /// </summary>
    /// <remarks>
    /// Embed dims: [32, 64, 160, 256], Depths: [2, 2, 2, 2], Heads: [1, 2, 5, 8].
    /// </remarks>
    B0,

    /// <summary>
    /// MiT-B1: Small variant (13.7M params). Good speed/accuracy trade-off.
    /// </summary>
    /// <remarks>
    /// Embed dims: [64, 128, 320, 512], Depths: [2, 2, 2, 2], Heads: [1, 2, 5, 8].
    /// </remarks>
    B1,

    /// <summary>
    /// MiT-B2: Medium variant (25.4M params). Balanced for production use.
    /// </summary>
    /// <remarks>
    /// Embed dims: [64, 128, 320, 512], Depths: [3, 4, 6, 3], Heads: [1, 2, 5, 8].
    /// </remarks>
    B2,

    /// <summary>
    /// MiT-B3: Large variant (45.2M params). Higher accuracy.
    /// </summary>
    /// <remarks>
    /// Embed dims: [64, 128, 320, 512], Depths: [3, 4, 18, 3], Heads: [1, 2, 5, 8].
    /// </remarks>
    B3,

    /// <summary>
    /// MiT-B4: Extra-large variant (62.0M params). High accuracy for demanding tasks.
    /// </summary>
    /// <remarks>
    /// Embed dims: [64, 128, 320, 512], Depths: [3, 8, 27, 3], Heads: [1, 2, 5, 8].
    /// </remarks>
    B4,

    /// <summary>
    /// MiT-B5: Largest variant (82.0M params). Maximum accuracy.
    /// </summary>
    /// <remarks>
    /// Embed dims: [64, 128, 320, 512], Depths: [3, 6, 40, 3], Heads: [1, 2, 5, 8].
    /// </remarks>
    B5
}
