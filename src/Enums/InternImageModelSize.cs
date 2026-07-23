namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for InternImage (DCNv3-based CNN backbone).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> InternImage comes in five sizes (Tiny through Huge). It proves that
/// CNNs can match or exceed Vision Transformers when using modern deformable convolutions.
/// Tiny is great for quick experiments, while XL and Huge target maximum accuracy.
/// </para>
/// <para>
/// <b>Technical Details:</b> Each size uses a different DCNv3 backbone configuration with
/// varying channel widths, depths, and group sizes. All variants use the UPerNet decoder.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "InternImage: Exploring Large-Scale Vision Foundation
/// Models with Deformable Convolutions", CVPR 2023.
/// </para>
/// </remarks>
public enum InternImageModelSize
{
    /// <summary>
    /// InternImage-T: Tiny variant (30M params). Fast inference for prototyping.
    /// </summary>
    /// <remarks>
    /// Channel dims: [64, 128, 256, 512], Depths: [4, 4, 18, 4], Groups: [4, 8, 16, 32].
    /// </remarks>
    Tiny,

    /// <summary>
    /// InternImage-S: Small variant (50M params). Good balance of speed and accuracy.
    /// </summary>
    /// <remarks>
    /// Channel dims: [80, 160, 320, 640], Depths: [4, 4, 21, 4], Groups: [5, 10, 20, 40].
    /// </remarks>
    Small,

    /// <summary>
    /// InternImage-B: Base variant (97M params). Strong production baseline.
    /// </summary>
    /// <remarks>
    /// Channel dims: [112, 224, 448, 896], Depths: [4, 4, 21, 4], Groups: [7, 14, 28, 56].
    /// </remarks>
    Base,

    /// <summary>
    /// InternImage-XL: Extra-large variant (335M params). High accuracy for demanding tasks.
    /// </summary>
    /// <remarks>
    /// Channel dims: [192, 384, 768, 1536], Depths: [4, 4, 21, 4], Groups: [12, 24, 48, 96].
    /// </remarks>
    XL,

    /// <summary>
    /// InternImage-H: Huge variant (1.08B params). Maximum accuracy, research-grade.
    /// </summary>
    /// <remarks>
    /// Channel dims: [320, 640, 1280, 2560], Depths: [6, 6, 32, 6], Groups: [20, 40, 80, 160].
    /// </remarks>
    Huge
}
