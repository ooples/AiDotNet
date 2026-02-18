namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Mask2Former.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask2Former is a universal segmentation model that handles semantic,
/// instance, and panoptic segmentation with a single architecture. The backbone size controls
/// feature extraction capacity. Swin-T is efficient for prototyping, while Swin-L offers
/// the highest accuracy.
/// </para>
/// <para>
/// <b>Technical Details:</b> Mask2Former uses a Swin Transformer or ResNet backbone with a
/// pixel decoder (Multi-Scale Deformable Attention) and a transformer decoder with masked
/// cross-attention for query-based mask prediction.
/// </para>
/// <para>
/// <b>Reference:</b> Cheng et al., "Masked-attention Mask Transformer for Universal Image
/// Segmentation", CVPR 2022.
/// </para>
/// </remarks>
public enum Mask2FormerModelSize
{
    /// <summary>
    /// ResNet-50 backbone (44M params). Fast, established CNN baseline.
    /// </summary>
    R50,

    /// <summary>
    /// ResNet-101 backbone (63M params). Deeper CNN baseline.
    /// </summary>
    R101,

    /// <summary>
    /// Swin-T backbone (47M params). Efficient transformer with hierarchical features.
    /// </summary>
    SwinTiny,

    /// <summary>
    /// Swin-S backbone (69M params). Good speed-accuracy trade-off.
    /// </summary>
    SwinSmall,

    /// <summary>
    /// Swin-B backbone (107M params). Strong production baseline.
    /// </summary>
    SwinBase,

    /// <summary>
    /// Swin-L backbone (216M params). Maximum accuracy (57.8 PQ on COCO panoptic).
    /// </summary>
    SwinLarge
}
