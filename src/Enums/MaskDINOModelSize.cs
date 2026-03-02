namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Mask DINO.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask DINO unifies object detection and segmentation by extending the
/// DINO detector with a mask prediction branch. It handles instance, panoptic, and semantic
/// segmentation in one architecture.
/// </para>
/// <para>
/// <b>Technical Details:</b> Built on DINO (DETR with Improved deNoising anchOr boxes) with
/// an additional mask branch. Uses ResNet or Swin Transformer backbones with deformable
/// attention in the transformer encoder-decoder.
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "Mask DINO: Towards A Unified Transformer-based Framework
/// for Object Detection and Segmentation", CVPR 2023.
/// </para>
/// </remarks>
public enum MaskDINOModelSize
{
    /// <summary>
    /// ResNet-50 backbone (44M params). Fast baseline.
    /// </summary>
    R50,

    /// <summary>
    /// Swin-L backbone (218M params). Best accuracy with Swin Transformer.
    /// </summary>
    SwinLarge
}
