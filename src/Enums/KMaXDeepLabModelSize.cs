namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for kMaX-DeepLab panoptic segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> kMaX-DeepLab treats cross-attention as k-means clustering, achieving
/// 58.0 PQ on COCO panoptic with a clean and efficient architecture.
/// </para>
/// <para>
/// <b>Reference:</b> Yu et al., "k-means Mask Transformer", CVPR 2023.
/// </para>
/// </remarks>
public enum KMaXDeepLabModelSize
{
    /// <summary>ResNet-50 backbone. Efficient baseline.</summary>
    R50,
    /// <summary>ConvNeXt-L backbone. Best accuracy.</summary>
    ConvNeXtLarge
}
