namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for EoMT (Encoder-only Mask Transformer).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EoMT removes the complex pixel decoder and transformer decoder used
/// by models like Mask2Former, instead placing mask queries directly inside a plain Vision
/// Transformer (ViT/DINOv2). This makes it 4.4x faster than Mask2Former while maintaining
/// competitive accuracy.
/// </para>
/// <para>
/// <b>Technical Details:</b> Uses DINOv2 as the backbone. Queries are inserted at intermediate
/// ViT layers and processed alongside image tokens. No separate decoder needed. Achieves strong
/// results on COCO panoptic, ADE20K semantic, and Cityscapes instance.
/// </para>
/// <para>
/// <b>Reference:</b> Saporta et al., "Encoder-only Mask Transformer", CVPR 2025 Highlight.
/// </para>
/// </remarks>
public enum EoMTModelSize
{
    /// <summary>
    /// ViT-S/DINOv2 backbone (~22M params). Fast and efficient.
    /// </summary>
    Small,

    /// <summary>
    /// ViT-B/DINOv2 backbone (~86M params). Good balance of speed and accuracy.
    /// </summary>
    Base,

    /// <summary>
    /// ViT-L/DINOv2 backbone (~307M params). Best accuracy, 4.4x faster than Mask2Former.
    /// </summary>
    Large
}
