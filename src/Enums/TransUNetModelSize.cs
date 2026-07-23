namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for TransUNet medical segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical
/// Image Segmentation", arXiv 2021.
/// </para>
/// </remarks>
public enum TransUNetModelSize
{
    /// <summary>ViT-B/16 backbone. Standard hybrid model.</summary>
    Base,
    /// <summary>ViT-L/16 backbone. Higher capacity.</summary>
    Large
}
