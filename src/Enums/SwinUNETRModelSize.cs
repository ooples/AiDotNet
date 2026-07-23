namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Swin-UNETR medical segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation
/// of Brain Tumors in MRI Images", CVPR 2022.
/// </para>
/// </remarks>
public enum SwinUNETRModelSize
{
    /// <summary>Tiny Swin backbone.</summary>
    Tiny,
    /// <summary>Small Swin backbone.</summary>
    Small,
    /// <summary>Base Swin backbone. Standard for brain tumor segmentation.</summary>
    Base
}
