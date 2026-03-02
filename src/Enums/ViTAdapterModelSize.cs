namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for ViT-Adapter.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-Adapter adds spatial prior modules to plain Vision Transformers
/// so they can handle dense prediction tasks like segmentation. The sizes correspond to
/// different ViT backbone sizes (Small, Base, Large).
/// </para>
/// <para>
/// <b>Technical Details:</b> Each size uses a different ViT backbone with adapter modules
/// that inject multi-scale spatial priors. The adapter modules are lightweight and add only
/// a small percentage of extra parameters on top of the base ViT.
/// </para>
/// <para>
/// <b>Reference:</b> Chen et al., "Vision Transformer Adapter for Dense Predictions",
/// ICLR 2023 Spotlight.
/// </para>
/// </remarks>
public enum ViTAdapterModelSize
{
    /// <summary>
    /// ViT-Adapter-S: Small variant based on ViT-Small (48M params).
    /// </summary>
    /// <remarks>
    /// Embed dim: 384, Depths: [2, 2, 2, 2], Heads: [6, 6, 6, 6].
    /// </remarks>
    Small,

    /// <summary>
    /// ViT-Adapter-B: Base variant based on ViT-Base (86M params). Good for production.
    /// </summary>
    /// <remarks>
    /// Embed dim: 768, Depths: [2, 2, 2, 2], Heads: [12, 12, 12, 12].
    /// </remarks>
    Base,

    /// <summary>
    /// ViT-Adapter-L: Large variant based on ViT-Large (304M params). Maximum accuracy.
    /// </summary>
    /// <remarks>
    /// Embed dim: 1024, Depths: [2, 2, 2, 2], Heads: [16, 16, 16, 16].
    /// </remarks>
    Large
}
