namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Vision Mamba (Vim) segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with
/// Bidirectional State Space Model", ICML 2024.
/// </para>
/// </remarks>
public enum VisionMambaModelSize
{
    /// <summary>Vim-Ti: width 192, 24 layers. 2.8x faster, 86.8% less GPU memory than DeiT.</summary>
    Tiny,
    /// <summary>Vim-S: width 384, 24 layers, with a 2D rotary position embedding in the official segmentation
    /// configuration.</summary>
    Small
}
