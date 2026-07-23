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
    /// <summary>Tiny variant. 2.8x faster, 86.8% less GPU memory than DeiT.</summary>
    Tiny,
    /// <summary>Small variant. Good balance.</summary>
    Small,
    /// <summary>Base variant. Maximum accuracy.</summary>
    Base
}
