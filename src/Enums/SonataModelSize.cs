namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Sonata (3D point cloud segmentation with self-distillation).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> "Sonata: Self-supervised Distillation for 3D Scene Understanding",
/// CVPR 2025 Highlight.
/// </para>
/// </remarks>
public enum SonataModelSize
{
    /// <summary>Base variant.</summary>
    Base,
    /// <summary>Large variant. SOTA indoor/outdoor segmentation.</summary>
    Large
}
