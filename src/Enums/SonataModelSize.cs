namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Sonata (3D point cloud segmentation with self-distillation).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Wu et al., "Sonata: Self-Supervised Learning of Reliable Point Representations",
/// CVPR 2025 (arXiv:2503.16429).
/// </para>
/// </remarks>
public enum SonataModelSize
{
    /// <summary>Base variant.</summary>
    Base,
    /// <summary>Large variant. SOTA indoor/outdoor segmentation.</summary>
    Large
}
