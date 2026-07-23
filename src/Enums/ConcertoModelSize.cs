namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Concerto (joint 2D-3D point cloud segmentation).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> "Concerto: Joint 2D-3D Pretraining for 3D Segmentation", NeurIPS 2025.
/// </para>
/// </remarks>
public enum ConcertoModelSize
{
    /// <summary>Base variant.</summary>
    Base,
    /// <summary>Large variant. Extended Sonata with 2D-3D joint learning.</summary>
    Large
}
