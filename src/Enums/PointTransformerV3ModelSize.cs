namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for Point Transformer V3 (3D point cloud segmentation).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024 Oral.
/// </para>
/// </remarks>
public enum PointTransformerV3ModelSize
{
    /// <summary>Base variant. 3x faster, 10x memory-efficient vs V2.</summary>
    Base,
    /// <summary>Large variant. Maximum accuracy for 3D segmentation.</summary>
    Large
}
