namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for PIDNet real-time segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> Xu et al., "PIDNet: A Real-time Semantic Segmentation Network Inspired
/// by PID Controllers", CVPR 2023.
/// </para>
/// </remarks>
public enum PIDNetModelSize
{
    /// <summary>Small variant. 78.6% mIoU at 93.2 FPS on Cityscapes.</summary>
    Small,
    /// <summary>Medium variant. Higher accuracy with good speed.</summary>
    Medium,
    /// <summary>Large variant. Best accuracy.</summary>
    Large
}
