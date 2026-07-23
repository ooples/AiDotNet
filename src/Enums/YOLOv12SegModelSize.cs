namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for YOLOv12-Seg instance segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLOv12-Seg is the first attention-centric YOLO, matching CNN speed
/// while using attention mechanisms. Achieves 40.6% mAP at 1.64ms.
/// </para>
/// <para>
/// <b>Reference:</b> Sun et al., "YOLOv12: Attention-Centric Real-Time Object Detectors",
/// NeurIPS 2025.
/// </para>
/// </remarks>
public enum YOLOv12SegModelSize
{
    /// <summary>Nano variant. Ultra-fast.</summary>
    N,
    /// <summary>Small variant. Good balance.</summary>
    S,
    /// <summary>Medium variant. Higher accuracy.</summary>
    M,
    /// <summary>Large variant. Best accuracy.</summary>
    L,
    /// <summary>Extra-large variant. Maximum accuracy.</summary>
    X
}
