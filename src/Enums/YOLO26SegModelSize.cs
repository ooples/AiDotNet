namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for YOLO26-Seg instance segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLO26-Seg is NMS-free with ProgLoss and STAL training, offering
/// 43% faster CPU inference than YOLO11 with end-to-end operation.
/// </para>
/// <para>
/// <b>Reference:</b> Ultralytics YOLO26, 2025.
/// </para>
/// </remarks>
public enum YOLO26SegModelSize
{
    /// <summary>Nano variant. Ultra-fast for edge devices.</summary>
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
