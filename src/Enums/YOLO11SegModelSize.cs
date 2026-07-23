namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for YOLO11-Seg instance segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLO11-Seg offers 25% latency reduction and improved small object detection
/// compared to YOLOv8-Seg while maintaining similar accuracy.
/// </para>
/// <para>
/// <b>Reference:</b> Ultralytics YOLO11, 2024.
/// </para>
/// </remarks>
public enum YOLO11SegModelSize
{
    /// <summary>Nano variant (~2.6M params). Ultra-fast for edge devices.</summary>
    N,
    /// <summary>Small variant (~9.6M params). Good balance.</summary>
    S,
    /// <summary>Medium variant (~20.1M params). Higher accuracy.</summary>
    M,
    /// <summary>Large variant (~43.7M params). Best accuracy.</summary>
    L,
    /// <summary>Extra-large variant (~62.1M params). Maximum accuracy.</summary>
    X
}
