namespace AiDotNet.Enums;

/// <summary>
/// Defines the size variants for YOLOv8-Seg instance segmentation models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLOv8-Seg is a real-time instance segmentation model from Ultralytics.
/// Smaller sizes (N, S) are faster and suited for edge deployment, while larger sizes (L, X)
/// offer higher accuracy for server-side applications.
/// </para>
/// <para>
/// <b>Technical Details:</b> YOLOv8-Seg uses an anchor-free detection head with a YOLACT-style
/// prototype mask generation branch. The backbone is a CSPDarknet variant with C2f blocks.
/// </para>
/// <para>
/// <b>Reference:</b> Ultralytics, "YOLOv8", 2023.
/// </para>
/// </remarks>
public enum YOLOv8SegModelSize
{
    /// <summary>
    /// Nano (3.4M params). Fastest — for mobile/edge devices.
    /// </summary>
    N,

    /// <summary>
    /// Small (11.8M params). Good speed-accuracy tradeoff.
    /// </summary>
    S,

    /// <summary>
    /// Medium (27.3M params). Balanced for most use cases.
    /// </summary>
    M,

    /// <summary>
    /// Large (46.0M params). High accuracy for server deployment.
    /// </summary>
    L,

    /// <summary>
    /// Extra-Large (71.8M params). Maximum accuracy.
    /// </summary>
    X
}
