namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for EfficientTAM (Track Anything Model).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EfficientTAM provides SAM 2-comparable video segmentation at 1.6x speedup,
/// running at more than 10 FPS on iPhone 15 with a lightweight ViT encoder.
/// </para>
/// <para>
/// <b>Reference:</b> "EfficientTAM: Efficient Track Anything Model", ICCV 2025.
/// </para>
/// </remarks>
public enum EfficientTAMModelSize
{
    /// <summary>Tiny variant. Mobile-ready, &gt;10 FPS on iPhone 15.</summary>
    Tiny,
    /// <summary>Small variant. Good balance of speed and accuracy.</summary>
    Small
}
