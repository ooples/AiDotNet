namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for UniVS (Universal Video Segmentation).
/// </summary>
/// <remarks>
/// <para>
/// <b>Reference:</b> "UniVS: Unified and Universal Video Segmentation with Prompts as Queries",
/// CVPR 2024.
/// </para>
/// </remarks>
public enum UniVSModelSize
{
    /// <summary>ResNet-50 backbone.</summary>
    R50,
    /// <summary>Swin-L backbone. Best accuracy.</summary>
    SwinLarge
}
