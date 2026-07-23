namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for YOLOv9-Seg instance segmentation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLOv9-Seg extends YOLOv9 with instance segmentation using Programmable
/// Gradient Information (PGI) and Generalized ELAN (GELAN) architecture.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable
/// Gradient Information", arXiv 2024.
/// </para>
/// </remarks>
public enum YOLOv9SegModelSize
{
    /// <summary>Compact variant. Fast inference.</summary>
    C,
    /// <summary>Extended variant. Best accuracy.</summary>
    E
}
