namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for SAM-HQ (High-Quality Segment Anything Model).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM-HQ extends the Segment Anything Model (SAM) with a High-Quality
/// output token that produces significantly sharper and more accurate mask boundaries. It uses
/// the same ViT backbone sizes as the original SAM.
/// </para>
/// <para>
/// <b>Technical Details:</b> SAM-HQ adds an HQ output token and learnable global-local feature
/// fusion to the original SAM architecture. The backbone is a Vision Transformer (ViT) with
/// varying sizes. Training uses only 44K fine-grained masks.
/// </para>
/// <para>
/// <b>Reference:</b> Ke et al., "Segment Anything in High Quality", NeurIPS 2023.
/// </para>
/// </remarks>
public enum SAMHQModelSize
{
    /// <summary>
    /// ViT-B backbone (91M params). Good balance of speed and quality.
    /// </summary>
    ViTBase,

    /// <summary>
    /// ViT-L backbone (308M params). Higher accuracy for complex boundaries.
    /// </summary>
    ViTLarge,

    /// <summary>
    /// ViT-H backbone (636M params). Maximum accuracy for the finest boundaries.
    /// </summary>
    ViTHuge
}
