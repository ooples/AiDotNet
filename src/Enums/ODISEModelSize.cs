namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for ODISE (Open-vocabulary DIffusion-based panoptic SEgmentation).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ODISE uses text-to-image diffusion features fused with discriminative
/// features for open-vocabulary panoptic segmentation.
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "Open-Vocabulary Panoptic Segmentation with Text-to-Image
/// Diffusion Models", CVPR 2023 Highlight.
/// </para>
/// </remarks>
public enum ODISEModelSize
{
    /// <summary>Base variant with Stable Diffusion backbone.</summary>
    Base,
    /// <summary>Large variant with larger CLIP and diffusion models.</summary>
    Large
}
