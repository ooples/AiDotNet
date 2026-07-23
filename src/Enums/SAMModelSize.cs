namespace AiDotNet.Enums;

/// <summary>
/// Defines the backbone size variants for SAM (Segment Anything Model).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM is the original Segment Anything Model from Meta AI. It uses a
/// Vision Transformer (ViT) backbone with varying sizes. Larger backbones produce more accurate
/// masks but are slower. Choose ViTBase for a good balance, or ViTHuge for maximum accuracy.
/// </para>
/// <para>
/// <b>Technical Details:</b> The ViT encoder processes images at 1024x1024 resolution using
/// 16x16 patches. The model was trained on the SA-1B dataset containing 1B+ masks.
/// </para>
/// <para>
/// <b>Reference:</b> Kirillov et al., "Segment Anything", ICCV 2023.
/// </para>
/// </remarks>
public enum SAMModelSize
{
    /// <summary>
    /// ViT-B backbone (91M params). Good balance of speed and quality.
    /// </summary>
    ViTBase,

    /// <summary>
    /// ViT-L backbone (308M params). Higher accuracy, especially for complex scenes.
    /// </summary>
    ViTLarge,

    /// <summary>
    /// ViT-H backbone (636M params). Maximum accuracy — the default SAM configuration.
    /// </summary>
    ViTHuge
}
