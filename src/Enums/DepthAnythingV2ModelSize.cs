namespace AiDotNet.Enums;

/// <summary>
/// Defines the encoder size variants for Depth Anything V2.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Depth Anything V2 estimates how far away each pixel in an image is.
/// It comes in three sizes built on the same design: a bigger encoder gives more accurate
/// depth but takes longer to run. Start with <see cref="Base"/>.
/// </para>
/// <para>
/// Each size selects the DINOv2 encoder the paper pairs with it, which fixes both the feature
/// width and the number of transformer blocks.
/// </para>
/// <para>
/// <b>Reference:</b> "Depth Anything V2", NeurIPS 2024.
/// </para>
/// </remarks>
public enum DepthAnythingV2ModelSize
{
    /// <summary>Small variant, on ViT-S: 384 features across 12 blocks. Fastest, least accurate.</summary>
    Small,

    /// <summary>Base variant, on ViT-B: 768 features across 12 blocks. The balanced default.</summary>
    Base,

    /// <summary>Large variant, on ViT-L: 1024 features across 24 blocks. Slowest, most accurate.</summary>
    Large
}
