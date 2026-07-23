namespace AiDotNet.Enums;

/// <summary>
/// Defines the model size variants for ViT-CoMer (hybrid CNN-Transformer).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ViT-CoMer combines a CNN branch with a Vision Transformer branch
/// to get the best of both worlds — CNNs excel at fine local details while transformers
/// capture global context. The result is better boundary quality in segmentation.
/// </para>
/// <para>
/// <b>Technical Details:</b> Each size uses a different ViT backbone paired with a CNN branch
/// of matching capacity. The two branches exchange information through multi-scale feature
/// interaction modules.
/// </para>
/// <para>
/// <b>Reference:</b> Xia et al., "ViT-CoMer: Vision Transformer with Convolutional
/// Multi-scale Feature Interaction for Dense Predictions", CVPR 2024.
/// </para>
/// </remarks>
public enum ViTCoMerModelSize
{
    /// <summary>
    /// ViT-CoMer-S: Small variant (45M params). Efficient with good accuracy.
    /// </summary>
    /// <remarks>
    /// Embed dim: 384, CNN channels: [64, 128, 320, 512], Depths: [2, 2, 6, 2].
    /// </remarks>
    Small,

    /// <summary>
    /// ViT-CoMer-B: Base variant (100M params). Strong production baseline.
    /// </summary>
    /// <remarks>
    /// Embed dim: 768, CNN channels: [64, 128, 320, 512], Depths: [2, 2, 6, 2].
    /// </remarks>
    Base,

    /// <summary>
    /// ViT-CoMer-L: Large variant (350M params). Maximum accuracy with hybrid features.
    /// </summary>
    /// <remarks>
    /// Embed dim: 1024, CNN channels: [96, 192, 384, 768], Depths: [2, 2, 6, 2].
    /// </remarks>
    Large
}
