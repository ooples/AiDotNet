namespace AiDotNet.Enums;

/// <summary>
/// How a deformable convolution normalizes its modulation scalars.
/// </summary>
/// <remarks>
/// <para>
/// DCNv2 (Zhu et al., "Deformable ConvNets v2", CVPR 2019) passes each sample point's modulation
/// scalar through a sigmoid independently, giving every point a weight in (0, 1) with no
/// relationship between them.
/// </para>
/// <para>
/// DCNv3 (Wang et al., "InternImage: Exploring Large-Scale Vision Foundation Models with
/// Deformable Convolutions", CVPR 2023) changes this deliberately: the scalars are normalized with
/// a softmax ACROSS the K sample points, so they sum to one at every location. The paper gives the
/// reason directly -- an unbounded sum of modulation weights makes gradients and feature magnitudes
/// grow with model scale, and normalizing them stabilizes training for large models. Using sigmoid
/// under a DCNv3 configuration is therefore not a small deviation; it removes one of the three
/// changes that define DCNv3.
/// </para>
/// </remarks>
public enum DeformableModulationNormalization
{
    /// <summary>
    /// Choose from the layer's configuration: softmax when it is set up as DCNv3 (a separable
    /// offset projection), sigmoid otherwise.
    /// </summary>
    /// <remarks>
    /// The default, and the paper-faithful choice in both directions. It must stay the zero value:
    /// a checkpoint written before this option existed has no key for it, and the reconstruction
    /// path falls back to the zero member.
    /// </remarks>
    Auto = 0,

    /// <summary>Per-point sigmoid, as in DCNv2.</summary>
    Sigmoid = 1,

    /// <summary>Softmax across the K sample points, as in DCNv3.</summary>
    Softmax = 2,
}
