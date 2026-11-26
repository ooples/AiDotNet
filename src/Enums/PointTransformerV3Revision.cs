namespace AiDotNet.Enums;

/// <summary>
/// The two published Point Transformer V3 backbones in Pointcept.
/// </summary>
/// <remarks>
/// <para>
/// Both share serialization, serialized patch attention, xCPE and the block layout. They differ in
/// the parts around the blocks:
/// </para>
/// <list type="bullet">
/// <item><see cref="M1Base"/> (<c>PT-v3m1</c>, the CVPR 2024 paper): a 5x5x5 submanifold
/// convolution stem, BatchNorm in the stem, pooling and unpooling, pooling by serialized code
/// (<c>code &gt;&gt; 3</c>), PyTorch default initialisation.</item>
/// <item><see cref="M2Sonata"/> (<c>PT-v3m2</c>, the backbone Sonata pretrains): a linear stem,
/// LayerNorm everywhere, grid pooling followed by re-serialization, an optional mask token and
/// <c>trunc_normal_(0.02)</c> initialisation.</item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Pick <see cref="M1Base"/> to reproduce the PTv3 paper and
/// <see cref="M2Sonata"/> to reproduce the Sonata paper's encoder.
/// </para>
/// </remarks>
public enum PointTransformerV3Revision
{
    /// <summary>Pointcept <c>PT-v3m1</c>, the backbone of the PTv3 paper.</summary>
    M1Base,

    /// <summary>Pointcept <c>PT-v3m2</c>, the backbone of the Sonata paper.</summary>
    M2Sonata
}
