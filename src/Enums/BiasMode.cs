namespace AiDotNet.Enums;

/// <summary>
/// Whether a layer that can carry an additive bias should actually do so.
/// </summary>
/// <remarks>
/// <para>
/// A bias is redundant when the layer's output is immediately normalized by something that applies
/// its own learnable shift. The normalization subtracts the mean, which removes whatever constant
/// the bias added, then adds its own beta in its place. Two parameters then describe one degree of
/// freedom: the bias cannot receive a meaningful gradient, and it costs memory, checkpoint size and
/// optimizer state for nothing.
/// </para>
/// <para>
/// Reference implementations usually hard-code this decision against the norm type they happen to
/// use. pix2pix (Isola et al., 2017) writes <c>use_bias = norm_layer == nn.InstanceNorm2d</c>, which
/// is right only because their InstanceNorm is built with <c>affine=False</c> and their BatchNorm
/// with <c>affine=True</c>. The same expression is wrong for <c>InstanceNorm2d(affine=True)</c>,
/// which does supply a shift, and wrong for a scale-only normalization such as RMSNorm, which does
/// not. <see cref="Auto"/> asks the following layer instead of inferring from its type, so it stays
/// correct for normalizations those implementations never considered.
/// </para>
/// <para><b>For Beginners:</b> a bias lets a layer shift its output up or down. If the next step is
/// a normalization that already shifts, the bias has nothing left to do. Leave this at
/// <see cref="Auto"/> and the layer works it out; set it explicitly only to force the issue.
/// </para>
/// </remarks>
public enum BiasMode
{
    /// <summary>
    /// Decide from what follows: no bias when the next layer supplies a learnable shift, a bias
    /// otherwise.
    /// </summary>
    /// <remarks>
    /// The default, and what a layer assumes when nothing tells it what comes next -- a standalone
    /// convolution keeps its bias, matching <c>nn.Conv2d(bias=True)</c>.
    /// </remarks>
    Auto = 0,

    /// <summary>Always carry a bias, even under a normalization that already shifts.</summary>
    Always = 1,

    /// <summary>
    /// Never carry a bias. The layer reports no bias parameters at all -- absent from the parameter
    /// count, gradients and checkpoints, not merely frozen at zero.
    /// </summary>
    Never = 2,
}
