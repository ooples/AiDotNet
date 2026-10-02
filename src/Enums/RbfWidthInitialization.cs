namespace AiDotNet.Enums;

/// <summary>
/// How <see cref="AiDotNet.NeuralNetworks.Layers.RBFLayer{T}"/> sets each unit's initial width.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Each RBF unit responds to inputs near its centre, and its width sets how near.
/// A width that is far too small makes the unit respond to almost nothing, so its output is zero for every input
/// and it never trains. These choices decide how the widths start out before training adjusts them.</para>
/// </remarks>
public enum RbfWidthInitialization
{
    /// <summary>
    /// Every width is d_max / sqrt(2M), where d_max is the largest distance between two initial centres and M is
    /// the number of centres (Broomhead and Lowe 1988; Haykin, Neural Networks, section 5.10). The units then
    /// overlap enough to cover the region the centres span, and none starts saturated. This is the default.
    /// </summary>
    CenterSpread,

    /// <summary>
    /// Each width is drawn from Uniform(0, 1), the layer's original behaviour. A width near 0 makes its unit
    /// output zero for every input, so this is kept only for reproducing older runs.
    /// </summary>
    Uniform,
}