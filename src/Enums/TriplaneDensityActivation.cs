namespace AiDotNet.Enums;

/// <summary>
/// The activation a triplane NeRF decoder applies to its raw density output (after the density bias) to obtain a
/// non-negative volume density.
/// </summary>
/// <remarks>
/// <para>
/// TripoSR (Tochilkin et al. 2024) decodes <c>density_act = activation(density + density_bias)</c> with
/// <see cref="Exp"/> and a bias of -1.
/// </para>
/// <para>
/// <b>For Beginners:</b> The network outputs any real number for "how solid is this point"; this activation turns it
/// into a density that is never negative.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new TripoSROptions { DensityActivation = TriplaneDensityActivation.Softplus };
/// </code>
/// </example>
public enum TriplaneDensityActivation
{
    /// <summary>The exponential, TripoSR's default (<c>density_activation: exp</c>).</summary>
    Exp,

    /// <summary>softplus(x) = log(1 + exp(x)).</summary>
    Softplus,

    /// <summary>max(0, x).</summary>
    ReLU
}
