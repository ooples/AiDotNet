using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Identity link function: g(μ) = μ.
/// </summary>
/// <remarks>
/// <para>
/// The identity link is the canonical link for the Normal (Gaussian) distribution.
/// It makes no transformation, so the linear predictor equals the mean directly.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this for standard linear regression where predictions
/// can be any real number. There's no transformation - what you predict is what you get.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class IdentityLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Identity";

    /// <inheritdoc/>
    public bool IsCanonical => true;

    /// <inheritdoc/>
    public T Link(T mu) => mu;

    /// <inheritdoc/>
    public T InverseLink(T eta) => eta;

    /// <inheritdoc/>
    public T LinkDerivative(T mu) => NumOps.One;

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta) => NumOps.One;

    /// <inheritdoc/>
    public T Variance(T mu) => NumOps.One;
}
