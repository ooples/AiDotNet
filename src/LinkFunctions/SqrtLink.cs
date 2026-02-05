using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Square root link function: g(μ) = √μ.
/// </summary>
/// <remarks>
/// <para>
/// The square root link is often used with Poisson count data as an alternative
/// to the log link. It provides variance stabilization.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this when:
/// - You have count data but want to moderate extreme predictions
/// - The log link produces predictions that are too extreme
///
/// The square root link is gentler than the log link:
/// - log(100) = 4.6, so exp(5) = 148
/// - sqrt(100) = 10, so 10² = 100
///
/// This means changes in the linear predictor have a more moderate
/// effect on predictions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SqrtLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Sqrt";

    /// <inheritdoc/>
    public bool IsCanonical => false;

    /// <inheritdoc/>
    public T Link(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(0, m);
        return NumOps.FromDouble(Math.Sqrt(m));
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        // Square the linear predictor (allow negative η for flexibility)
        return NumOps.FromDouble(e * e);
    }

    /// <inheritdoc/>
    public T LinkDerivative(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, m);
        return NumOps.FromDouble(1 / (2 * Math.Sqrt(m)));
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        double e = NumOps.ToDouble(eta);
        return NumOps.FromDouble(2 * e);
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        // For Poisson-like: V(μ) = μ
        return mu;
    }
}
