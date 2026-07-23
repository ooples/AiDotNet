using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Log link function: g(μ) = log(μ).
/// </summary>
/// <remarks>
/// <para>
/// The log link is the canonical link for the Poisson distribution and is also
/// commonly used with Gamma and other positive-valued distributions.
/// It maps positive values to (-∞,∞).
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this when your response variable is always positive:
/// - Count data (number of events)
/// - Positive continuous values (income, time, distance)
///
/// The log link ensures predictions are always positive (after inverse transform).
/// A one-unit increase in a predictor multiplies the expected response by exp(β).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LogLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Log";

    /// <inheritdoc/>
    public bool IsCanonical => true;

    /// <inheritdoc/>
    public T Link(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, m);  // Avoid log(0)
        return NumOps.FromDouble(Math.Log(m));
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        // Clamp to avoid overflow
        e = Math.Min(700, e);  // exp(700) is close to double max
        return NumOps.FromDouble(Math.Exp(e));
    }

    /// <inheritdoc/>
    public T LinkDerivative(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, m);
        return NumOps.FromDouble(1 / m);
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        return InverseLink(eta);  // d/dη exp(η) = exp(η)
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        // For Poisson: V(μ) = μ
        return mu;
    }
}
