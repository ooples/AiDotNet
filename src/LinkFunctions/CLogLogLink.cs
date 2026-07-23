using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Complementary log-log link function: g(μ) = log(-log(1-μ)).
/// </summary>
/// <remarks>
/// <para>
/// The complementary log-log link is asymmetric around 0.5, unlike logit and probit.
/// It's useful for modeling probabilities with asymmetric behavior.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this when:
/// - You're modeling the probability of an event that becomes increasingly likely
/// - The probability approaches 1 faster than it approaches 0
/// - Survival analysis with complementary log-log model
///
/// Unlike logit (symmetric), cloglog is asymmetric:
/// - cloglog(0.5) ≈ -0.37 (not 0 like logit)
/// - Approaches 0 slowly from below
/// - Approaches 1 quickly from above
///
/// This is the canonical link for the extreme value (Gumbel) distribution.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CLogLogLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "CLogLog";

    /// <inheritdoc/>
    public bool IsCanonical => false;

    /// <inheritdoc/>
    public T Link(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        return NumOps.FromDouble(Math.Log(-Math.Log(1 - m)));
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        // 1 - exp(-exp(η))
        if (e > 700) return NumOps.FromDouble(1 - 1e-10);
        if (e < -700) return NumOps.FromDouble(1e-10);
        return NumOps.FromDouble(1 - Math.Exp(-Math.Exp(e)));
    }

    /// <inheritdoc/>
    public T LinkDerivative(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        double logTerm = -Math.Log(1 - m);
        return NumOps.FromDouble(1 / ((1 - m) * logTerm));
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        double e = NumOps.ToDouble(eta);
        if (e > 700 || e < -700) return NumOps.FromDouble(0.0);
        double expE = Math.Exp(e);
        return NumOps.FromDouble(expE * Math.Exp(-expE));
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        return NumOps.FromDouble(m * (1 - m));
    }
}
