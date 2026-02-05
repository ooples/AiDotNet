using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Inverse (reciprocal) link function: g(μ) = 1/μ.
/// </summary>
/// <remarks>
/// <para>
/// The inverse link is the canonical link for the Gamma distribution.
/// It's useful when the relationship is inversely proportional.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this when doubling a predictor halves the response
/// (inverse relationship). Common in:
/// - Time-to-event data with constant hazard
/// - Some physics/engineering applications
///
/// Note: The inverse link can produce negative predictions, so ensure
/// the linear predictor stays positive for meaningful results.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReciprocalLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Inverse";

    /// <inheritdoc/>
    public bool IsCanonical => true;

    /// <inheritdoc/>
    public T Link(T mu)
    {
        double m = NumOps.ToDouble(mu);
        if (Math.Abs(m) < 1e-10)
        {
            m = Math.Sign(m) * 1e-10;
            if (m == 0) m = 1e-10;
        }
        return NumOps.FromDouble(1 / m);
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        if (Math.Abs(e) < 1e-10)
        {
            e = Math.Sign(e) * 1e-10;
            if (e == 0) e = 1e-10;
        }
        return NumOps.FromDouble(1 / e);
    }

    /// <inheritdoc/>
    public T LinkDerivative(T mu)
    {
        double m = NumOps.ToDouble(mu);
        if (Math.Abs(m) < 1e-10)
        {
            m = Math.Sign(m) * 1e-10;
            if (m == 0) m = 1e-10;
        }
        return NumOps.FromDouble(-1 / (m * m));
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        double e = NumOps.ToDouble(eta);
        if (Math.Abs(e) < 1e-10)
        {
            e = Math.Sign(e) * 1e-10;
            if (e == 0) e = 1e-10;
        }
        return NumOps.FromDouble(-1 / (e * e));
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        // For Gamma: V(μ) = μ²
        double m = NumOps.ToDouble(mu);
        return NumOps.FromDouble(m * m);
    }
}
