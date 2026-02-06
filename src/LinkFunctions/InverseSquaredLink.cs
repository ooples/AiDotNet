using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Inverse squared link function: g(mu) = 1/mu^2.
/// </summary>
/// <remarks>
/// <para>
/// The inverse squared link is the canonical link for the Inverse Gaussian distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b> This link function maps the mean to 1/mu^2.
/// It is used primarily with Inverse Gaussian GLMs in insurance and reliability modeling.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InverseSquaredLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "InverseSquared";

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
        return NumOps.FromDouble(1.0 / (m * m));
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        if (e <= 0) e = 1e-10;
        return NumOps.FromDouble(1.0 / Math.Sqrt(e));
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
        return NumOps.FromDouble(-2.0 / (m * m * m));
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        double e = NumOps.ToDouble(eta);
        if (e <= 0) e = 1e-10;
        return NumOps.FromDouble(-0.5 / (e * Math.Sqrt(e)));
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        // For Inverse Gaussian: V(mu) = mu^3
        double m = NumOps.ToDouble(mu);
        return NumOps.FromDouble(m * m * m);
    }
}
