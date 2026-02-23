using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Logit link function: g(μ) = log(μ/(1-μ)).
/// </summary>
/// <remarks>
/// <para>
/// The logit link is the canonical link for the Binomial distribution.
/// It maps probabilities from (0,1) to (-∞,∞).
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this for binary classification (logistic regression).
/// It ensures your predictions are valid probabilities between 0 and 1.
///
/// The logit function is the log-odds:
/// - logit(0.5) = 0 (50-50 odds)
/// - logit(0.9) = 2.2 (9:1 odds)
/// - logit(0.1) = -2.2 (1:9 odds)
///
/// The inverse logit (sigmoid) converts back:
/// - sigmoid(0) = 0.5
/// - sigmoid(2.2) ≈ 0.9
/// - sigmoid(-2.2) ≈ 0.1
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LogitLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Logit";

    /// <inheritdoc/>
    public bool IsCanonical => true;

    /// <inheritdoc/>
    public T Link(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));  // Clamp to avoid log(0)
        return NumOps.FromDouble(Math.Log(m / (1 - m)));
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        // Numerically stable sigmoid
        if (e >= 0)
        {
            return NumOps.FromDouble(1 / (1 + Math.Exp(-e)));
        }
        else
        {
            double expE = Math.Exp(e);
            return NumOps.FromDouble(expE / (1 + expE));
        }
    }

    /// <inheritdoc/>
    public T LinkDerivative(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        return NumOps.FromDouble(1 / (m * (1 - m)));
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        T p = InverseLink(eta);
        double pDouble = NumOps.ToDouble(p);
        return NumOps.FromDouble(pDouble * (1 - pDouble));
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        return NumOps.FromDouble(m * (1 - m));
    }
}
