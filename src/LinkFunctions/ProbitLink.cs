using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.LinkFunctions;

/// <summary>
/// Probit link function: g(μ) = Φ⁻¹(μ), where Φ is the standard normal CDF.
/// </summary>
/// <remarks>
/// <para>
/// The probit link maps probabilities to z-scores from the standard normal distribution.
/// It's an alternative to logit for binary classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> Probit is similar to logit but uses the normal distribution
/// instead of the logistic distribution. Key differences:
///
/// - Probit: Based on normal distribution (bell curve tails)
/// - Logit: Based on logistic distribution (slightly heavier tails)
///
/// In practice, results are usually very similar. Probit is sometimes preferred when:
/// - The underlying process is modeled as a latent normal variable
/// - You want consistency with other normal-distribution-based methods
///
/// Interpretation: The coefficient represents the change in z-score for
/// a one-unit change in the predictor.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ProbitLink<T> : ILinkFunction<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "Probit";

    /// <inheritdoc/>
    public bool IsCanonical => false;

    /// <inheritdoc/>
    public T Link(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        return NumOps.FromDouble(NormalInverseCDF(m));
    }

    /// <inheritdoc/>
    public T InverseLink(T eta)
    {
        double e = NumOps.ToDouble(eta);
        return NumOps.FromDouble(NormalCDF(e));
    }

    /// <inheritdoc/>
    public T LinkDerivative(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        double phi = NormalInverseCDF(m);
        double pdf = NormalPDF(phi);
        return NumOps.FromDouble(1 / (pdf + 1e-10));
    }

    /// <inheritdoc/>
    public T InverseLinkDerivative(T eta)
    {
        double e = NumOps.ToDouble(eta);
        return NumOps.FromDouble(NormalPDF(e));
    }

    /// <inheritdoc/>
    public T Variance(T mu)
    {
        double m = NumOps.ToDouble(mu);
        m = Math.Max(1e-10, Math.Min(1 - 1e-10, m));
        return NumOps.FromDouble(m * (1 - m));
    }

    /// <summary>
    /// Standard normal CDF using approximation.
    /// </summary>
    private static double NormalCDF(double x)
    {
        // Abramowitz and Stegun approximation
        double t = 1 / (1 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

        return x > 0 ? 1 - p : p;
    }

    /// <summary>
    /// Standard normal PDF.
    /// </summary>
    private static double NormalPDF(double x)
    {
        return Math.Exp(-x * x / 2) / Math.Sqrt(2 * Math.PI);
    }

    /// <summary>
    /// Inverse normal CDF (quantile function) using rational approximation.
    /// </summary>
    private static double NormalInverseCDF(double p)
    {
        // Beasley-Springer-Moro algorithm
        const double a1 = -39.6968302866538;
        const double a2 = 220.946098424521;
        const double a3 = -275.928510446969;
        const double a4 = 138.357751867269;
        const double a5 = -30.6647980661472;
        const double a6 = 2.50662823884;

        const double b1 = -54.4760987982241;
        const double b2 = 161.585836858041;
        const double b3 = -155.698979859887;
        const double b4 = 66.8013118877197;
        const double b5 = -13.2806815528857;

        const double c1 = -0.00778489400243029;
        const double c2 = -0.322396458041136;
        const double c3 = -2.40075827716184;
        const double c4 = -2.54973253934373;
        const double c5 = 4.37466414146497;
        const double c6 = 2.93816398269878;

        const double d1 = 0.00778469570904146;
        const double d2 = 0.32246712907004;
        const double d3 = 2.445134137143;
        const double d4 = 3.75440866190742;

        double pLow = 0.02425;
        double pHigh = 1 - pLow;

        double q, r;

        if (p < pLow)
        {
            q = Math.Sqrt(-2 * Math.Log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
        }
        else
        {
            q = Math.Sqrt(-2 * Math.Log(1 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                    ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
        }
    }
}
