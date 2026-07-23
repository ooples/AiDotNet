using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// López de Prado's Deflated Sharpe Ratio (DSR): the probability that a strategy's <i>true</i> Sharpe
/// ratio is positive, after deflating the observed Sharpe for (a) the number of strategy configurations
/// tried (multiple-testing / selection bias) and (b) non-normality (skew and excess kurtosis) of returns.
/// </summary>
/// <remarks>
/// <para>
/// When many strategies are backtested, the best observed Sharpe is upward-biased simply by chance. The
/// DSR compares the observed Sharpe against the <i>expected maximum</i> Sharpe that <c>N</c> independent
/// trials of a truly skill-less process would produce, and expresses the result as a probability in
/// [0, 1] using the Sharpe-ratio estimator's standard error (which itself depends on skew and kurtosis).
/// A DSR above ~0.95 is the usual bar for declaring a discovery genuine.
/// </para>
/// <para><b>For Beginners:</b> If you try 1,000 random trading rules, one of them will look great purely
/// by luck. The DSR corrects for that: it asks "given that I tested N strategies, how surprising is this
/// Sharpe ratio, really?" and answers with a probability that the strategy's edge is real (true Sharpe
/// &gt; 0). It also accounts for the fact that real returns have fat tails and asymmetry, which make a
/// raw Sharpe ratio less trustworthy. Higher DSR = more confidence the result is not a fluke.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public static class DeflatedSharpeRatio<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Euler-Mascheroni constant and a high-quantile helper constant for the expected-max formula.
    private const double EulerMascheroni = 0.5772156649015329;

    /// <summary>
    /// Expected maximum Sharpe ratio (per-observation, i.e. non-annualized) achievable by <paramref name="nTrials"/>
    /// independent trials of a skill-less process whose Sharpe estimates have unit variance. This is the
    /// deflation benchmark: E[max] ≈ (1-γ)·Z⁻¹(1 - 1/N) + γ·Z⁻¹(1 - 1/(N·e)), with γ the Euler-Mascheroni
    /// constant and Z⁻¹ the standard-normal quantile.
    /// </summary>
    /// <param name="nTrials">Number of independent strategy configurations tested (N ≥ 1).</param>
    public static T ExpectedMaxSharpe(int nTrials)
    {
        return NumOps.FromDouble(ExpectedMaxSharpeDouble(nTrials));
    }

    private static double ExpectedMaxSharpeDouble(int nTrials)
    {
        if (nTrials <= 1)
        {
            return 0.0;
        }

        double n = nTrials;
        double q1 = NormalQuantile(1.0 - 1.0 / n);
        double q2 = NormalQuantile(1.0 - 1.0 / (n * Math.E));
        return (1.0 - EulerMascheroni) * q1 + EulerMascheroni * q2;
    }

    /// <summary>
    /// Computes the Deflated Sharpe Ratio.
    /// </summary>
    /// <param name="observedSharpe">
    /// The observed (per-observation, non-annualized) Sharpe ratio of the selected strategy. If you have an
    /// annualized Sharpe, divide by sqrt(periodsPerYear) before passing it here.
    /// </param>
    /// <param name="nObservations">Number of return observations the Sharpe was estimated from (≥ 2).</param>
    /// <param name="nTrials">Number of strategy configurations tested (multiple-testing breadth, N ≥ 1).</param>
    /// <param name="skew">Skewness of the strategy returns (0 for normal).</param>
    /// <param name="kurtosis">
    /// Kurtosis of the strategy returns. Pass the full (non-excess) kurtosis; 3.0 corresponds to normal.
    /// </param>
    /// <returns>Probability in [0, 1] that the true Sharpe exceeds the deflation benchmark.</returns>
    public static T Compute(
        double observedSharpe,
        int nObservations,
        int nTrials,
        double skew = 0.0,
        double kurtosis = 3.0)
    {
        if (nObservations < 2)
        {
            return NumOps.Zero;
        }

        double sharpeBenchmark = ExpectedMaxSharpeDouble(nTrials);

        // Standard error of the Sharpe-ratio estimator under non-normal returns (Mertens / Lo):
        // SE(SR) = sqrt( (1 - skew·SR + (kurtosis-1)/4 · SR^2) / (n - 1) ).
        double sr = observedSharpe;
        double variance = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr;
        if (variance < 0.0)
        {
            variance = 0.0;
        }

        double se = Math.Sqrt(variance / (nObservations - 1));
        if (se <= 0.0)
        {
            // Degenerate estimator variance: DSR is a step function at the benchmark.
            return sr > sharpeBenchmark ? NumOps.One : NumOps.Zero;
        }

        double z = (sr - sharpeBenchmark) / se;
        double dsr = NormalCdf(z);
        // Guard the [0,1] range against floating-point drift (net471-safe).
        dsr = Math.Max(0.0, Math.Min(1.0, dsr));
        return NumOps.FromDouble(dsr);
    }

    // ---- standard-normal CDF and inverse CDF (double precision) ----

    /// <summary>Standard normal CDF via the erf relation, with a high-accuracy erf approximation.</summary>
    private static double NormalCdf(double x)
    {
        return 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
    }

    /// <summary>Abramowitz &amp; Stegun 7.1.26 erf approximation (|error| &lt; 1.5e-7).</summary>
    private static double Erf(double x)
    {
        double sign = x >= 0.0 ? 1.0 : -1.0;
        double ax = Math.Abs(x);

        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t * Math.Exp(-ax * ax);
        return sign * y;
    }

    /// <summary>Inverse standard-normal CDF (probit) via Acklam's rational approximation.</summary>
    private static double NormalQuantile(double p)
    {
        if (p <= 0.0)
        {
            return double.NegativeInfinity;
        }

        if (p >= 1.0)
        {
            return double.PositiveInfinity;
        }

        // Coefficients for Acklam's algorithm.
        double[] a =
        {
            -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
        };
        double[] b =
        {
            -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01,
        };
        double[] c =
        {
            -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
        };
        double[] d =
        {
            7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
            3.754408661907416e+00,
        };

        const double pLow = 0.02425;
        const double pHigh = 1.0 - pLow;
        double q, r, x;

        if (p < pLow)
        {
            q = Math.Sqrt(-2.0 * Math.Log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        }
        else
        {
            q = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }

        // One Halley refinement step for full double precision.
        double e = NormalCdf(x) - p;
        double u = e * Math.Sqrt(2.0 * Math.PI) * Math.Exp(x * x / 2.0);
        x -= u / (1.0 + x * u / 2.0);
        return x;
    }
}
