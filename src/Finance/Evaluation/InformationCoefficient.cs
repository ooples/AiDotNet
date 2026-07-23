using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Finance.Evaluation;

/// <summary>
/// Information Coefficient (IC): the correlation between predicted and realized forward returns, plus
/// its statistical significance (t-statistic, two-sided p-value) and the IC information ratio (ICIR)
/// over a time series of per-period ICs.
/// </summary>
/// <remarks>
/// <para>
/// The Information Coefficient is the north-star metric of return-forecasting research: it measures how
/// well a model's predictions rank/track the returns that actually occurred. Both the Pearson (linear)
/// and Spearman (rank) flavors are provided. An IC of about 0.03-0.05 that is statistically significant
/// already represents a real, exploitable edge in liquid markets.
/// </para>
/// <para><b>For Beginners:</b> Imagine you predict tomorrow's stock moves for 500 stocks, then wait and
/// see what actually happened. The IC asks: "did the stocks I was most bullish on actually go up the
/// most?" An IC of +1 means a perfect match, 0 means no relationship (coin flip), and -1 means your
/// predictions were exactly backwards. Because even a tiny real edge is valuable, we also compute a
/// t-statistic and p-value to check that the IC is "really there" and not just luck. The ICIR is like a
/// Sharpe ratio for your forecasting skill: it rewards a positive IC that shows up <i>consistently</i>
/// period after period, not just on average.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public static class InformationCoefficient<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Pearson (linear) correlation between predicted and realized returns. Returns 0 when either series
    /// has zero variance or fewer than two paired observations.
    /// </summary>
    public static T Pearson(IReadOnlyList<T> predicted, IReadOnlyList<T> realized)
    {
        return NumOps.FromDouble(PearsonDouble(ToDoubles(predicted), ToDoubles(realized)));
    }

    /// <summary>
    /// Spearman rank correlation: Pearson correlation of the (tie-averaged) ranks of the two series.
    /// Robust to outliers and monotone (non-linear) relationships. Returns 0 for degenerate input.
    /// </summary>
    public static T Spearman(IReadOnlyList<T> predicted, IReadOnlyList<T> realized)
    {
        var p = ToDoubles(predicted);
        var r = ToDoubles(realized);
        return NumOps.FromDouble(PearsonDouble(Ranks(p), Ranks(r)));
    }

    /// <summary>
    /// Two-sided significance test for an IC value over <paramref name="n"/> observations.
    /// Computes the t-statistic IC·sqrt((n-2)/(1-IC^2)) and its Student-t two-sided p-value.
    /// </summary>
    /// <param name="ic">The information coefficient (Pearson or Spearman), in [-1, 1].</param>
    /// <param name="n">Number of paired observations the IC was computed from.</param>
    /// <returns>The t-statistic and its two-sided p-value (degrees of freedom = n - 2).</returns>
    public static (T TStatistic, T PValue) Significance(T ic, int n)
    {
        var (t, p) = SignificanceDouble(NumOps.ToDouble(ic), n);
        return (NumOps.FromDouble(t), NumOps.FromDouble(p));
    }

    /// <summary>
    /// Summarizes a time series of per-period ICs into mean IC, IC standard deviation, and the
    /// IC information ratio: ICIR = meanIC / stdIC · sqrt(periods). ICIR rewards a stable, consistently
    /// positive IC. Returns zeros when fewer than two periods are supplied or the IC series has no spread.
    /// </summary>
    public static (T MeanIc, T StdIc, T Icir) InformationRatio(IReadOnlyList<T> perPeriodIcs)
    {
        int periods = perPeriodIcs.Count;
        if (periods < 2)
        {
            return (NumOps.Zero, NumOps.Zero, NumOps.Zero);
        }

        double sum = 0.0;
        for (int i = 0; i < periods; i++)
        {
            sum += NumOps.ToDouble(perPeriodIcs[i]);
        }

        double mean = sum / periods;

        double sse = 0.0;
        for (int i = 0; i < periods; i++)
        {
            double d = NumOps.ToDouble(perPeriodIcs[i]) - mean;
            sse += d * d;
        }

        // Sample standard deviation (n-1), consistent with the rest of the Finance utilities.
        double std = Math.Sqrt(sse / (periods - 1));
        double icir = std > 0.0 ? mean / std * Math.Sqrt(periods) : 0.0;
        return (NumOps.FromDouble(mean), NumOps.FromDouble(std), NumOps.FromDouble(icir));
    }

    // ---- internal double-precision helpers (statistical kernels) ----

    private static double[] ToDoubles(IReadOnlyList<T> values)
    {
        var result = new double[values.Count];
        for (int i = 0; i < values.Count; i++)
        {
            result[i] = NumOps.ToDouble(values[i]);
        }

        return result;
    }

    private static double PearsonDouble(double[] x, double[] y)
    {
        int n = Math.Min(x.Length, y.Length);
        if (n < 2)
        {
            return 0.0;
        }

        double mx = 0.0, my = 0.0;
        for (int i = 0; i < n; i++)
        {
            mx += x[i];
            my += y[i];
        }

        mx /= n;
        my /= n;

        double sxy = 0.0, sxx = 0.0, syy = 0.0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx;
            double dy = y[i] - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }

        double denom = Math.Sqrt(sxx * syy);
        if (denom <= 0.0)
        {
            return 0.0;
        }

        double r = sxy / denom;
        // Clamp to [-1, 1] to absorb floating-point overshoot (net471-safe: no Math.Clamp).
        return Math.Max(-1.0, Math.Min(1.0, r));
    }

    /// <summary>Tie-averaged ("fractional") ranks of a sample, 1-based.</summary>
    private static double[] Ranks(double[] values)
    {
        int n = values.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++)
        {
            idx[i] = i;
        }

        Array.Sort(idx, (a, b) => values[a].CompareTo(values[b]));

        var ranks = new double[n];
        int j = 0;
        while (j < n)
        {
            int k = j;
            while (k + 1 < n && values[idx[k + 1]] == values[idx[j]])
            {
                k++;
            }

            // Average rank for the tied block [j, k] (1-based ranks => +1).
            double avg = (j + k) / 2.0 + 1.0;
            for (int t = j; t <= k; t++)
            {
                ranks[idx[t]] = avg;
            }

            j = k + 1;
        }

        return ranks;
    }

    private static (double T, double P) SignificanceDouble(double ic, int n)
    {
        int df = n - 2;
        if (df <= 0)
        {
            return (0.0, 1.0);
        }

        double oneMinus = 1.0 - ic * ic;
        if (oneMinus <= 0.0)
        {
            // |IC| == 1 (perfect correlation) => infinite t, p == 0.
            double sign = ic >= 0.0 ? 1.0 : -1.0;
            return (sign * double.PositiveInfinity, 0.0);
        }

        double t = ic * Math.Sqrt(df / oneMinus);
        double p = StudentTwoSidedPValue(t, df);
        return (t, p);
    }

    /// <summary>
    /// Two-sided p-value for a Student-t statistic with <paramref name="df"/> degrees of freedom,
    /// via the regularized incomplete beta function: p = I_{df/(df+t^2)}(df/2, 1/2).
    /// </summary>
    private static double StudentTwoSidedPValue(double t, int df)
    {
        if (double.IsInfinity(t))
        {
            return 0.0;
        }

        double x = df / (df + t * t);
        return RegularizedIncompleteBeta(x, df / 2.0, 0.5);
    }

    /// <summary>Regularized incomplete beta I_x(a, b) via the Lentz continued fraction (NR-style).</summary>
    private static double RegularizedIncompleteBeta(double x, double a, double b)
    {
        if (x <= 0.0)
        {
            return 0.0;
        }

        if (x >= 1.0)
        {
            return 1.0;
        }

        double lbeta = LogGamma(a) + LogGamma(b) - LogGamma(a + b);
        double front = Math.Exp(a * Math.Log(x) + b * Math.Log(1.0 - x) - lbeta);

        // Use the symmetry relation for faster CF convergence on the right tail.
        if (x < (a + 1.0) / (a + b + 2.0))
        {
            return front * BetaContinuedFraction(x, a, b) / a;
        }

        return 1.0 - front * BetaContinuedFraction(1.0 - x, b, a) / b;
    }

    private static double BetaContinuedFraction(double x, double a, double b)
    {
        const double tiny = 1e-30;
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < tiny)
        {
            d = tiny;
        }

        d = 1.0 / d;
        double h = d;

        for (int m = 1; m <= 200; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < tiny)
            {
                d = tiny;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < tiny)
            {
                c = tiny;
            }

            d = 1.0 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.Abs(d) < tiny)
            {
                d = tiny;
            }

            c = 1.0 + aa / c;
            if (Math.Abs(c) < tiny)
            {
                c = tiny;
            }

            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.Abs(del - 1.0) < 1e-12)
            {
                break;
            }
        }

        return h;
    }

    /// <summary>Lanczos approximation of ln(Gamma(z)) for z &gt; 0.</summary>
    private static double LogGamma(double z)
    {
        double[] g =
        {
            676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7,
        };

        if (z < 0.5)
        {
            // Reflection formula.
            return Math.Log(Math.PI / Math.Sin(Math.PI * z)) - LogGamma(1.0 - z);
        }

        z -= 1.0;
        double a = 0.99999999999980993;
        double tval = z + 7.5;
        for (int i = 0; i < g.Length; i++)
        {
            a += g[i] / (z + i + 1.0);
        }

        return 0.5 * Math.Log(2.0 * Math.PI) + (z + 0.5) * Math.Log(tval) - tval + Math.Log(a);
    }
}
