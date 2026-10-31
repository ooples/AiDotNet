namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// The detection statistics shared by the keyed watermarkers: z-scores, their one-sided p-values, and the mapping
/// from z to the confidence a watermarker reports.
/// </summary>
/// <remarks>
/// <para>
/// A keyed detector computes a statistic that is approximately standard normal when the content carries no mark of
/// its key (unmarked content, or content marked with another key) and grows with the evidence when it does:
/// </para>
/// <list type="bullet">
/// <item><description>a count test, z = (hits - gamma * n) / sqrt(n * gamma * (1 - gamma)), for watermarks that make a
/// keyed choice per unit (the green-list test of Kirchenbauer et al. 2023, or a keyed synonym choice);</description></item>
/// <item><description>a correlation test, z = sum(w_k * x_k) / sqrt(sum(w_k^2)), for spread-spectrum marks: with a
/// key-independent weight w_k per coefficient and the key's +/-1 pattern x_k, the sum has mean 0 and variance
/// sum(w_k^2) when the pattern is independent of the content.</description></item>
/// </list>
/// <para>
/// <see cref="Confidence"/> maps z linearly from 0 at z = 3 (a one-sided false-positive rate of 0.135%) to 1 at z = 6.
/// The same mapping is used by the keyed image watermarkers, so a confidence means the same thing in every modality.
/// </para>
/// <para>
/// <b>For Beginners:</b> the z-score says how many standard deviations the evidence lies above what chance alone would
/// give. Unmarked content scores around 0, rarely above 3; marked content scores far higher.
/// </para>
/// </remarks>
public static class WatermarkStatistics
{
    /// <summary>The z-score at and below which the confidence is 0. One-sided false-positive rate: 0.135%.</summary>
    public const double ZeroConfidenceZ = 3.0;

    /// <summary>The z-score from which the confidence is 1.</summary>
    public const double FullConfidenceZ = 6.0;

    /// <summary>
    /// The detection confidence for a z-score: 0 up to <see cref="ZeroConfidenceZ"/>, rising linearly to 1 at
    /// <see cref="FullConfidenceZ"/>. NaN is no evidence and scores 0.
    /// </summary>
    public static double Confidence(double z)
    {
        if (double.IsNaN(z)) return 0.0;
        return Math.Max(0.0, Math.Min(1.0, (z - ZeroConfidenceZ) / (FullConfidenceZ - ZeroConfidenceZ)));
    }

    /// <summary>
    /// The z-score of <paramref name="hits"/> successes in <paramref name="trials"/> trials that each succeed with
    /// probability <paramref name="gamma"/> by chance; 0 when there are no trials.
    /// </summary>
    public static double CountZ(int hits, int trials, double gamma)
    {
        if (trials <= 0) return 0.0;
        if (!(gamma > 0 && gamma < 1)) throw new ArgumentOutOfRangeException(nameof(gamma), "gamma must lie strictly between 0 and 1.");
        return (hits - gamma * trials) / Math.Sqrt(trials * gamma * (1 - gamma));
    }

    /// <summary>
    /// The correlation z-score sum(w_k * x_k) / sqrt(sum(w_k^2)) from its numerator and the sum of squared weights;
    /// 0 when the weights carry no energy.
    /// </summary>
    public static double CorrelationZ(double numerator, double sumOfSquaredWeights)
    {
        if (!(sumOfSquaredWeights > 0)) return 0.0;
        double z = numerator / Math.Sqrt(sumOfSquaredWeights);
        return double.IsNaN(z) || double.IsInfinity(z) ? 0.0 : z;
    }

    /// <summary>
    /// The one-sided p-value P(Z &gt;= z) of a standard normal Z, accurate to a relative error below 1.2e-7.
    /// </summary>
    public static double UpperTailProbability(double z)
    {
        if (double.IsNaN(z)) return 1.0;
        return 0.5 * Erfc(z / Math.Sqrt(2.0));
    }

    /// <summary>
    /// The z-score whose one-sided p-value is <paramref name="p"/> (the inverse of
    /// <see cref="UpperTailProbability"/>), by Acklam's rational approximation with one Newton refinement step.
    /// </summary>
    public static double ZForUpperTail(double p)
    {
        if (double.IsNaN(p)) return 0.0;
        if (p <= 0) return double.PositiveInfinity;
        if (p >= 1) return double.NegativeInfinity;
        double z = -NormalQuantile(p);
        // One Halley step on f(z) = P(Z >= z) - p, with f' = -density and f'' = z * density.
        double e = UpperTailProbability(z) - p;
        double density = Math.Exp(-0.5 * z * z) / Math.Sqrt(2 * Math.PI);
        if (density > 0)
        {
            double u = e / density;
            double step = u / (1 - z * u / 2);
            if (!double.IsNaN(step) && !double.IsInfinity(step)) z += step;
        }

        return z;
    }

    /// <summary>
    /// The z-score of the best of <paramref name="tests"/> tests (independent or not) after a Bonferroni correction:
    /// the smallest p-value is multiplied by the number of tests, so the false-positive rate of "any test fires" stays
    /// at the single-test rate. An adjusted p-value of 0.5 or more is no evidence and gives z = 0, so the result is
    /// never negative and is not standard normal under the null hypothesis; only its upper tail is calibrated.
    /// </summary>
    public static double BonferroniZ(double bestZ, int tests)
    {
        if (tests <= 1) return bestZ;
        if (double.IsNaN(bestZ)) return 0.0;
        double p = UpperTailProbability(bestZ) * tests;
        if (p >= 0.5) return 0.0;
        // Beyond z of about 37 the tail probability underflows to 0; the correction is then far below any threshold.
        if (!(p > 1e-300)) return bestZ;
        return Math.Min(bestZ, ZForUpperTail(p));
    }

    /// <summary>A <see cref="WatermarkResult"/> for a z-score over <paramref name="scoredUnits"/> units.</summary>
    internal static WatermarkResult Result(double z, int scoredUnits, string modality, string watermarkType, double reportThreshold)
    {
        double confidence = Confidence(z);
        return new WatermarkResult
        {
            WatermarkDetected = confidence > 0 && confidence >= reportThreshold,
            Confidence = confidence,
            ZScore = z,
            PValue = UpperTailProbability(z),
            ScoredUnits = scoredUnits,
            Modality = modality,
            WatermarkType = watermarkType
        };
    }

    // Complementary error function (Numerical Recipes, erfcc): fractional error below 1.2e-7 everywhere.
    private static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double ans = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
            + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
            + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? ans : 2.0 - ans;
    }

    // Acklam's inverse of the standard normal CDF, relative error below 1.15e-9.
    private static double NormalQuantile(double p)
    {
        double[] a = { -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
        double[] b = { -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
        double[] c = { -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
        double[] d = { 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
        const double low = 0.02425, high = 1 - low;
        if (p < low)
        {
            double q = Math.Sqrt(-2 * Math.Log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }

        if (p > high)
        {
            double q = Math.Sqrt(-2 * Math.Log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }

        double r = p - 0.5, s = r * r;
        return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1);
    }
}
