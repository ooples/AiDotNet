using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.CertifiedRobustness;

/// <summary>
/// Engine-op helpers shared by the certified defenses: class decisions and certified-accuracy summaries.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
internal static class CertificationMath<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Index of the largest entry (first one on ties), or 0 for an empty vector.
    /// </summary>
    public static int ArgMax(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return 0;
        }

        return AiDotNetEngine.Current.TensorArgMax(values.AsTensor(), 0).ToArray()[0];
    }

    /// <summary>
    /// Builds the certified-accuracy metrics from per-input counts and the radii of certified inputs.
    /// </summary>
    /// <param name="cleanCorrect">Inputs the base model classifies correctly.</param>
    /// <param name="certifiedCorrect">Inputs certified correct at the evaluation radius.</param>
    /// <param name="radius">The evaluation radius.</param>
    /// <param name="certifiedCount">Inputs that were certified at all (not abstained).</param>
    /// <param name="radii">The certified radius of every input, zero where it was not certified.</param>
    /// <remarks>
    /// The average and median run over all examples, as <see cref="CertifiedAccuracyMetrics{T}"/> documents;
    /// the certification rate is the fraction of examples that were certified.
    /// </remarks>
    public static CertifiedAccuracyMetrics<T> Summarize(
        int cleanCorrect,
        int certifiedCorrect,
        int certifiedCount,
        T radius,
        List<T> radii)
    {
        var metrics = new CertifiedAccuracyMetrics<T>
        {
            CleanAccuracy = radii.Count > 0 ? (double)cleanCorrect / radii.Count : 0.0,
            CertifiedAccuracy = radii.Count > 0 ? (double)certifiedCorrect / radii.Count : 0.0,
            CertificationRadius = radius,
            CertificationRate = radii.Count > 0 ? (double)certifiedCount / radii.Count : 0.0,
            AverageCertifiedRadius = NumOps.Zero,
            MedianCertifiedRadius = NumOps.Zero
        };

        if (radii.Count > 0)
        {
            var engine = AiDotNetEngine.Current;
            var values = new Vector<T>(radii);
            metrics.AverageCertifiedRadius = engine.Mean<T>(values);

            // Median: the middle order statistic, or the mean of the two middle ones for an even count.
            var order = engine.TensorArgsort(values.AsTensor(), 0, false);
            var sorted = engine.TensorGather(values.AsTensor(), order, 0).ToVector();
            int count = sorted.Length;
            int mid = count / 2;
            var span = sorted.AsSpan();
            metrics.MedianCertifiedRadius = count % 2 == 0
                ? NumOps.Divide(NumOps.Add(span[mid - 1], span[mid]), NumOps.FromDouble(2.0))
                : span[mid];
        }

        return metrics;
    }
}

/// <summary>
/// The statistics of randomized smoothing (Cohen et al. 2019), computed in double whatever the model's numeric
/// type: binomial bounds of counts up to <see cref="int.MaxValue"/>, and the Gaussian quantile of a probability
/// that may round to 1 in a narrow type such as Half.
/// </summary>
internal static class SmoothingStatistics
{
    /// <summary>
    /// The one-sided Clopper-Pearson lower confidence bound of a binomial proportion at level
    /// 1 - alpha = <paramref name="confidence"/> (Cohen et al. 2019, LowerConfBound).
    /// </summary>
    public static double ClopperPearsonLower(int successes, int trials, double confidence)
        => StatisticsHelper<double>.CalculateClopperPearsonLowerBound(successes, trials, confidence);

    /// <summary>The upper end of the two-sided Clopper-Pearson interval at <paramref name="confidence"/>.</summary>
    public static double ClopperPearsonUpper(int successes, int trials, double confidence)
        => StatisticsHelper<double>.CalculateClopperPearsonInterval(successes, trials, confidence).Upper;

    /// <summary>
    /// The two-sided binomial test p-value of k successes in n trials against p = 1/2 (scipy binomtest), for k
    /// at least n / 2: min(1, 2 P(X &gt;= k)), with P(X &gt;= k) = I_{1/2}(k, n - k + 1).
    /// </summary>
    public static double TwoSidedBinomialPValue(int successes, int trials)
    {
        if (trials <= 0 || successes <= 0)
        {
            return 1.0;
        }

        double upperTail = StatisticsHelper<double>.CalculateBetaCDF(0.5, successes, trials - successes + 1.0);
        return Math.Min(1.0, 2.0 * upperTail);
    }

    /// <summary>
    /// The certified L2 radius sigma * Phi^-1(p_A_lower) (Cohen et al. 2019, Theorem 1 with p_B = 1 - p_A), or 0
    /// when p_A_lower is at most 1/2.
    /// </summary>
    public static double Radius(double lowerBound, double sigma)
    {
        if (!(lowerBound > 0.5))
        {
            return 0.0;
        }

        return Math.Max(sigma * InverseNormal(lowerBound), 0.0);
    }

    /// <summary>
    /// The standard normal quantile Phi^-1(p) by Wichura's algorithm AS 241 (PPND16), accurate to about 1e-16.
    /// </summary>
    /// <remarks>
    /// The certified radius scales this quantile, so it must not overstate it: the rational approximation of
    /// Abramowitz and Stegun 26.2.23 errs by up to 4.5e-4, which would certify a radius the smoothed classifier
    /// does not have.
    /// </remarks>
    public static double InverseNormal(double p)
    {
        if (!(p > 0.0 && p < 1.0))
        {
            throw new ArgumentOutOfRangeException(nameof(p), "Probability must be strictly between 0 and 1.");
        }

        double q = p - 0.5;
        if (Math.Abs(q) <= 0.425)
        {
            double r = 0.180625 - q * q;
            return q * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                        + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                        + 133.14166789178437745) * r + 3.387132872796366608)
                / (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                        + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                        + 42.313330701600911252) * r + 1.0);
        }

        double tail = q < 0 ? p : 1.0 - p;
        double s = Math.Sqrt(-Math.Log(tail));
        double value;
        if (s <= 5.0)
        {
            s -= 1.6;
            value = (((((((7.7454501427834140764e-4 * s + 0.0227238449892691845833) * s + 0.24178072517745061177) * s
                        + 1.27045825245236838258) * s + 3.64784832476320460504) * s + 5.7694972214606914055) * s
                        + 4.6303378461565452959) * s + 1.42343711074968357734)
                / (((((((1.05075007164441684324e-9 * s + 5.475938084995344946e-4) * s + 0.0151986665636164571966) * s
                        + 0.14810397642748007459) * s + 0.68976733498510000455) * s + 1.6763848301838038494) * s
                        + 2.05319162663775882187) * s + 1.0);
        }
        else
        {
            s -= 5.0;
            value = (((((((2.01033439929228813265e-7 * s + 2.71155556874348757815e-5) * s + 0.0012426609473880784386) * s
                        + 0.026532189526576123093) * s + 0.29656057182850489123) * s + 1.7848265399172913358) * s
                        + 5.4637849111641143699) * s + 6.6579046435011037772)
                / (((((((2.04426310338993978564e-15 * s + 1.4215117583164458887e-7) * s + 1.8463183175100546818e-5) * s
                        + 7.868691311456132591e-4) * s + 0.0148753612908506148525) * s + 0.13692988092273580531) * s
                        + 0.59983220655588793769) * s + 1.0);
        }

        return q < 0 ? -value : value;
    }
}
