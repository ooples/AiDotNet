using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Paired t-test: compares means of two related samples.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Use this test when you have paired observations, such as:
/// <list type="bullet">
/// <item>Before/after measurements on the same subjects</item>
/// <item>Same dataset evaluated by two different models (paired by sample)</item>
/// <item>Cross-validation fold results for two models on the same folds</item>
/// </list>
/// The test determines if there's a significant difference between the pairs.</para>
/// <para><b>Assumptions:</b>
/// <list type="bullet">
/// <item>The differences between pairs are approximately normally distributed</item>
/// <item>Pairs are independent of each other</item>
/// <item>Data is continuous (or at least ordinal with many levels)</item>
/// </list>
/// </para>
/// </remarks>
public class PairedTTest<T> : IPairedTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "Paired t-test";
    public string Description => "Compares means of two related samples (e.g., same dataset, two models).";

    public StatisticalTestResult<T> Test(T[] sample1, T[] sample2, double alpha = 0.05)
    {
        if (sample1.Length != sample2.Length)
            throw new ArgumentException("Paired samples must have the same length.");
        if (sample1.Length < 2)
            throw new ArgumentException("Need at least 2 paired observations.");

        int n = sample1.Length;

        // Calculate differences
        var differences = new double[n];
        for (int i = 0; i < n; i++)
        {
            differences[i] = NumOps.ToDouble(sample1[i]) - NumOps.ToDouble(sample2[i]);
        }

        // Calculate mean and standard deviation of differences
        double meanDiff = differences.Average();
        double sumSqDiff = differences.Sum(d => (d - meanDiff) * (d - meanDiff));
        double stdDiff = Math.Sqrt(sumSqDiff / (n - 1));

        // Calculate t-statistic
        double standardError = stdDiff / Math.Sqrt(n);
        double tStat = meanDiff / standardError;

        // Calculate degrees of freedom
        int df = n - 1;

        // Calculate p-value using t-distribution approximation
        double pValue = ApproximateTDistributionPValue(Math.Abs(tStat), df);

        // Calculate Cohen's d effect size
        double cohensD = meanDiff / stdDiff;

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(tStat),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < alpha,
            Alpha = alpha,
            DegreesOfFreedom = df,
            EffectSize = NumOps.FromDouble(cohensD),
            Description = $"Mean difference = {meanDiff:F4}, Cohen's d = {cohensD:F4}"
        };
    }

    private static double ApproximateTDistributionPValue(double t, int df)
    {
        // Using normal approximation for large df, or simple approximation for small df
        if (df > 100)
        {
            // Approximate with normal distribution for large df
            return 2 * (1 - NormalCdf(t));
        }

        // Simple approximation using transformation
        double x = df / (df + t * t);
        double p = IncompleteBeta(df / 2.0, 0.5, x);
        return p; // Two-tailed
    }

    private static double NormalCdf(double x)
    {
        // Approximation of standard normal CDF
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x) / Math.Sqrt(2);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return 0.5 * (1.0 + sign * y);
    }

    private static double IncompleteBeta(double a, double b, double x)
    {
        // Simple approximation of incomplete beta function
        if (x < 0 || x > 1) return 0;
        if (x == 0) return 0;
        if (x == 1) return 1;

        // Use continued fraction approximation
        double bt = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x) + b * Math.Log(1 - x));

        if (x < (a + 1) / (a + b + 2))
            return bt * BetaContinuedFraction(a, b, x) / a;
        else
            return 1 - bt * BetaContinuedFraction(b, a, 1 - x) / b;
    }

    private static double BetaContinuedFraction(double a, double b, double x)
    {
        const int maxIterations = 100;
        const double epsilon = 1e-10;

        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.Abs(d) < epsilon) d = epsilon;
        d = 1 / d;
        double h = d;

        for (int m = 1; m <= maxIterations; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < epsilon) d = epsilon;
            c = 1 + aa / c;
            if (Math.Abs(c) < epsilon) c = epsilon;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < epsilon) d = epsilon;
            c = 1 + aa / c;
            if (Math.Abs(c) < epsilon) c = epsilon;
            d = 1 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < epsilon) break;
        }

        return h;
    }

    private static double LogGamma(double x)
    {
        // Stirling's approximation for log(Gamma(x))
        double[] coef = { 76.18009173, -86.50532033, 24.01409822, -1.231739516, 0.00120858003, -0.00000536382 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
            ser += coef[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }
}
