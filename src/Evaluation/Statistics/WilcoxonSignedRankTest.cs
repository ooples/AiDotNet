using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Wilcoxon signed-rank test: non-parametric paired comparison test.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the non-parametric alternative to the paired t-test.
/// Use it when:
/// <list type="bullet">
/// <item>Your differences are not normally distributed</item>
/// <item>You have ordinal data</item>
/// <item>Your sample size is small and you can't verify normality</item>
/// <item>You want a more robust test (less sensitive to outliers)</item>
/// </list>
/// </para>
/// <para><b>Common use in ML:</b> Comparing cross-validation scores of two models when
/// you're not sure the performance differences are normally distributed.</para>
/// </remarks>
public class WilcoxonSignedRankTest<T> : IPairedTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "Wilcoxon Signed-Rank Test";
    public string Description => "Non-parametric test for paired samples (robust to non-normality).";

    public StatisticalTestResult<T> Test(T[] sample1, T[] sample2, double alpha = 0.05)
    {
        if (sample1.Length != sample2.Length)
            throw new ArgumentException("Paired samples must have the same length.");
        if (sample1.Length < 5)
            throw new ArgumentException("Need at least 5 paired observations for Wilcoxon test.");

        int n = sample1.Length;

        // Calculate differences and remove zeros
        var diffsWithIndex = new List<(double diff, int sign, double absVal)>();
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(sample1[i]) - NumOps.ToDouble(sample2[i]);
            if (Math.Abs(diff) > 1e-10) // Exclude exact ties
            {
                diffsWithIndex.Add((diff, diff > 0 ? 1 : -1, Math.Abs(diff)));
            }
        }

        if (diffsWithIndex.Count < 3)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                Alpha = alpha,
                Description = "Too few non-zero differences to perform test."
            };
        }

        int nr = diffsWithIndex.Count; // Number of non-zero differences

        // Rank by absolute value
        var sorted = diffsWithIndex.OrderBy(x => x.absVal).ToList();

        // Assign ranks (handle ties by averaging)
        var ranks = new double[nr];
        int i2 = 0;
        while (i2 < nr)
        {
            int j = i2;
            // Find run of ties
            while (j < nr - 1 && Math.Abs(sorted[j].absVal - sorted[j + 1].absVal) < 1e-10)
                j++;

            // Average rank for ties
            double avgRank = (i2 + j + 2.0) / 2.0; // +2 because ranks start at 1
            for (int k = i2; k <= j; k++)
                ranks[k] = avgRank;

            i2 = j + 1;
        }

        // Calculate W+ (sum of ranks for positive differences)
        double wPlus = 0;
        double wMinus = 0;
        for (int i = 0; i < nr; i++)
        {
            if (sorted[i].sign > 0)
                wPlus += ranks[i];
            else
                wMinus += ranks[i];
        }

        // Test statistic is the smaller of W+ and W-
        double w = Math.Min(wPlus, wMinus);

        // For n >= 10, use normal approximation
        double mean = nr * (nr + 1) / 4.0;
        double std = Math.Sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0);

        // Continuity correction
        double z = (w - mean + 0.5) / std;

        // Two-tailed p-value using normal approximation
        double pValue = 2 * NormalCdf(-Math.Abs(z));

        // Effect size: rank-biserial correlation
        double effectSize = (wPlus - wMinus) / (nr * (nr + 1) / 2.0);

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(w),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < alpha,
            Alpha = alpha,
            EffectSize = NumOps.FromDouble(effectSize),
            Description = $"W+ = {wPlus:F1}, W- = {wMinus:F1}, Effect size (r) = {effectSize:F4}"
        };
    }

    private static double NormalCdf(double x)
    {
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
}
