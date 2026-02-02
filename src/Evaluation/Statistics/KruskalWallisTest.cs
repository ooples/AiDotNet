using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Kruskal-Wallis H test for comparing multiple independent groups.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Kruskal-Wallis is the non-parametric alternative to one-way ANOVA:
/// <list type="bullet">
/// <item>Compares medians of 3+ independent groups</item>
/// <item>No assumption of normal distribution</item>
/// <item>Uses ranks instead of raw values</item>
/// <item>Tests if at least one group differs from others</item>
/// </list>
/// </para>
/// <para><b>Difference from Friedman test:</b>
/// <list type="bullet">
/// <item>Kruskal-Wallis: independent groups (different samples)</item>
/// <item>Friedman: paired/matched groups (same samples, different treatments)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KruskalWallisTest<T> : IMultipleComparisonTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "Kruskal-Wallis H Test";
    public string Description => "Non-parametric test for comparing multiple independent groups.";
    public bool IsPaired => false;
    public bool IsNonParametric => true;

    /// <summary>
    /// Tests if multiple groups have significantly different distributions.
    /// </summary>
    /// <param name="groups">Array of score arrays, one per group.</param>
    /// <returns>Statistical test result with p-value.</returns>
    public StatisticalTestResult<T> Test(T[][] groups)
    {
        int k = groups.Length;
        if (k < 2)
            throw new ArgumentException("Need at least 2 groups for Kruskal-Wallis test.");
        if (groups.Any(g => g.Length == 0))
            throw new ArgumentException("Each group must contain at least one observation.", nameof(groups));

        var groupSizes = groups.Select(g => g.Length).ToArray();
        int N = groupSizes.Sum();

        // Pool all observations and compute ranks
        var pooled = new List<(double value, int groupIdx, int sampleIdx)>();
        for (int g = 0; g < k; g++)
        {
            for (int i = 0; i < groups[g].Length; i++)
            {
                pooled.Add((NumOps.ToDouble(groups[g][i]), g, i));
            }
        }

        // Sort and assign ranks (handle ties by average rank)
        var sorted = pooled.OrderBy(x => x.value).ToList();
        var ranks = new double[N];

        int idx = 0;
        while (idx < N)
        {
            int start = idx;
            double value = sorted[start].value;

            // Find all tied values
            while (idx < N && Math.Abs(sorted[idx].value - value) < 1e-10)
                idx++;

            // Assign average rank to all tied values
            double avgRank = (start + 1 + idx) / 2.0;
            for (int i = start; i < idx; i++)
            {
                int originalIdx = pooled.IndexOf(sorted[i]);
                ranks[originalIdx] = avgRank;
            }
        }

        // Compute rank sums for each group
        var rankSums = new double[k];
        int currentIdx = 0;
        for (int g = 0; g < k; g++)
        {
            for (int i = 0; i < groups[g].Length; i++)
            {
                rankSums[g] += ranks[currentIdx];
                currentIdx++;
            }
        }

        // Compute H statistic
        double sumRankSquaredOverN = 0;
        for (int g = 0; g < k; g++)
        {
            sumRankSquaredOverN += (rankSums[g] * rankSums[g]) / groupSizes[g];
        }

        double H = (12.0 / (N * (N + 1))) * sumRankSquaredOverN - 3 * (N + 1);

        // Tie correction
        var tieGroups = sorted.GroupBy(x => x.value).Where(g => g.Count() > 1).Select(g => g.Count()).ToList();
        if (tieGroups.Count > 0)
        {
            double tieCorrection = 1 - tieGroups.Sum(t => (double)t * t * t - t) / (N * N * N - N);
            if (tieCorrection > 1e-10)
                H /= tieCorrection;
        }

        int df = k - 1;
        double pValue = 1 - ChiSquareCDF(H, df);

        // Compute median for each group
        var medians = groups.Select(g => Median(g)).ToArray();

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(H),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < 0.05,
            DegreesOfFreedom = df,
            Interpretation = pValue < 0.05
                ? $"Significant difference among {k} groups (medians: {string.Join(", ", medians.Select(m => $"{m:F3}"))})"
                : $"No significant difference among {k} groups",
            Description = "Kruskal-Wallis H test (non-parametric ANOVA alternative)."
        };
    }

    /// <summary>
    /// Performs Dunn's post-hoc test for pairwise comparisons.
    /// </summary>
    public Dictionary<(int, int), double> DunnPostHoc(T[][] groups)
    {
        int k = groups.Length;
        int N = groups.Sum(g => g.Length);
        var results = new Dictionary<(int, int), double>();

        // Pool and rank (same as main test)
        var pooled = new List<(double value, int groupIdx)>();
        for (int g = 0; g < k; g++)
        {
            foreach (var val in groups[g])
                pooled.Add((NumOps.ToDouble(val), g));
        }

        var sorted = pooled.OrderBy(x => x.value).ToList();
        var ranks = new double[N];
        int idx = 0;
        int rankIdx = 0;
        while (idx < N)
        {
            int start = idx;
            double value = sorted[start].value;
            while (idx < N && Math.Abs(sorted[idx].value - value) < 1e-10)
                idx++;
            double avgRank = (start + 1 + idx) / 2.0;
            for (int i = start; i < idx; i++)
                ranks[rankIdx++] = avgRank;
        }

        // Compute mean ranks per group
        var meanRanks = new double[k];
        var groupSizes = new int[k];
        rankIdx = 0;
        for (int g = 0; g < k; g++)
        {
            groupSizes[g] = groups[g].Length;
            for (int i = 0; i < groupSizes[g]; i++)
            {
                meanRanks[g] += ranks[rankIdx++];
            }
            meanRanks[g] /= groupSizes[g];
        }

        // Pairwise comparisons
        for (int i = 0; i < k - 1; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                double z = Math.Abs(meanRanks[i] - meanRanks[j]) /
                          Math.Sqrt((N * (N + 1) / 12.0) * (1.0 / groupSizes[i] + 1.0 / groupSizes[j]));
                double pValue = 2 * (1 - NormalCDF(z));
                results[(i, j)] = pValue;
            }
        }

        return results;
    }

    private double Median(T[] values)
    {
        var sorted = values.Select(v => NumOps.ToDouble(v)).OrderBy(x => x).ToArray();
        int n = sorted.Length;
        if (n == 0) return 0;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[n / 2];
    }

    private double NormalCDF(double z)
    {
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(z));
        double d = 0.3989423 * Math.Exp(-z * z / 2);
        double prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return z > 0 ? 1 - prob : prob;
    }

    private double ChiSquareCDF(double x, int df)
    {
        if (x <= 0) return 0;
        double a = df / 2.0;
        double x2 = x / 2.0;
        return LowerIncompleteGamma(a, x2) / GammaFunction(a);
    }

    private double LowerIncompleteGamma(double a, double x)
    {
        if (x <= 0) return 0;
        double sum = 1.0 / a;
        double term = 1.0 / a;
        for (int n = 1; n < 100; n++)
        {
            term *= x / (a + n);
            sum += term;
            if (Math.Abs(term) < 1e-10 * Math.Abs(sum)) break;
        }
        return Math.Exp(-x + a * Math.Log(x)) * sum;
    }

    private double GammaFunction(double x)
    {
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
            -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double sum = 1.000000000190015;
        for (int j = 0; j < 6; j++)
        {
            y += 1;
            sum += c[j] / y;
        }
        return Math.Exp(-tmp + Math.Log(2.5066282746310005 * sum / x));
    }
}
