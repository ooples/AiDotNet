using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Friedman test: non-parametric test for comparing multiple classifiers across multiple datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Friedman test is the standard statistical test for comparing
/// multiple machine learning algorithms across multiple datasets. It's recommended by Demsar (2006)
/// for ML algorithm comparison.
/// <list type="bullet">
/// <item>Ranks each algorithm within each dataset</item>
/// <item>Tests if the average ranks are significantly different</item>
/// <item>Non-parametric: doesn't assume normal distribution of scores</item>
/// </list>
/// </para>
/// <para><b>Typical workflow:</b>
/// <list type="number">
/// <item>Run k algorithms on n datasets (e.g., via cross-validation)</item>
/// <item>Use Friedman test to see if there's any significant difference</item>
/// <item>If significant, use post-hoc tests (e.g., Nemenyi) to find which pairs differ</item>
/// </list>
/// </para>
/// </remarks>
public class FriedmanTest<T> : IMultipleSampleTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "Friedman Test";
    public string Description => "Non-parametric test for comparing multiple algorithms across datasets.";

    /// <summary>
    /// Performs the Friedman test.
    /// </summary>
    /// <param name="samples">Array of performance scores: samples[algorithm][dataset].</param>
    /// <param name="alpha">Significance level.</param>
    /// <returns>Test result.</returns>
    public StatisticalTestResult<T> Test(T[][] samples, double alpha = 0.05)
    {
        if (samples.Length < 2)
            throw new ArgumentException("Need at least 2 algorithms to compare.");

        int k = samples.Length; // Number of algorithms
        int n = samples[0].Length; // Number of datasets

        // Verify all algorithms have same number of datasets
        foreach (var s in samples)
        {
            if (s.Length != n)
                throw new ArgumentException("All algorithms must be evaluated on the same number of datasets.");
        }

        if (n < 3)
            throw new ArgumentException("Need at least 3 datasets for Friedman test.");

        // Convert to double and transpose: results[dataset][algorithm]
        var results = new double[n][];
        for (int i = 0; i < n; i++)
        {
            results[i] = new double[k];
            for (int j = 0; j < k; j++)
            {
                results[i][j] = NumOps.ToDouble(samples[j][i]);
            }
        }

        // Rank each dataset (higher value = rank 1 for performance metrics)
        var ranks = new double[k];
        for (int i = 0; i < n; i++)
        {
            var datasetRanks = RankWithTies(results[i], ascending: false);
            for (int j = 0; j < k; j++)
            {
                ranks[j] += datasetRanks[j];
            }
        }

        // Average ranks
        for (int j = 0; j < k; j++)
            ranks[j] /= n;

        // Friedman statistic (chi-squared approximation)
        double sumSquaredRanks = ranks.Sum(r => r * r);
        double chiSquared = 12.0 * n / (k * (k + 1)) * (sumSquaredRanks - k * Math.Pow((k + 1) / 2.0, 2));

        // Iman-Davenport correction (more conservative, recommended)
        double fStat = (n - 1) * chiSquared / (n * (k - 1) - chiSquared);
        int df1 = k - 1;
        int df2 = (k - 1) * (n - 1);

        // Calculate p-value using F-distribution
        double pValue = FDistributionPValue(fStat, df1, df2);

        // Build description with average ranks
        var rankDescriptions = new List<string>();
        for (int j = 0; j < k; j++)
        {
            rankDescriptions.Add($"Algo{j + 1}={ranks[j]:F2}");
        }

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(fStat),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < alpha,
            Alpha = alpha,
            DegreesOfFreedom = df1,
            Description = $"Avg ranks: {string.Join(", ", rankDescriptions)}. F({df1},{df2}) = {fStat:F4}"
        };
    }

    private static double[] RankWithTies(double[] values, bool ascending = true)
    {
        int n = values.Length;
        var indexed = values.Select((v, i) => (value: v, index: i)).ToList();

        if (ascending)
            indexed = indexed.OrderBy(x => x.value).ToList();
        else
            indexed = indexed.OrderByDescending(x => x.value).ToList();

        var ranks = new double[n];
        int i = 0;
        while (i < n)
        {
            int j = i;
            // Find run of ties
            while (j < n - 1 && Math.Abs(indexed[j].value - indexed[j + 1].value) < 1e-10)
                j++;

            // Average rank for ties
            double avgRank = (i + j + 2.0) / 2.0;
            for (int m = i; m <= j; m++)
                ranks[indexed[m].index] = avgRank;

            i = j + 1;
        }

        return ranks;
    }

    private static double FDistributionPValue(double f, int df1, int df2)
    {
        if (f <= 0) return 1.0;

        // Using incomplete beta function: P(F > f) = I_x(df2/2, df1/2)
        // where x = df2 / (df2 + df1 * f)
        double x = df2 / (df2 + df1 * f);
        return IncompleteBeta(df2 / 2.0, df1 / 2.0, x);
    }

    private static double IncompleteBeta(double a, double b, double x)
    {
        if (x < 0 || x > 1) return 0;
        if (x == 0) return 0;
        if (x == 1) return 1;

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
