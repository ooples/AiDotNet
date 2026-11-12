using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "Kruskal-Wallis H Test";
    /// <inheritdoc />
    public string Description => "Non-parametric test for comparing multiple independent groups.";
    /// <inheritdoc />
    public bool IsPaired => false;
    /// <inheritdoc />
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

        // Pool all observations and compute ranks (tied values share their average rank)
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (pooled, groupIndex) = Pool(groups, N);
        var ranks = EvaluationTensorOps<T>.MidRanks(pooled);

        // Compute rank sums for each group
        var rankSums = engine.TensorIndexAdd(new Tensor<T>(new[] { k }), 0, groupIndex, ranks);
        var sizes = new Tensor<int>(groupSizes, new[] { k }).Cast<T>();

        // Compute H statistic
        double sumRankSquaredOverN = NumOps.ToDouble(engine.TensorSum(engine.TensorDivide(engine.TensorSquare(rankSums), sizes)));
        double H = (12.0 / ((double)N * (N + 1))) * sumRankSquaredOverN - 3.0 * (N + 1);

        // Tie correction: sum of t^3 - t over the groups of t equal values (singletons add 0)
        var tieCounts = engine.TensorUniqueWithInfo(pooled, true, false, true).Counts;
        if (tieCounts is not null && tieCounts.Length < N)
        {
            var t = tieCounts.Cast<T>();
            double tieSum = NumOps.ToDouble(engine.TensorSum(engine.TensorSubtract(engine.TensorMultiply(engine.TensorSquare(t), t), t)));
            double tieCorrection = 1 - tieSum / ((double)N * N * N - N);
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
    public Dictionary<(int, int), T> DunnPostHoc(T[][] groups)
    {
        int k = groups.Length;
        int N = groups.Sum(g => g.Length);
        var results = new Dictionary<(int, int), T>();

        // Pool and rank (same as main test)
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var groupSizes = groups.Select(g => g.Length).ToArray();
        var (pooled, groupIndex) = Pool(groups, N);
        var ranks = EvaluationTensorOps<T>.MidRanks(pooled);

        // Compute mean ranks per group: each observation's rank goes to its own group
        var sizes = new Tensor<int>(groupSizes, new[] { k }).Cast<T>();
        var meanRanks = engine.TensorDivide(engine.TensorIndexAdd(new Tensor<T>(new[] { k }), 0, groupIndex, ranks), sizes);

        // Pairwise comparisons: z = |R_i - R_j| / sqrt(N (N + 1) / 12 * (1 / n_i + 1 / n_j)), p = 2 * (1 - Phi(z))
        var shape = new[] { k, k };
        var column = new[] { k, 1 };
        var row = new[] { 1, k };
        var rankGap = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(meanRanks, column), shape),
            engine.TensorBroadcastTo(engine.Reshape(meanRanks, row), shape)));
        var inverseSizes = engine.TensorReciprocal(sizes);
        var inverseSizeSum = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.Reshape(inverseSizes, column), shape),
            engine.TensorBroadcastTo(engine.Reshape(inverseSizes, row), shape));
        var standardError = engine.TensorSqrt(engine.TensorMultiplyScalar(inverseSizeSum, NumOps.FromDouble((double)N * (N + 1) / 12.0)));
        var z = engine.TensorDivide(rankGap, standardError);
        var pValues = engine.TensorMultiplyScalar(engine.TensorNdtr(engine.TensorNegate(z)), NumOps.FromDouble(2.0));

        int cell = 0;
        foreach (T pValue in pValues)
        {
            int i = cell / k, j = cell % k;
            if (i < j) results[(i, j)] = pValue;
            cell++;
        }

        return results;
    }

    private static (Tensor<T> Pooled, Tensor<int> GroupIndex) Pool(T[][] groups, int total)
    {
        var parts = new Tensor<T>[groups.Length];
        var groupIndex = new int[total];
        int position = 0;
        for (int g = 0; g < groups.Length; g++)
        {
            parts[g] = new Tensor<T>(groups[g], new[] { groups[g].Length });
            for (int i = 0; i < groups[g].Length; i++)
                groupIndex[position++] = g;
        }

        return (AiDotNetEngine.Current.TensorConcatenate(parts, 0), new Tensor<int>(groupIndex, new[] { total }));
    }

    private static T Median(T[] values)
    {
        int n = values.Length;
        if (n == 0) return NumOps.Zero;
        var (lower, upper) = EvaluationTensorOps<T>.OrderStatistics(new Tensor<T>(values, new[] { n }), (n - 1) / 2, n / 2);
        return n % 2 == 0 ? NumOps.Divide(NumOps.Add(lower, upper), NumOps.FromDouble(2.0)) : upper;
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
