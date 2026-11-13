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
        var (pooled, groupIndex, sizes) = EvaluationTensorOps<T>.PoolNativeGroups(groups);
        var (ranks, tieSizes) = EvaluationTensorOps<T>.MidRanksWithTies(pooled);

        // Compute rank sums for each group
        var rankSums = EvaluationTensorOps<double>.GroupSums(ranks, groupIndex, k);

        // Compute H statistic
        double sumRankSquaredOverN = engine.TensorSum(engine.TensorDivide(engine.TensorSquare(rankSums), sizes));
        double H = (12.0 / ((double)N * (N + 1))) * sumRankSquaredOverN - 3.0 * (N + 1);

        // Tie correction: sum of t^3 - t over the groups of t equal values (singletons add 0)
        // Each tie of size t contributes t copies of t^2 - 1, totaling t^3 - t.
        double tieSum = engine.TensorSum(engine.TensorSquare(tieSizes.Cast<double>())) - N;
        if (tieSum > 0)
        {
            double tieCorrection = 1 - tieSum / ((double)N * N * N - N);
            if (tieCorrection > 1e-10) H /= tieCorrection;
        }

        int df = k - 1;
        double pValue = ChiSquareSurvival(H, df);

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
        if (k < 2) throw new ArgumentException("Need at least 2 groups for Dunn's test.", nameof(groups));
        if (groups.Any(group => group.Length == 0))
            throw new ArgumentException("Each group must contain at least one observation.", nameof(groups));
        int N = groups.Sum(g => g.Length);
        var results = new Dictionary<(int, int), double>();

        // Pool and rank (same as main test)
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (pooled, groupIndex, sizes) = EvaluationTensorOps<T>.PoolNativeGroups(groups);
        var (ranks, tieSizes) = EvaluationTensorOps<T>.MidRanksWithTies(pooled);

        double tieSum = engine.TensorSum(engine.TensorSquare(tieSizes.Cast<double>())) - N;
        double variance = Math.Max(0, (double)N * (N + 1) / 12.0 - tieSum / (12.0 * (N - 1)));
        if (variance == 0)
        {
            for (int i = 0; i < k; i++)
                for (int j = i + 1; j < k; j++) results[(i, j)] = 1.0;
            return results;
        }

        // Compute mean ranks per group: each observation's rank goes to its own group
        var meanRanks = engine.TensorDivide(EvaluationTensorOps<double>.GroupSums(ranks, groupIndex, k), sizes);

        // Pairwise comparisons use Dunn's tie-corrected rank variance.
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
        var standardError = engine.TensorSqrt(engine.TensorMultiplyScalar(inverseSizeSum, variance));
        var z = engine.TensorDivide(rankGap, standardError);
        var pValues = engine.TensorMultiplyScalar(engine.TensorNdtr(engine.TensorNegate(z)), 2.0);

        int cell = 0;
        foreach (double pValue in pValues)
        {
            int i = cell / k, j = cell % k;
            if (i < j) results[(i, j)] = pValue;
            cell++;
        }

        return results;
    }

    private static double Median(T[] values)
    {
        int n = values.Length;
        if (n == 0) return 0;
        var (lower, upper) = EvaluationTensorOps<double>.OrderStatistics(
            new Tensor<T>(values, new[] { n }).Cast<double>(), (n - 1) / 2, n / 2);
        return n % 2 == 0 ? (lower + upper) / 2 : upper;
    }

    private static double ChiSquareSurvival(double statistic, int degreesOfFreedom)
    {
        if (degreesOfFreedom <= 0) throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom));
        if (double.IsNaN(statistic)) return double.NaN;
        if (statistic <= 0) return 1;
        if (double.IsPositiveInfinity(statistic)) return 0;
        double shape = degreesOfFreedom / 2.0, x = statistic / 2.0;
        double logScale = shape * Math.Log(x) - x - LogGamma(shape);
        const double epsilon = 2e-15, tiny = 1e-300;
        const int maxIterations = 100000;
        if (x < shape + 1)
        {
            // The lower-tail series converges here; subtraction is safe while that tail is small.
            double sum = 1 / shape, term = sum;
            for (int n = 1; n <= maxIterations; n++)
            {
                term *= x / (shape + n);
                sum += term;
                if (Math.Abs(term) <= epsilon * Math.Abs(sum))
                    return Math.Max(0, Math.Min(1, 1 - Math.Exp(logScale) * sum));
            }
        }
        else
        {
            // Evaluate the upper tail directly with a continued fraction, avoiding 1 - CDF cancellation.
            double b = x + 1 - shape, c = 1 / tiny, d = 1 / b, fraction = d;
            for (int n = 1; n <= maxIterations; n++)
            {
                double numerator = -n * (n - shape);
                b += 2;
                d = numerator * d + b;
                if (Math.Abs(d) < tiny) d = d < 0 ? -tiny : tiny;
                c = b + numerator / c;
                if (Math.Abs(c) < tiny) c = c < 0 ? -tiny : tiny;
                d = 1 / d;
                double change = d * c;
                fraction *= change;
                if (Math.Abs(change - 1) <= epsilon)
                    return Math.Max(0, Math.Min(1, Math.Exp(logScale) * fraction));
            }
        }
        throw new ArithmeticException("The chi-square tail calculation did not converge.");
    }

    private static double LogGamma(double value)
    {
        // Lanczos approximation, evaluated in log space so large degrees of freedom stay finite.
        double[] coefficients =
        {
            676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        };
        double z = value - 1, sum = 0.99999999999980993;
        for (int i = 0; i < coefficients.Length; i++) sum += coefficients[i] / (z + i + 1);
        double t = z + 7.5;
        return 0.9189385332046727 + (z + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }

}
