using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Levene's test for equality of variances across groups.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Levene's test checks if groups have equal variance:
/// <list type="bullet">
/// <item>Important assumption for many statistical tests (t-test, ANOVA)</item>
/// <item>More robust than Bartlett's test to non-normality</item>
/// <item>Uses deviations from group means or medians</item>
/// </list>
/// </para>
/// <para><b>Variants:</b>
/// <list type="bullet">
/// <item>Mean-based: Original Levene's test</item>
/// <item>Median-based: Brown-Forsythe test (more robust)</item>
/// <item>Trimmed mean: Compromise between the two</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LeveneTest<T> : IStatisticalTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public enum CenterType { Mean, Median, TrimmedMean }
    private readonly CenterType _center;

    /// <inheritdoc />
    public string Name => _center == CenterType.Median ? "Brown-Forsythe Test" : "Levene's Test";
    /// <inheritdoc />
    public string Description => "Tests equality of variances across groups.";
    /// <inheritdoc />
    public bool IsPaired => false;
    /// <inheritdoc />
    public bool IsNonParametric => _center == CenterType.Median;

    /// <summary>
    /// Initializes Levene's test.
    /// </summary>
    /// <param name="center">Center measure to use (Mean, Median, or TrimmedMean).</param>
    public LeveneTest(CenterType center = CenterType.Median)
    {
        _center = center;
    }

    /// <summary>
    /// Tests if multiple groups have equal variances.
    /// </summary>
    /// <param name="groups">Array of score arrays, one per group.</param>
    /// <returns>Statistical test result with p-value.</returns>
    public StatisticalTestResult<T> Test(T[][] groups)
    {
        if (groups is null) throw new ArgumentNullException(nameof(groups));

        int k = groups.Length;
        if (k < 2)
            throw new ArgumentException("Need at least 2 groups for Levene's test.");

        if (groups.Any(g => g is null || g.Length == 0))
            throw new ArgumentException("Each group must contain at least one observation.", nameof(groups));

        var groupSizes = groups.Select(g => g.Length).ToArray();
        int N = groupSizes.Sum();

        // Compute center for each group
        var centers = new double[k];
        for (int g = 0; g < k; g++)
        {
            centers[g] = ComputeCenter(groups[g]);
        }

        // Compute absolute deviations from center
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var (pooled, groupIndex, sizes) = EvaluationTensorOps<T>.PoolGroups(groups);
        var centerOf = engine.TensorGather(new Tensor<double>(centers, new[] { k }), groupIndex, 0);
        var Z = engine.TensorAbs(engine.TensorSubtract(pooled, centerOf));

        // Compute group means of Z
        var ZMeans = engine.TensorDivide(EvaluationTensorOps<double>.GroupSums(Z, groupIndex, k), sizes);
        double ZGrandMean = engine.TensorMean(Z);

        // Compute Levene statistic (F-test on Z values)
        double SSBetween = engine.TensorSum(
            engine.TensorMultiply(sizes, engine.TensorSquare(engine.TensorSubtractScalar(ZMeans, ZGrandMean))));
        double SSWithin = engine.TensorSumOfSquares(
            engine.TensorSubtract(Z, engine.TensorGather(ZMeans, groupIndex, 0)));

        int dfBetween = k - 1;
        int dfWithin = N - k;

        if (dfWithin <= 0 || SSWithin < 1e-10)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                Description = "Insufficient data for Levene's test."
            };
        }

        double F = (SSBetween / dfBetween) / (SSWithin / dfWithin);
        double pValue = 1 - FCDF(F, dfBetween, dfWithin);

        // Compute variances for each group
        var means = engine.TensorDivide(EvaluationTensorOps<double>.GroupSums(pooled, groupIndex, k), sizes);
        var squaredDeviations = engine.TensorSquare(engine.TensorSubtract(pooled, engine.TensorGather(means, groupIndex, 0)));
        var variances = engine.TensorDivide(
            EvaluationTensorOps<double>.GroupSums(squaredDeviations, groupIndex, k), engine.TensorSubtractScalar(sizes, 1.0)).ToArray();

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(F),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < 0.05,
            DegreesOfFreedom = dfBetween,
            Interpretation = pValue < 0.05
                ? $"Variances are significantly different (vars: {string.Join(", ", variances.Select(v => $"{v:F3}"))})"
                : $"No significant difference in variances",
            Description = $"{Name} for homogeneity of variances using {_center} as center."
        };
    }

    private double ComputeCenter(T[] group)
    {
        using var noGrad = new NoGradScope<T>();
        var values = new Tensor<T>(group, new[] { group.Length }).Cast<double>();
        return _center switch
        {
            CenterType.Mean => Mean(values),
            CenterType.Median => Median(values),
            CenterType.TrimmedMean => TrimmedMean(values, 0.1),
            _ => Mean(values)
        };
    }

    private static double Mean(Tensor<double> values)
        => AiDotNetEngine.Current.TensorMean(values);

    private static double Median(Tensor<double> values)
    {
        int n = values.Length;
        var sorted = EvaluationTensorOps<double>.SortNaNFirst(values);
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[n / 2];
    }

    private static double TrimmedMean(Tensor<double> values, double proportion)
    {
        int n = values.Length;
        int trimCount = (int)(n * proportion);
        if (trimCount * 2 >= n) return Median(values);
        var sorted = EvaluationTensorOps<double>.SortNaNFirst(values);
        return AiDotNetEngine.Current.TensorMean(AiDotNetEngine.Current.TensorNarrow(sorted, 0, trimCount, n - 2 * trimCount));
    }

    private double FCDF(double f, int df1, int df2)
    {
        if (f <= 0) return 0;
        double x = df2 / (df2 + df1 * f);
        return 1 - IncompleteBeta(df2 / 2.0, df1 / 2.0, x);
    }

    private double IncompleteBeta(double a, double b, double x)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        // Use continued fraction expansion
        double bt = Math.Exp(
            LogGamma(a + b) - LogGamma(a) - LogGamma(b) +
            a * Math.Log(x) + b * Math.Log(1 - x));

        if (x < (a + 1) / (a + b + 2))
            return bt * BetaCF(a, b, x) / a;
        else
            return 1 - bt * BetaCF(b, a, 1 - x) / b;
    }

    private double BetaCF(double a, double b, double x)
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
            double delta = d * c;
            h *= delta;

            if (Math.Abs(delta - 1) < epsilon) break;
        }

        return h;
    }

    private double LogGamma(double x)
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
        return -tmp + Math.Log(2.5066282746310005 * sum / x);
    }
}
