using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "Wilcoxon Signed-Rank Test";
    /// <inheritdoc />
    public string Description => "Non-parametric test for paired samples (robust to non-normality).";

    /// <inheritdoc />
    public StatisticalTestResult<T> Test(T[] sample1, T[] sample2, double alpha = 0.05)
    {
        if (sample1.Length != sample2.Length)
            throw new ArgumentException("Paired samples must have the same length.");
        if (sample1.Length < 5)
            throw new ArgumentException("Need at least 5 paired observations for Wilcoxon test.");

        int n = sample1.Length;

        // Calculate differences and remove zeros
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var tolerance = 1e-10;
        var allDiffs = engine.TensorSubtract(new Tensor<T>(sample1, new[] { n }).Cast<double>(), new Tensor<T>(sample2, new[] { n }).Cast<double>());
        var nonZero = engine.TensorGreaterThan(engine.TensorAbs(allDiffs), tolerance); // Exclude exact ties
        int nr = EvaluationTensorOps<double>.Count(nonZero); // Number of non-zero differences

        if (nr < 3)
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

        // Rank by absolute value; a run of absolute values whose neighbouring gaps are below 1e-10 shares its
        // average rank
        var diffs = EvaluationTensorOps<double>.Select(allDiffs, nonZero);
        var (_, order, ranks) = EvaluationTensorOps<double>.SortedRunRanks(engine.TensorAbs(diffs), 1e-10);

        // Calculate W+ (sum of ranks for positive differences) and W- (the rest)
        var positive = engine.TensorGather(engine.TensorGreaterThan(diffs, 0.0), order, 0);
        double wPlus = engine.TensorSum(engine.TensorMultiply(ranks, positive));
        double wMinus = engine.TensorSum(ranks) - wPlus;

        // Test statistic is the smaller of W+ and W-
        double w = Math.Min(wPlus, wMinus);

        // For n >= 10, use normal approximation
        double mean = (double)nr * (nr + 1) / 4.0;
        double std = Math.Sqrt((double)nr * (nr + 1) * (2.0 * nr + 1) / 24.0);

        // Continuity correction
        double z = (w - mean + 0.5) / std;

        // Two-tailed p-value using normal approximation
        double pValue = 2 * NormalCdf(-Math.Abs(z));

        // Effect size: rank-biserial correlation
        double effectSize = (wPlus - wMinus) / ((double)nr * (nr + 1) / 2.0);

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
