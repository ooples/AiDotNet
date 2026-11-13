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

        using var noGrad = new NoGradScope<T>();
        var type = Type.GetTypeCode(typeof(T));
        bool native = type is TypeCode.Decimal or TypeCode.Byte or TypeCode.SByte or TypeCode.Int16 or
            TypeCode.UInt16 or TypeCode.Int32 or TypeCode.UInt32 or TypeCode.Int64 or TypeCode.UInt64;
        var (nr, wPlus, wMinus) = native ? NativeRankSums(sample1, sample2) : FloatingRankSums(sample1, sample2);

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

    private static (int Count, double Positive, double Negative) FloatingRankSums(T[] sample1, T[] sample2)
    {
        using var noGrad = new NoGradScope<double>();
        var engine = AiDotNetEngine.Current;
        int n = sample1.Length;
        var allDiffs = engine.TensorSubtract(new Tensor<T>(sample1, new[] { n }).Cast<double>(),
            new Tensor<T>(sample2, new[] { n }).Cast<double>());
        var nonZero = engine.TensorGreaterThan(engine.TensorAbs(allDiffs), 1e-10);
        int count = EvaluationTensorOps<double>.Count(nonZero);
        if (count < 3) return (count, 0, 0);
        var differences = EvaluationTensorOps<double>.Select(allDiffs, nonZero);
        var (_, order, ranks) = EvaluationTensorOps<double>.SortedRunRanks(engine.TensorAbs(differences), 1e-10);
        var positive = engine.TensorGather(engine.TensorGreaterThan(differences, 0.0), order, 0);
        double wPlus = engine.TensorSum(engine.TensorMultiply(ranks, positive));
        return (count, wPlus, engine.TensorSum(ranks) - wPlus);
    }

    private static (int Count, double Positive, double Negative) NativeRankSums(T[] sample1, T[] sample2)
    {
        // Decimal's 96-bit coefficient at a common scale gives exact differences even across its full range.
        // Int64 and UInt64 also fit this representation without rounding or overflowing subtraction.
        var observations = new List<(System.Numerics.BigInteger Magnitude, bool Positive)>(sample1.Length);
        var tolerance = System.Numerics.BigInteger.Pow(10, 18); // 1e-10 in units of 1e-28.
        for (int i = 0; i < sample1.Length; i++)
        {
            var difference = DecimalUnits(Convert.ToDecimal(sample1[i])) - DecimalUnits(Convert.ToDecimal(sample2[i]));
            var magnitude = System.Numerics.BigInteger.Abs(difference);
            if (magnitude > tolerance) observations.Add((magnitude, difference.Sign > 0));
        }
        observations.Sort((left, right) => left.Magnitude.CompareTo(right.Magnitude));
        double wPlus = 0;
        for (int start = 0; start < observations.Count;)
        {
            int end = start + 1;
            while (end < observations.Count && observations[end].Magnitude - observations[end - 1].Magnitude < tolerance)
                end++;
            double rank = (start + (double)end + 1) / 2;
            for (int i = start; i < end; i++)
                if (observations[i].Positive) wPlus += rank;
            start = end;
        }
        double total = observations.Count * (observations.Count + 1.0) / 2;
        return (observations.Count, wPlus, total - wPlus);
    }

    private static System.Numerics.BigInteger DecimalUnits(decimal value)
    {
        var bits = decimal.GetBits(value);
        var coefficient = (System.Numerics.BigInteger)(uint)bits[0]
            | ((System.Numerics.BigInteger)(uint)bits[1] << 32)
            | ((System.Numerics.BigInteger)(uint)bits[2] << 64);
        int scale = (bits[3] >> 16) & 0xff;
        var units = coefficient * System.Numerics.BigInteger.Pow(10, 28 - scale);
        return bits[3] < 0 ? -units : units;
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
