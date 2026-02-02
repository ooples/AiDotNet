using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Bootstrap-based hypothesis test for comparing two models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Bootstrap testing uses resampling to test significance:
/// <list type="bullet">
/// <item>Repeatedly samples with replacement from your data</item>
/// <item>Computes the statistic of interest on each sample</item>
/// <item>Builds a distribution to test your hypothesis</item>
/// <item>No assumptions about the underlying distribution</item>
/// </list>
/// </para>
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Works for any metric, not just specific test statistics</item>
/// <item>No distributional assumptions</item>
/// <item>Can compute confidence intervals for any statistic</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BootstrapTest<T> : IStatisticalTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numBootstraps;
    private readonly int? _randomSeed;

    public string Name => "Bootstrap Test";
    public string Description => "Non-parametric bootstrap test for comparing two distributions.";
    public bool IsPaired => true;
    public bool IsNonParametric => true;

    /// <summary>
    /// Initializes the bootstrap test.
    /// </summary>
    /// <param name="numBootstraps">Number of bootstrap samples. Default: 10000.</param>
    /// <param name="randomSeed">Optional random seed for reproducibility.</param>
    public BootstrapTest(int numBootstraps = 10000, int? randomSeed = null)
    {
        if (numBootstraps <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBootstraps), "Number of bootstraps must be positive.");
        _numBootstraps = numBootstraps;
        _randomSeed = randomSeed;
    }

    /// <summary>
    /// Tests if two sets of scores differ significantly using bootstrap.
    /// </summary>
    /// <param name="scores1">Performance scores from first model.</param>
    /// <param name="scores2">Performance scores from second model.</param>
    /// <returns>Statistical test result with p-value.</returns>
    public StatisticalTestResult<T> Test(T[] scores1, T[] scores2)
    {
        if (scores1.Length != scores2.Length)
            throw new ArgumentException("Score arrays must have the same length.");

        int n = scores1.Length;
        if (n < 2)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                EffectSize = NumOps.Zero,
                Description = "Not enough samples for bootstrap test."
            };
        }

        var random = _randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Compute observed difference
        double observedDiff = ComputeMeanDifference(scores1, scores2);

        // For paired data, compute pairwise differences
        var differences = new double[n];
        for (int i = 0; i < n; i++)
        {
            differences[i] = NumOps.ToDouble(scores1[i]) - NumOps.ToDouble(scores2[i]);
        }

        // Paired permutation test: under null, signs of differences are random
        // We randomly flip signs to generate null distribution
        int moreExtreme = 0;
        for (int b = 0; b < _numBootstraps; b++)
        {
            double permMeanDiff = 0;
            for (int i = 0; i < n; i++)
            {
                // Randomly flip the sign of each difference (with 50% probability)
                double sign = random.Next(2) == 0 ? 1.0 : -1.0;
                permMeanDiff += sign * differences[i];
            }
            permMeanDiff /= n;

            if (Math.Abs(permMeanDiff) >= Math.Abs(observedDiff))
                moreExtreme++;
        }

        double pValue = (double)(moreExtreme + 1) / (_numBootstraps + 1);

        // Compute effect size (Cohen's d)
        double std1 = ComputeStd(scores1);
        double std2 = ComputeStd(scores2);
        double pooledStd = Math.Sqrt((std1 * std1 + std2 * std2) / 2);
        double cohensD = pooledStd > 1e-10 ? observedDiff / pooledStd : 0;

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(observedDiff),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < 0.05,
            EffectSize = NumOps.FromDouble(cohensD),
            Interpretation = pValue < 0.05
                ? $"Significant difference (Δ={observedDiff:F4}, Cohen's d={cohensD:F2})"
                : $"No significant difference (Δ={observedDiff:F4})",
            Description = $"Bootstrap permutation test with {_numBootstraps} resamples."
        };
    }

    /// <summary>
    /// Computes bootstrap confidence interval for the mean difference.
    /// </summary>
    public (double Lower, double Upper) ComputeCI(T[] scores1, T[] scores2, double confidenceLevel = 0.95)
    {
        if (scores1.Length != scores2.Length)
            throw new ArgumentException("Score arrays must have the same length.");
        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentOutOfRangeException(nameof(confidenceLevel), "Confidence level must be between 0 and 1 (exclusive).");

        int n = scores1.Length;
        if (n == 0) return (0, 0);

        var random = _randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        var diffs = new double[_numBootstraps];
        for (int b = 0; b < _numBootstraps; b++)
        {
            double sum1 = 0, sum2 = 0;
            for (int i = 0; i < n; i++)
            {
                int idx = random.Next(n);
                sum1 += NumOps.ToDouble(scores1[idx]);
                sum2 += NumOps.ToDouble(scores2[idx]);
            }
            diffs[b] = sum1 / n - sum2 / n;
        }

        Array.Sort(diffs);
        double alpha = 1 - confidenceLevel;
        int lowerIdx = Math.Max(0, (int)(alpha / 2 * _numBootstraps));
        int upperIdx = Math.Min(_numBootstraps - 1, (int)((1 - alpha / 2) * _numBootstraps) - 1);

        return (diffs[lowerIdx], diffs[upperIdx]);
    }

    private double ComputeMeanDifference(T[] scores1, T[] scores2)
    {
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < scores1.Length; i++)
        {
            sum1 += NumOps.ToDouble(scores1[i]);
            sum2 += NumOps.ToDouble(scores2[i]);
        }
        return sum1 / scores1.Length - sum2 / scores2.Length;
    }

    private double ComputeStd(T[] scores)
    {
        int n = scores.Length;
        if (n < 2) return 0;
        double mean = scores.Select(s => NumOps.ToDouble(s)).Average();
        double sumSq = scores.Select(s => NumOps.ToDouble(s)).Sum(x => (x - mean) * (x - mean));
        return Math.Sqrt(sumSq / (n - 1));
    }
}
