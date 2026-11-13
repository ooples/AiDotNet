using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

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

    /// <inheritdoc />
    public string Name => "Bootstrap Test";
    /// <inheritdoc />
    public string Description => "Non-parametric bootstrap test for comparing two distributions.";
    /// <inheritdoc />
    public bool IsPaired => true;
    /// <inheritdoc />
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

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var sample1 = new Tensor<T>(scores1, new[] { n }).Cast<double>();
        var sample2 = new Tensor<T>(scores2, new[] { n }).Cast<double>();

        double observedDiff = ComputeMeanDifference(sample1, sample2);
        double observedMagnitude = Math.Abs(observedDiff);

        // For paired data, compute pairwise differences
        var differences = engine.TensorSubtract(sample1, sample2);

        // Paired permutation test: under null, signs of differences are random
        // We randomly flip signs to generate null distribution, a block of resamples at a time
        int moreExtreme = 0;
        double count = n;
        int blockSize = BlockSize(n);
        for (int start = 0; start < _numBootstraps; start += blockSize)
        {
            int rows = Math.Min(blockSize, _numBootstraps - start);
            var signs = new int[rows * n];
            for (int s = 0; s < signs.Length; s++)
            {
                // Randomly flip the sign of each difference (with 50% probability)
                signs[s] = random.Next(2) == 0 ? 1 : -1;
            }

            var signed = engine.TensorMultiply(
                new Tensor<int>(signs, new[] { rows, n }).Cast<double>(),
                engine.TensorBroadcastTo(engine.Reshape(differences, new[] { 1, n }), new[] { rows, n }));
            var permMeanDiffs = engine.TensorDivideScalar(engine.ReduceSum(signed, new[] { 1 }, false), count);
            moreExtreme += EvaluationTensorOps<double>.Count(
                EvaluationTensorOps<double>.GreaterOrEqualMask(engine.TensorAbs(permMeanDiffs), observedMagnitude));
        }

        double pValue = (double)(moreExtreme + 1) / (_numBootstraps + 1);

        // Compute effect size (Cohen's d)
        double std1 = ComputeStd(sample1);
        double std2 = ComputeStd(sample2);
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

        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var sample1 = new Tensor<T>(scores1, new[] { n }).Cast<double>();
        var sample2 = new Tensor<T>(scores2, new[] { n }).Cast<double>();
        double nT = n;
        var blocks = new List<Tensor<double>>();
        int blockSize = BlockSize(n);
        for (int start = 0; start < _numBootstraps; start += blockSize)
        {
            int rows = Math.Min(blockSize, _numBootstraps - start);
            var indices = new int[rows * n];
            for (int s = 0; s < indices.Length; s++)
            {
                indices[s] = random.Next(n);
            }

            var flat = new Tensor<int>(indices, new[] { rows * n });
            var shape = new[] { rows, n };
            var mean1 = engine.TensorDivideScalar(
                engine.ReduceSum(engine.Reshape(engine.TensorGather(sample1, flat, 0), shape), new[] { 1 }, false), nT);
            var mean2 = engine.TensorDivideScalar(
                engine.ReduceSum(engine.Reshape(engine.TensorGather(sample2, flat, 0), shape), new[] { 1 }, false), nT);
            blocks.Add(engine.TensorSubtract(mean1, mean2));
        }

        var diffs = blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
        double alpha = 1 - confidenceLevel;
        int lowerIdx = Math.Max(0, (int)(alpha / 2 * _numBootstraps));
        int upperIdx = Math.Min(_numBootstraps - 1, (int)((1 - alpha / 2) * _numBootstraps) - 1);

        var (lower, upper) = EvaluationTensorOps<double>.OrderStatistics(diffs, lowerIdx, upperIdx);
        return (lower, upper);
    }

    // Resamples are processed in blocks of at most about one million drawn values to bound memory.
    private static int BlockSize(int n) => Math.Max(1, (1 << 20) / n);

    private static double ComputeMeanDifference(Tensor<double> scores1, Tensor<double> scores2)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMean(scores1) - engine.TensorMean(scores2);
    }

    private static double ComputeStd(Tensor<double> scores)
    {
        int n = scores.Length;
        if (n < 2) return 0;
        var engine = AiDotNetEngine.Current;
        double sumSq = engine.TensorSumOfSquares(engine.TensorSubtractScalar(scores, engine.TensorMean(scores)));
        return Math.Sqrt(sumSq / (n - 1));
    }
}
