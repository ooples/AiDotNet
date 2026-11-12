using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Nemenyi post-hoc test: pairwise comparisons after Friedman test.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After the Friedman test shows a significant difference exists
/// among multiple algorithms, the Nemenyi test helps identify which specific pairs differ.
/// <list type="bullet">
/// <item>Controls family-wise error rate (FWER)</item>
/// <item>Computes critical difference (CD) for significance</item>
/// <item>Two algorithms are significantly different if |rank_i - rank_j| > CD</item>
/// </list>
/// </para>
/// <para><b>Critical Difference Diagram:</b> The results can be visualized in a CD diagram,
/// a standard visualization in ML literature for algorithm comparison.</para>
/// </remarks>
public class NemenyiPostHocTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Critical values of q_α for Nemenyi test (from Demsar 2006)
    // Index: (numAlgorithms - 2, alphaIndex) where alphaIndex: 0=0.10, 1=0.05, 2=0.01
    private static readonly double[,] QValues = new double[,]
    {
        { 1.960, 2.241, 2.638 }, // k=2
        { 2.343, 2.571, 2.913 }, // k=3
        { 2.569, 2.773, 3.080 }, // k=4
        { 2.728, 2.919, 3.203 }, // k=5
        { 2.850, 3.031, 3.299 }, // k=6
        { 2.949, 3.124, 3.379 }, // k=7
        { 3.031, 3.200, 3.447 }, // k=8
        { 3.102, 3.266, 3.505 }, // k=9
        { 3.164, 3.324, 3.557 }, // k=10
    };

    /// <summary>
    /// Performs Nemenyi post-hoc test on multiple algorithms.
    /// </summary>
    /// <param name="samples">Performance scores: samples[algorithm][dataset].</param>
    /// <param name="alpha">Significance level (0.01, 0.05, or 0.10).</param>
    /// <returns>Critical difference and pairwise comparison results.</returns>
    public NemenyiResult<T> Test(T[][] samples, double alpha = 0.05)
    {
        int k = samples.Length; // Number of algorithms
        int n = samples[0].Length; // Number of datasets

        if (k < 2 || k > 10)
            throw new ArgumentException("Nemenyi test supports 2-10 algorithms (critical values not available for k > 10).");

        // Get alpha index
        int alphaIndex = alpha <= 0.01 ? 2 : (alpha <= 0.05 ? 1 : 0);

        // Get q value (k-2 gives index 0 for k=2, index 8 for k=10)
        double q = QValues[k - 2, alphaIndex];

        // Calculate critical difference
        double cd = q * Math.Sqrt(k * (k + 1) / (6.0 * n));

        // Calculate average ranks (same as Friedman)
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        var results = engine.TensorPermute(EvaluationTensorOps<T>.StackRows(samples, n), new[] { 1, 0 });
        var datasetRanks = EvaluationTensorOps<T>.RowRanksDescending(results, NumOps.FromDouble(1e-10));
        var ranks = engine.TensorDivideScalar(engine.ReduceSum(datasetRanks, new[] { 0 }, false), NumOps.FromDouble(n));

        // Perform pairwise comparisons: |rank_i - rank_j| > CD (the diagonal difference is 0, never above CD)
        var shape = new[] { k, k };
        var rankDifferences = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(ranks, new[] { k, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(ranks, new[] { 1, k }), shape)));
        var significant = engine.TensorGreaterThan(rankDifferences, NumOps.FromDouble(cd)).Cast<bool>().ToArray();
        var pairwise = new bool[k, k];
        Buffer.BlockCopy(significant, 0, pairwise, 0, k * k * sizeof(bool));

        return new NemenyiResult<T>
        {
            CriticalDifference = cd,
            AverageRanks = ranks.ToVector(),
            SignificantDifferences = pairwise,
            Alpha = alpha,
            NumAlgorithms = k,
            NumDatasets = n
        };
    }
}

/// <summary>
/// Results from Nemenyi post-hoc test.
/// </summary>
public class NemenyiResult<T>
{
    /// <summary>
    /// Critical difference for significance at the specified alpha level.
    /// </summary>
    public double CriticalDifference { get; init; }

    /// <summary>
    /// Average ranks for each algorithm.
    /// </summary>
    public Vector<T> AverageRanks { get; init; } = new Vector<T>(0);

    /// <summary>
    /// Matrix of significant differences: true if algorithms i and j differ significantly.
    /// </summary>
    public bool[,] SignificantDifferences { get; init; } = new bool[0, 0];

    /// <summary>
    /// Significance level used.
    /// </summary>
    public double Alpha { get; init; }

    /// <summary>
    /// Number of algorithms compared.
    /// </summary>
    public int NumAlgorithms { get; init; }

    /// <summary>
    /// Number of datasets used.
    /// </summary>
    public int NumDatasets { get; init; }

    /// <summary>
    /// Gets pairs of algorithms that are significantly different.
    /// </summary>
    public IEnumerable<(int AlgoA, int AlgoB)> GetSignificantPairs()
    {
        for (int i = 0; i < NumAlgorithms; i++)
        {
            for (int j = i + 1; j < NumAlgorithms; j++)
            {
                if (SignificantDifferences[i, j])
                    yield return (i, j);
            }
        }
    }

    /// <summary>
    /// Returns a summary of the test results.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            $"Nemenyi Test Results (α = {Alpha})",
            $"Critical Difference: {CriticalDifference:F4}",
            "Average Ranks:"
        };

        for (int i = 0; i < NumAlgorithms; i++)
            lines.Add($"  Algorithm {i + 1}: {AverageRanks[i]:F4}");

        var sigPairs = GetSignificantPairs().ToList();
        if (sigPairs.Count > 0)
        {
            lines.Add("Significant differences:");
            foreach (var (a, b) in sigPairs)
                lines.Add($"  Algorithms {a + 1} vs {b + 1}");
        }
        else
        {
            lines.Add("No significant pairwise differences found.");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
