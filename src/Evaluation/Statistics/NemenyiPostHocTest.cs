using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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

        if (k < 2 || k > 11)
            throw new ArgumentException("Nemenyi test supports 2-11 algorithms.");

        // Get alpha index
        int alphaIndex = alpha <= 0.01 ? 2 : (alpha <= 0.05 ? 1 : 0);

        // Get q value
        double q = QValues[Math.Min(k - 2, 8), alphaIndex];

        // Calculate critical difference
        double cd = q * Math.Sqrt(k * (k + 1) / (6.0 * n));

        // Calculate average ranks (same as Friedman)
        var ranks = new double[k];
        for (int i = 0; i < n; i++)
        {
            var datasetScores = new double[k];
            for (int j = 0; j < k; j++)
                datasetScores[j] = NumOps.ToDouble(samples[j][i]);

            var datasetRanks = RankWithTies(datasetScores, ascending: false);
            for (int j = 0; j < k; j++)
                ranks[j] += datasetRanks[j];
        }
        for (int j = 0; j < k; j++)
            ranks[j] /= n;

        // Perform pairwise comparisons
        var pairwise = new bool[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                pairwise[i, j] = Math.Abs(ranks[i] - ranks[j]) > cd;
                pairwise[j, i] = pairwise[i, j];
            }
        }

        return new NemenyiResult<T>
        {
            CriticalDifference = cd,
            AverageRanks = ranks,
            SignificantDifferences = pairwise,
            Alpha = alpha,
            NumAlgorithms = k,
            NumDatasets = n
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
            while (j < n - 1 && Math.Abs(indexed[j].value - indexed[j + 1].value) < 1e-10)
                j++;

            double avgRank = (i + j + 2.0) / 2.0;
            for (int m = i; m <= j; m++)
                ranks[indexed[m].index] = avgRank;

            i = j + 1;
        }

        return ranks;
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
    public double[] AverageRanks { get; init; } = Array.Empty<double>();

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
