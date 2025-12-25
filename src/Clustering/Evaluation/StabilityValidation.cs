using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Stability-based validation for evaluating clustering quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stability validation measures how consistent clustering results are across
/// different random subsamples of the data. A good clustering should be stable
/// - similar subsamples should produce similar clusters.
/// </para>
/// <para>
/// Algorithm:
/// 1. Generate multiple random subsamples of the data
/// 2. Cluster each subsample
/// 3. Measure agreement between clusterings on shared points
/// 4. High stability indicates a robust clustering
/// </para>
/// <para><b>For Beginners:</b> Stability validation asks "Is this clustering reliable?"
///
/// If you cluster random samples of your data:
/// - High stability: Same clusters appear each time (good!)
/// - Low stability: Different clusters each time (concerning!)
///
/// A stable clustering is more trustworthy because it's not sensitive to
/// which exact points are included.
/// </para>
/// </remarks>
public class StabilityValidation<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numSubsamples;
    private readonly double _subsampleFraction;
    private readonly int? _randomState;

    /// <summary>
    /// Initializes a new StabilityValidation instance.
    /// </summary>
    /// <param name="numSubsamples">Number of subsamples to generate. Default is 20.</param>
    /// <param name="subsampleFraction">Fraction of data to include in each subsample. Default is 0.8.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    public StabilityValidation(int numSubsamples = 20, double subsampleFraction = 0.8, int? randomState = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numSubsamples = numSubsamples;
        _subsampleFraction = subsampleFraction;
        _randomState = randomState;
    }

    /// <summary>
    /// Evaluates clustering stability for a given number of clusters.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="numClusters">Number of clusters.</param>
    /// <returns>Stability result containing average agreement and per-pair statistics.</returns>
    public StabilityResult Evaluate(Matrix<T> data, int numClusters)
    {
        int n = data.Rows;
        int subsampleSize = Math.Max(2, (int)(n * _subsampleFraction));
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : new Random();

        var allLabels = new List<(int[] Indices, Vector<T> Labels)>();

        // Generate subsamples and cluster each
        for (int s = 0; s < _numSubsamples; s++)
        {
            // Random subsample without replacement
            var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).Take(subsampleSize).ToArray();
            var subsample = ExtractSubMatrix(data, indices);

            // Cluster the subsample
            var kmeans = new KMeans<T>(new KMeansOptions<T>
            {
                NumClusters = numClusters,
                MaxIterations = 100,
                NumInitializations = 3,
                RandomState = _randomState
            });

            try
            {
                kmeans.Train(subsample);
                if (kmeans.Labels is not null)
                {
                    allLabels.Add((indices, kmeans.Labels));
                }
            }
            catch
            {
                // Skip failed clusterings
            }
        }

        if (allLabels.Count < 2)
        {
            return new StabilityResult
            {
                NumClusters = numClusters,
                AverageStability = 0,
                StabilityStdDev = 0,
                NumComparisons = 0
            };
        }

        // Compare all pairs of clusterings
        var stabilityScores = new List<double>();
        var jaccard = new JaccardIndex<T>();

        for (int i = 0; i < allLabels.Count; i++)
        {
            for (int j = i + 1; j < allLabels.Count; j++)
            {
                double score = ComputePairwiseStability(
                    allLabels[i].Indices, allLabels[i].Labels,
                    allLabels[j].Indices, allLabels[j].Labels);
                stabilityScores.Add(score);
            }
        }

        double avgStability = stabilityScores.Average();
        double stdDev = stabilityScores.Count > 1
            ? Math.Sqrt(stabilityScores.Sum(x => Math.Pow(x - avgStability, 2)) / (stabilityScores.Count - 1))
            : 0;

        return new StabilityResult
        {
            NumClusters = numClusters,
            AverageStability = avgStability,
            StabilityStdDev = stdDev,
            NumComparisons = stabilityScores.Count,
            AllScores = stabilityScores.ToArray()
        };
    }

    /// <summary>
    /// Evaluates stability across a range of cluster counts to find optimal K.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="minClusters">Minimum number of clusters.</param>
    /// <param name="maxClusters">Maximum number of clusters.</param>
    /// <returns>Results for each K value with recommended optimal K.</returns>
    public StabilityAnalysisResult EvaluateRange(Matrix<T> data, int minClusters = 2, int maxClusters = 10)
    {
        int n = data.Rows;
        maxClusters = Math.Min(maxClusters, n - 1);

        var results = new List<StabilityResult>();

        for (int k = minClusters; k <= maxClusters; k++)
        {
            var result = Evaluate(data, k);
            results.Add(result);
        }

        // Find optimal K (highest stability)
        int optimalK = minClusters;
        double maxStability = 0;

        foreach (var result in results)
        {
            if (result.AverageStability > maxStability)
            {
                maxStability = result.AverageStability;
                optimalK = result.NumClusters;
            }
        }

        return new StabilityAnalysisResult
        {
            OptimalK = optimalK,
            Results = results.ToArray()
        };
    }

    private double ComputePairwiseStability(int[] indices1, Vector<T> labels1, int[] indices2, Vector<T> labels2)
    {
        // Find shared indices
        var shared1 = new Dictionary<int, int>(); // original index -> position in indices1
        for (int i = 0; i < indices1.Length; i++)
        {
            shared1[indices1[i]] = i;
        }

        var sharedIndices = new List<int>();
        var sharedPositions1 = new List<int>();
        var sharedPositions2 = new List<int>();

        for (int i = 0; i < indices2.Length; i++)
        {
            if (shared1.TryGetValue(indices2[i], out int pos1))
            {
                sharedIndices.Add(indices2[i]);
                sharedPositions1.Add(pos1);
                sharedPositions2.Add(i);
            }
        }

        if (sharedIndices.Count < 2)
        {
            return 0;
        }

        // Build label vectors for shared points
        var sharedLabels1 = new Vector<T>(sharedIndices.Count);
        var sharedLabels2 = new Vector<T>(sharedIndices.Count);

        for (int i = 0; i < sharedIndices.Count; i++)
        {
            sharedLabels1[i] = labels1[sharedPositions1[i]];
            sharedLabels2[i] = labels2[sharedPositions2[i]];
        }

        // Use Adjusted Rand Index for comparison
        var ari = new RandIndex<T>(adjusted: true);
        return ari.Compute(sharedLabels1, sharedLabels2);
    }

    private Matrix<T> ExtractSubMatrix(Matrix<T> data, int[] indices)
    {
        var subMatrix = new Matrix<T>(indices.Length, data.Columns);
        for (int i = 0; i < indices.Length; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                subMatrix[i, j] = data[indices[i], j];
            }
        }
        return subMatrix;
    }
}

/// <summary>
/// Result of stability validation for a single K value.
/// </summary>
public class StabilityResult
{
    /// <summary>
    /// Number of clusters tested.
    /// </summary>
    public int NumClusters { get; set; }

    /// <summary>
    /// Average stability score across all subsample pairs.
    /// </summary>
    public double AverageStability { get; set; }

    /// <summary>
    /// Standard deviation of stability scores.
    /// </summary>
    public double StabilityStdDev { get; set; }

    /// <summary>
    /// Number of pairwise comparisons made.
    /// </summary>
    public int NumComparisons { get; set; }

    /// <summary>
    /// All individual stability scores.
    /// </summary>
    public double[] AllScores { get; set; } = Array.Empty<double>();
}

/// <summary>
/// Result of stability analysis across multiple K values.
/// </summary>
public class StabilityAnalysisResult
{
    /// <summary>
    /// Recommended optimal number of clusters.
    /// </summary>
    public int OptimalK { get; set; }

    /// <summary>
    /// Results for each K value tested.
    /// </summary>
    public StabilityResult[] Results { get; set; } = Array.Empty<StabilityResult>();

    /// <summary>
    /// Gets a summary of the stability analysis.
    /// </summary>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Stability Analysis - Optimal K: {OptimalK}");
        sb.AppendLine();
        sb.AppendLine("K\tStability\tStd Dev");
        sb.AppendLine("---\t---------\t-------");

        foreach (var result in Results)
        {
            string marker = result.NumClusters == OptimalK ? " *" : "";
            sb.AppendLine($"{result.NumClusters}{marker}\t{result.AverageStability:F4}\t\t{result.StabilityStdDev:F4}");
        }

        return sb.ToString();
    }
}
