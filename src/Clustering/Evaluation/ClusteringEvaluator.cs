using AiDotNet.Clustering.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Comprehensive evaluator for clustering results.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ClusteringEvaluator provides a unified interface to compute multiple
/// cluster validity indices and metrics. It supports both internal metrics
/// (using only the data) and external metrics (comparing to ground truth).
/// </para>
/// <para><b>For Beginners:</b> This is your "one-stop shop" for evaluating clusters.
///
/// Instead of calling each metric separately:
/// - Create a ClusteringEvaluator
/// - Call EvaluateAll() to get all metrics at once
/// - Compare different clusterings easily
///
/// The evaluator handles:
/// - Internal validity: How well-structured are the clusters?
/// - External validity: How well do clusters match known labels?
/// - Model selection: Which K is best? Which algorithm works best?
///
/// Use this when:
/// - Comparing different clustering algorithms
/// - Tuning parameters (like number of clusters)
/// - Validating clustering results
/// </para>
/// </remarks>
public class ClusteringEvaluator<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly List<IClusterMetric<T>> _internalMetrics;
    private readonly List<IExternalClusterMetric<T>> _externalMetrics;

    /// <summary>
    /// Initializes a new ClusteringEvaluator with default metrics.
    /// </summary>
    public ClusteringEvaluator()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _internalMetrics = new List<IClusterMetric<T>>
        {
            new SilhouetteScore<T>(),
            new DaviesBouldinIndex<T>(),
            new CalinskiHarabaszIndex<T>(),
            new DunnIndex<T>(),
            new ConnectivityIndex<T>()
        };

        _externalMetrics = new List<IExternalClusterMetric<T>>
        {
            new AdjustedRandIndex<T>(),
            new NormalizedMutualInformation<T>(),
            new VMeasure<T>(),
            new FowlkesMallowsIndex<T>()
        };
    }

    /// <summary>
    /// Adds a custom internal metric to the evaluator.
    /// </summary>
    /// <param name="metric">The metric to add.</param>
    public void AddInternalMetric(IClusterMetric<T> metric)
    {
        _internalMetrics.Add(metric);
    }

    /// <summary>
    /// Adds a custom external metric to the evaluator.
    /// </summary>
    /// <param name="metric">The metric to add.</param>
    public void AddExternalMetric(IExternalClusterMetric<T> metric)
    {
        _externalMetrics.Add(metric);
    }

    /// <summary>
    /// Evaluates all internal metrics for a clustering result.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labels">The cluster assignments.</param>
    /// <returns>Dictionary of metric name to value.</returns>
    public Dictionary<string, double> EvaluateInternal(Matrix<T> data, Vector<T> labels)
    {
        var results = new Dictionary<string, double>();

        foreach (var metric in _internalMetrics)
        {
            try
            {
                double value = metric.Compute(data, labels);
                results[metric.Name] = value;
            }
            catch (Exception)
            {
                results[metric.Name] = double.NaN;
            }
        }

        return results;
    }

    /// <summary>
    /// Evaluates all external metrics for a clustering result.
    /// </summary>
    /// <param name="trueLabels">The ground truth labels.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <returns>Dictionary of metric name to value.</returns>
    public Dictionary<string, double> EvaluateExternal(Vector<T> trueLabels, Vector<T> predictedLabels)
    {
        var results = new Dictionary<string, double>();

        foreach (var metric in _externalMetrics)
        {
            try
            {
                double value = metric.Compute(trueLabels, predictedLabels);
                results[GetMetricName(metric)] = value;
            }
            catch (Exception)
            {
                results[GetMetricName(metric)] = double.NaN;
            }
        }

        return results;
    }

    /// <summary>
    /// Evaluates all metrics for a clustering result.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="predictedLabels">The predicted cluster assignments.</param>
    /// <param name="trueLabels">Optional ground truth labels for external metrics.</param>
    /// <returns>Complete evaluation results.</returns>
    public ClusteringEvaluationResult EvaluateAll(Matrix<T> data, Vector<T> predictedLabels, Vector<T>? trueLabels = null)
    {
        var result = new ClusteringEvaluationResult
        {
            InternalMetrics = EvaluateInternal(data, predictedLabels)
        };

        if (trueLabels is not null)
        {
            result.ExternalMetrics = EvaluateExternal(trueLabels, predictedLabels);
        }

        // Compute summary statistics
        result.NumClusters = CountUniqueClusters(predictedLabels);
        result.NumPoints = predictedLabels.Length;
        result.ClusterSizes = ComputeClusterSizes(predictedLabels, result.NumClusters);

        return result;
    }

    /// <summary>
    /// Compares multiple clustering results and ranks them.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="labelsList">List of different clustering label assignments.</param>
    /// <param name="algorithmNames">Optional names for each algorithm.</param>
    /// <returns>Ranked comparison results.</returns>
    public List<ClusteringComparisonResult> CompareClusterings(
        Matrix<T> data,
        List<Vector<T>> labelsList,
        List<string>? algorithmNames = null)
    {
        var results = new List<ClusteringComparisonResult>();

        for (int i = 0; i < labelsList.Count; i++)
        {
            var labels = labelsList[i];
            var eval = EvaluateAll(data, labels);

            var comparison = new ClusteringComparisonResult
            {
                AlgorithmName = algorithmNames?[i] ?? $"Algorithm {i + 1}",
                NumClusters = eval.NumClusters,
                Evaluation = eval
            };

            // Compute a composite score (higher is better)
            comparison.CompositeScore = ComputeCompositeScore(eval);

            results.Add(comparison);
        }

        // Rank by composite score
        return results.OrderByDescending(r => r.CompositeScore).ToList();
    }

    /// <summary>
    /// Finds the optimal number of clusters using multiple criteria.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="clusteringFunction">Function that takes K and returns labels.</param>
    /// <param name="kRange">Range of K values to test.</param>
    /// <returns>Analysis results for each K.</returns>
    public OptimalKAnalysis FindOptimalK(
        Matrix<T> data,
        Func<int, Vector<T>> clusteringFunction,
        (int min, int max) kRange)
    {
        var analysis = new OptimalKAnalysis
        {
            KRange = kRange,
            Results = new Dictionary<int, Dictionary<string, double>>(),
            Recommendations = new Dictionary<string, int>()
        };

        for (int k = kRange.min; k <= kRange.max; k++)
        {
            var labels = clusteringFunction(k);
            var metrics = EvaluateInternal(data, labels);
            analysis.Results[k] = metrics;
        }

        // Recommend K for each metric
        foreach (var metricName in analysis.Results[kRange.min].Keys)
        {
            var metric = _internalMetrics.FirstOrDefault(m => m.Name == metricName);
            bool higherIsBetter = metric is IClusterMetric<T> m && m is DunnIndex<T> or SilhouetteScore<T> or CalinskiHarabaszIndex<T>;

            int bestK = higherIsBetter
                ? analysis.Results.OrderByDescending(kv => kv.Value[metricName]).First().Key
                : analysis.Results.OrderBy(kv => kv.Value[metricName]).First().Key;

            analysis.Recommendations[metricName] = bestK;
        }

        // Consensus recommendation (most common K)
        analysis.ConsensusK = analysis.Recommendations.Values
            .GroupBy(k => k)
            .OrderByDescending(g => g.Count())
            .First()
            .Key;

        return analysis;
    }

    private int CountUniqueClusters(Vector<T> labels)
    {
        var unique = new HashSet<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                unique.Add(label);
            }
        }
        return unique.Count;
    }

    private int[] ComputeClusterSizes(Vector<T> labels, int numClusters)
    {
        var sizes = new int[numClusters];
        for (int i = 0; i < labels.Length; i++)
        {
            int label = (int)_numOps.ToDouble(labels[i]);
            if (label >= 0 && label < numClusters)
            {
                sizes[label]++;
            }
        }
        return sizes;
    }

    private double ComputeCompositeScore(ClusteringEvaluationResult eval)
    {
        double score = 0;
        int count = 0;

        // Silhouette: -1 to 1, higher is better
        if (eval.InternalMetrics.TryGetValue("Silhouette Score", out double silhouette) && !double.IsNaN(silhouette))
        {
            score += (silhouette + 1) / 2; // Normalize to 0-1
            count++;
        }

        // Davies-Bouldin: 0+, lower is better
        if (eval.InternalMetrics.TryGetValue("Davies-Bouldin Index", out double db) && !double.IsNaN(db))
        {
            score += 1 / (1 + db); // Invert and normalize
            count++;
        }

        // Calinski-Harabasz: 0+, higher is better
        if (eval.InternalMetrics.TryGetValue("Calinski-Harabasz Index", out double ch) && !double.IsNaN(ch))
        {
            score += Math.Min(ch / 1000, 1); // Cap at 1000 and normalize
            count++;
        }

        // Dunn Index: 0+, higher is better
        if (eval.InternalMetrics.TryGetValue("Dunn Index", out double dunn) && !double.IsNaN(dunn))
        {
            score += Math.Min(dunn, 1); // Cap at 1
            count++;
        }

        return count > 0 ? score / count : 0;
    }

    private string GetMetricName(IExternalClusterMetric<T> metric)
    {
        return metric switch
        {
            AdjustedRandIndex<T> => "Adjusted Rand Index",
            NormalizedMutualInformation<T> => "Normalized Mutual Information",
            VMeasure<T> => "V-Measure",
            FowlkesMallowsIndex<T> => "Fowlkes-Mallows Index",
            _ => metric.GetType().Name
        };
    }
}

/// <summary>
/// Results from clustering evaluation.
/// </summary>
public class ClusteringEvaluationResult
{
    /// <summary>
    /// Gets or sets the internal metric values.
    /// </summary>
    public Dictionary<string, double> InternalMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets the external metric values.
    /// </summary>
    public Dictionary<string, double> ExternalMetrics { get; set; } = new();

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    public int NumClusters { get; set; }

    /// <summary>
    /// Gets or sets the number of data points.
    /// </summary>
    public int NumPoints { get; set; }

    /// <summary>
    /// Gets or sets the size of each cluster.
    /// </summary>
    public int[] ClusterSizes { get; set; } = Array.Empty<int>();
}

/// <summary>
/// Results from comparing multiple clusterings.
/// </summary>
public class ClusteringComparisonResult
{
    /// <summary>
    /// Gets or sets the algorithm name.
    /// </summary>
    public string AlgorithmName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of clusters.
    /// </summary>
    public int NumClusters { get; set; }

    /// <summary>
    /// Gets or sets the full evaluation result.
    /// </summary>
    public ClusteringEvaluationResult Evaluation { get; set; } = new();

    /// <summary>
    /// Gets or sets the composite score for ranking.
    /// </summary>
    public double CompositeScore { get; set; }
}

/// <summary>
/// Analysis results for finding optimal K.
/// </summary>
public class OptimalKAnalysis
{
    /// <summary>
    /// Gets or sets the range of K values tested.
    /// </summary>
    public (int min, int max) KRange { get; set; }

    /// <summary>
    /// Gets or sets the metric values for each K.
    /// </summary>
    public Dictionary<int, Dictionary<string, double>> Results { get; set; } = new();

    /// <summary>
    /// Gets or sets the recommended K for each metric.
    /// </summary>
    public Dictionary<string, int> Recommendations { get; set; } = new();

    /// <summary>
    /// Gets or sets the consensus K (most recommended).
    /// </summary>
    public int ConsensusK { get; set; }
}
