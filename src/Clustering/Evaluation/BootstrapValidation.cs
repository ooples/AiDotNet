using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Clustering.Evaluation;

/// <summary>
/// Bootstrap validation for evaluating clustering quality and confidence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bootstrap validation uses resampling with replacement to estimate the
/// uncertainty in clustering metrics. It provides confidence intervals for
/// internal validation measures like silhouette score.
/// </para>
/// <para>
/// Algorithm:
/// 1. Create B bootstrap samples (sample with replacement)
/// 2. Cluster each bootstrap sample
/// 3. Compute metrics on each clustering
/// 4. Use the distribution to estimate confidence intervals
/// </para>
/// <para><b>For Beginners:</b> Bootstrap asks "How confident are we in these results?"
///
/// By creating many random samples (with repeats allowed):
/// - We see how much our metrics vary
/// - We can give confidence intervals, not just single numbers
/// - We understand which results are reliable
///
/// If the bootstrap results vary a lot, be cautious about the clustering!
/// </para>
/// </remarks>
public class BootstrapValidation<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numBootstraps;
    private readonly int? _randomState;

    /// <summary>
    /// Initializes a new BootstrapValidation instance.
    /// </summary>
    /// <param name="numBootstraps">Number of bootstrap samples. Default is 100.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    public BootstrapValidation(int numBootstraps = 100, int? randomState = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numBootstraps = numBootstraps;
        _randomState = randomState;
    }

    /// <summary>
    /// Evaluates clustering with bootstrap confidence intervals.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="numClusters">Number of clusters.</param>
    /// <param name="confidenceLevel">Confidence level for intervals. Default is 0.95.</param>
    /// <returns>Bootstrap result with confidence intervals for metrics.</returns>
    public BootstrapResult Evaluate(Matrix<T> data, int numClusters, double confidenceLevel = 0.95)
    {
        int n = data.Rows;
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var silhouetteScores = new List<double>();
        var inertiaValues = new List<double>();
        var daviesBouldinScores = new List<double>();
        var calinskiHarabaszScores = new List<double>();

        var silhouetteMetric = new SilhouetteScore<T>();
        var daviesBouldinMetric = new DaviesBouldinIndex<T>();
        var calinskiHarabaszMetric = new CalinskiHarabaszIndex<T>();

        for (int b = 0; b < _numBootstraps; b++)
        {
            // Bootstrap sample (with replacement)
            var indices = new int[n];
            for (int i = 0; i < n; i++)
            {
                indices[i] = rand.Next(n);
            }

            var bootstrapSample = ExtractSubMatrix(data, indices);

            // Cluster the bootstrap sample
            var kmeans = new KMeans<T>(new KMeansOptions<T>
            {
                NumClusters = numClusters,
                MaxIterations = 100,
                NumInitializations = 3,
                RandomState = _randomState.HasValue ? _randomState.Value + b : null
            });

            try
            {
                kmeans.Train(bootstrapSample);

                if (kmeans.Labels is not null)
                {
                    // Compute metrics
                    double silhouette = silhouetteMetric.Compute(bootstrapSample, kmeans.Labels);
                    silhouetteScores.Add(silhouette);

                    double daviesBouldin = daviesBouldinMetric.Compute(bootstrapSample, kmeans.Labels);
                    daviesBouldinScores.Add(daviesBouldin);

                    double calinskiHarabasz = calinskiHarabaszMetric.Compute(bootstrapSample, kmeans.Labels);
                    calinskiHarabaszScores.Add(calinskiHarabasz);

                    if (kmeans.Inertia is not null)
                    {
                        inertiaValues.Add(_numOps.ToDouble(kmeans.Inertia));
                    }
                }
            }
            catch
            {
                // Skip failed bootstraps
            }
        }

        return new BootstrapResult
        {
            NumClusters = numClusters,
            ConfidenceLevel = confidenceLevel,
            NumBootstraps = _numBootstraps,
            Silhouette = ComputeConfidenceInterval(silhouetteScores, confidenceLevel),
            Inertia = ComputeConfidenceInterval(inertiaValues, confidenceLevel),
            DaviesBouldin = ComputeConfidenceInterval(daviesBouldinScores, confidenceLevel),
            CalinskiHarabasz = ComputeConfidenceInterval(calinskiHarabaszScores, confidenceLevel)
        };
    }

    /// <summary>
    /// Evaluates bootstrap confidence intervals across a range of cluster counts.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="minClusters">Minimum number of clusters.</param>
    /// <param name="maxClusters">Maximum number of clusters.</param>
    /// <param name="confidenceLevel">Confidence level for intervals.</param>
    /// <returns>Results for each K value.</returns>
    public BootstrapAnalysisResult EvaluateRange(Matrix<T> data, int minClusters = 2, int maxClusters = 10, double confidenceLevel = 0.95)
    {
        int n = data.Rows;
        maxClusters = Math.Min(maxClusters, n - 1);

        var results = new List<BootstrapResult>();

        for (int k = minClusters; k <= maxClusters; k++)
        {
            var result = Evaluate(data, k, confidenceLevel);
            results.Add(result);
        }

        // Find optimal K based on silhouette score
        int optimalK = minClusters;
        double maxSilhouette = double.MinValue;

        foreach (var result in results)
        {
            if (result.Silhouette.Mean > maxSilhouette)
            {
                maxSilhouette = result.Silhouette.Mean;
                optimalK = result.NumClusters;
            }
        }

        return new BootstrapAnalysisResult
        {
            OptimalK = optimalK,
            Results = results.ToArray()
        };
    }

    /// <summary>
    /// Evaluates cluster assignment confidence for each point.
    /// </summary>
    /// <param name="data">The data matrix.</param>
    /// <param name="numClusters">Number of clusters.</param>
    /// <returns>Per-point confidence scores.</returns>
    /// <remarks>
    /// <para>
    /// For each data point, computes the fraction of bootstrap runs where
    /// it was assigned to its most common cluster. High values indicate
    /// the point is reliably assigned; low values suggest it's on a boundary.
    /// </para>
    /// </remarks>
    public double[] ComputeAssignmentConfidence(Matrix<T> data, int numClusters)
    {
        int n = data.Rows;
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Track cluster assignments for each original point
        var assignmentCounts = new Dictionary<int, Dictionary<int, int>>();
        var pointAppearances = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            assignmentCounts[i] = new Dictionary<int, int>();
            pointAppearances[i] = 0;
        }

        for (int b = 0; b < _numBootstraps; b++)
        {
            // Bootstrap sample
            var indices = new int[n];
            for (int i = 0; i < n; i++)
            {
                indices[i] = rand.Next(n);
            }

            var bootstrapSample = ExtractSubMatrix(data, indices);

            var kmeans = new KMeans<T>(new KMeansOptions<T>
            {
                NumClusters = numClusters,
                MaxIterations = 100,
                NumInitializations = 3,
                RandomState = _randomState.HasValue ? _randomState.Value + b : null
            });

            try
            {
                kmeans.Train(bootstrapSample);

                if (kmeans.Labels is not null)
                {
                    // Record assignments
                    for (int i = 0; i < n; i++)
                    {
                        int originalIdx = indices[i];
                        int cluster = (int)_numOps.ToDouble(kmeans.Labels[i]);

                        pointAppearances[originalIdx]++;
                        if (!assignmentCounts[originalIdx].ContainsKey(cluster))
                            assignmentCounts[originalIdx][cluster] = 0;
                        assignmentCounts[originalIdx][cluster]++;
                    }
                }
            }
            catch
            {
                // Skip failed bootstraps
            }
        }

        // Compute confidence as fraction assigned to most common cluster
        var confidence = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (pointAppearances[i] > 0 && assignmentCounts[i].Count > 0)
            {
                int maxCount = assignmentCounts[i].Values.Max();
                confidence[i] = (double)maxCount / pointAppearances[i];
            }
            else
            {
                confidence[i] = 0;
            }
        }

        return confidence;
    }

    private ConfidenceInterval ComputeConfidenceInterval(List<double> values, double confidenceLevel)
    {
        if (values.Count == 0)
        {
            return new ConfidenceInterval();
        }

        var sorted = values.OrderBy(x => x).ToList();
        double mean = sorted.Average();
        double stdDev = sorted.Count > 1
            ? Math.Sqrt(sorted.Sum(x => Math.Pow(x - mean, 2)) / (sorted.Count - 1))
            : 0;

        double alpha = 1 - confidenceLevel;
        int lowerIdx = Math.Max(0, (int)(sorted.Count * alpha / 2) - 1);
        int upperIdx = Math.Min(sorted.Count - 1, (int)(sorted.Count * (1 - alpha / 2)));

        return new ConfidenceInterval
        {
            Mean = mean,
            StdDev = stdDev,
            Lower = sorted[lowerIdx],
            Upper = sorted[upperIdx],
            Median = sorted.Count % 2 == 1
                ? sorted[sorted.Count / 2]
                : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2,
            NumSamples = sorted.Count
        };
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
/// A confidence interval for a metric.
/// </summary>
public class ConfidenceInterval
{
    /// <summary>
    /// Mean value across bootstrap samples.
    /// </summary>
    public double Mean { get; set; }

    /// <summary>
    /// Standard deviation across bootstrap samples.
    /// </summary>
    public double StdDev { get; set; }

    /// <summary>
    /// Lower bound of confidence interval.
    /// </summary>
    public double Lower { get; set; }

    /// <summary>
    /// Upper bound of confidence interval.
    /// </summary>
    public double Upper { get; set; }

    /// <summary>
    /// Median value across bootstrap samples.
    /// </summary>
    public double Median { get; set; }

    /// <summary>
    /// Number of valid samples.
    /// </summary>
    public int NumSamples { get; set; }

    /// <inheritdoc />
    public override string ToString()
    {
        return $"{Mean:F4} [{Lower:F4}, {Upper:F4}]";
    }
}

/// <summary>
/// Result of bootstrap validation for a single K value.
/// </summary>
public class BootstrapResult
{
    /// <summary>
    /// Number of clusters tested.
    /// </summary>
    public int NumClusters { get; set; }

    /// <summary>
    /// Confidence level used.
    /// </summary>
    public double ConfidenceLevel { get; set; }

    /// <summary>
    /// Number of bootstrap samples.
    /// </summary>
    public int NumBootstraps { get; set; }

    /// <summary>
    /// Silhouette score confidence interval.
    /// </summary>
    public ConfidenceInterval Silhouette { get; set; } = new ConfidenceInterval();

    /// <summary>
    /// Inertia confidence interval.
    /// </summary>
    public ConfidenceInterval Inertia { get; set; } = new ConfidenceInterval();

    /// <summary>
    /// Davies-Bouldin index confidence interval.
    /// </summary>
    public ConfidenceInterval DaviesBouldin { get; set; } = new ConfidenceInterval();

    /// <summary>
    /// Calinski-Harabasz index confidence interval.
    /// </summary>
    public ConfidenceInterval CalinskiHarabasz { get; set; } = new ConfidenceInterval();
}

/// <summary>
/// Result of bootstrap analysis across multiple K values.
/// </summary>
public class BootstrapAnalysisResult
{
    /// <summary>
    /// Recommended optimal number of clusters.
    /// </summary>
    public int OptimalK { get; set; }

    /// <summary>
    /// Results for each K value tested.
    /// </summary>
    public BootstrapResult[] Results { get; set; } = Array.Empty<BootstrapResult>();

    /// <summary>
    /// Gets a summary of the bootstrap analysis.
    /// </summary>
    public string GetSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Bootstrap Analysis - Optimal K: {OptimalK}");
        sb.AppendLine();
        sb.AppendLine("K\tSilhouette\t\tDavies-Bouldin");
        sb.AppendLine("---\t----------\t\t--------------");

        foreach (var result in Results)
        {
            string marker = result.NumClusters == OptimalK ? " *" : "";
            sb.AppendLine($"{result.NumClusters}{marker}\t{result.Silhouette}\t{result.DaviesBouldin}");
        }

        return sb.ToString();
    }
}
