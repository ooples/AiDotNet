using AiDotNet.Clustering.AutoK;
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.AutoML;

/// <summary>
/// Automatic machine learning for clustering - selects best algorithm and parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ClusteringAutoML provides automatic selection of clustering algorithms
/// and hyperparameters. It evaluates multiple algorithms with various
/// configurations and returns the best performing model.
/// </para>
/// <para><b>For Beginners:</b> AutoML takes the guesswork out of clustering.
///
/// Instead of manually trying:
/// - Different algorithms (KMeans, DBSCAN, etc.)
/// - Different parameters (K values, epsilon, etc.)
/// - Evaluating results yourself
///
/// AutoML does it all automatically:
/// 1. Tries multiple algorithms
/// 2. Searches parameter spaces
/// 3. Evaluates with multiple metrics
/// 4. Returns the best solution
///
/// Just provide your data and AutoML finds the best clustering!
/// </para>
/// </remarks>
public class ClusteringAutoML<T>
{
    private readonly ClusteringAutoMLOptions _options;
    private readonly ClusteringEvaluator<T> _evaluator;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ClusteringAutoML instance.
    /// </summary>
    /// <param name="options">AutoML configuration options.</param>
    public ClusteringAutoML(ClusteringAutoMLOptions? options = null)
    {
        _options = options ?? new ClusteringAutoMLOptions();
        _evaluator = new ClusteringEvaluator<T>();
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Runs automatic clustering algorithm and parameter selection.
    /// </summary>
    /// <param name="data">The data matrix to cluster.</param>
    /// <returns>The best clustering result found.</returns>
    public ClusteringAutoMLResult<T> Fit(Matrix<T> data)
    {
        var allResults = new List<ClusteringTrialResult<T>>();

        // Try KMeans with different K values
        if (_options.TryKMeans)
        {
            var kmeansResults = TryKMeans(data);
            allResults.AddRange(kmeansResults);
        }

        // Try DBSCAN with different eps values
        if (_options.TryDBSCAN)
        {
            var dbscanResults = TryDBSCAN(data);
            allResults.AddRange(dbscanResults);
        }

        // Try Gaussian Mixture
        if (_options.TryGaussianMixture)
        {
            var gmmResults = TryGaussianMixture(data);
            allResults.AddRange(gmmResults);
        }

        // Try Fuzzy C-Means
        if (_options.TryFuzzyCMeans)
        {
            var fcmResults = TryFuzzyCMeans(data);
            allResults.AddRange(fcmResults);
        }

        // Try Auto-K algorithms (X-Means, G-Means)
        if (_options.TryAutoK)
        {
            var autoKResults = TryAutoK(data);
            allResults.AddRange(autoKResults);
        }

        // Rank results by composite score
        var rankedResults = allResults
            .Where(r => r.Evaluation.NumClusters >= 2)
            .OrderByDescending(r => r.CompositeScore)
            .ToList();

        var bestResult = rankedResults.FirstOrDefault();

        return new ClusteringAutoMLResult<T>
        {
            BestResult = bestResult,
            AllTrials = rankedResults,
            TotalTrials = allResults.Count,
            SuccessfulTrials = rankedResults.Count
        };
    }

    private List<ClusteringTrialResult<T>> TryKMeans(Matrix<T> data)
    {
        var results = new List<ClusteringTrialResult<T>>();

        foreach (int k in _options.KRange)
        {
            try
            {
                var options = new KMeansOptions<T>
                {
                    NumClusters = k,
                    MaxIterations = _options.MaxIterationsPerTrial,
                    Seed = _options.RandomSeed
                };

                var kmeans = new KMeans<T>(options);
                var labels = kmeans.FitPredict(data);
                var evaluation = _evaluator.EvaluateAll(data, labels);

                results.Add(new ClusteringTrialResult<T>
                {
                    AlgorithmName = "KMeans",
                    Parameters = new Dictionary<string, object> { ["K"] = k },
                    Labels = labels,
                    Evaluation = evaluation,
                    CompositeScore = ComputeCompositeScore(evaluation)
                });
            }
            catch
            {
                // Skip failed trials
            }
        }

        return results;
    }

    private List<ClusteringTrialResult<T>> TryDBSCAN(Matrix<T> data)
    {
        var results = new List<ClusteringTrialResult<T>>();

        // Estimate eps range based on data
        var epsValues = EstimateEpsRange(data);

        foreach (double eps in epsValues)
        {
            foreach (int minPts in _options.MinPtsRange)
            {
                try
                {
                    var options = new DBSCANOptions<T>
                    {
                        Epsilon = eps,
                        MinPoints = minPts
                    };

                    var dbscan = new DBSCAN<T>(options);
                    var labels = dbscan.FitPredict(data);

                    // Count unique clusters (excluding noise)
                    int numClusters = CountClusters(labels);
                    if (numClusters < 2) continue;

                    var evaluation = _evaluator.EvaluateAll(data, labels);

                    results.Add(new ClusteringTrialResult<T>
                    {
                        AlgorithmName = "DBSCAN",
                        Parameters = new Dictionary<string, object>
                        {
                            ["Epsilon"] = eps,
                            ["MinPoints"] = minPts
                        },
                        Labels = labels,
                        Evaluation = evaluation,
                        CompositeScore = ComputeCompositeScore(evaluation)
                    });
                }
                catch
                {
                    // Skip failed trials
                }
            }
        }

        return results;
    }

    private List<ClusteringTrialResult<T>> TryGaussianMixture(Matrix<T> data)
    {
        var results = new List<ClusteringTrialResult<T>>();

        foreach (int k in _options.KRange)
        {
            try
            {
                var options = new GMMOptions<T>
                {
                    NumComponents = k,
                    MaxIterations = _options.MaxIterationsPerTrial,
                    Seed = _options.RandomSeed
                };

                var gmm = new GaussianMixtureModel<T>(options);
                var labels = gmm.FitPredict(data);
                var evaluation = _evaluator.EvaluateAll(data, labels);

                results.Add(new ClusteringTrialResult<T>
                {
                    AlgorithmName = "GaussianMixture",
                    Parameters = new Dictionary<string, object> { ["NumComponents"] = k },
                    Labels = labels,
                    Evaluation = evaluation,
                    CompositeScore = ComputeCompositeScore(evaluation)
                });
            }
            catch
            {
                // Skip failed trials
            }
        }

        return results;
    }

    private List<ClusteringTrialResult<T>> TryFuzzyCMeans(Matrix<T> data)
    {
        var results = new List<ClusteringTrialResult<T>>();

        foreach (int k in _options.KRange)
        {
            foreach (double m in _options.FuzzinessRange)
            {
                try
                {
                    var options = new FuzzyCMeansOptions<T>
                    {
                        NumClusters = k,
                        Fuzziness = m,
                        MaxIterations = _options.MaxIterationsPerTrial,
                        Seed = _options.RandomSeed
                    };

                    var fcm = new FuzzyCMeans<T>(options);
                    var labels = fcm.FitPredict(data);
                    var evaluation = _evaluator.EvaluateAll(data, labels);

                    results.Add(new ClusteringTrialResult<T>
                    {
                        AlgorithmName = "FuzzyCMeans",
                        Parameters = new Dictionary<string, object>
                        {
                            ["NumClusters"] = k,
                            ["Fuzziness"] = m
                        },
                        Labels = labels,
                        Evaluation = evaluation,
                        CompositeScore = ComputeCompositeScore(evaluation)
                    });
                }
                catch
                {
                    // Skip failed trials
                }
            }
        }

        return results;
    }

    private List<ClusteringTrialResult<T>> TryAutoK(Matrix<T> data)
    {
        var results = new List<ClusteringTrialResult<T>>();

        // Try X-Means
        try
        {
            var xmeansOptions = new XMeansOptions<T>
            {
                MaxClusters = _options.KRange.Max(),
                Seed = _options.RandomSeed
            };

            var xmeans = new XMeans<T>(xmeansOptions);
            var labels = xmeans.FitPredict(data);
            var evaluation = _evaluator.EvaluateAll(data, labels);

            results.Add(new ClusteringTrialResult<T>
            {
                AlgorithmName = "XMeans",
                Parameters = new Dictionary<string, object>
                {
                    ["MaxClusters"] = _options.KRange.Max(),
                    ["FinalClusters"] = evaluation.NumClusters
                },
                Labels = labels,
                Evaluation = evaluation,
                CompositeScore = ComputeCompositeScore(evaluation)
            });
        }
        catch
        {
            // Skip if fails
        }

        // Try G-Means
        try
        {
            var gmeansOptions = new GMeansOptions<T>
            {
                MaxClusters = _options.KRange.Max(),
                Seed = _options.RandomSeed
            };

            var gmeans = new GMeans<T>(gmeansOptions);
            var labels = gmeans.FitPredict(data);
            var evaluation = _evaluator.EvaluateAll(data, labels);

            results.Add(new ClusteringTrialResult<T>
            {
                AlgorithmName = "GMeans",
                Parameters = new Dictionary<string, object>
                {
                    ["MaxClusters"] = _options.KRange.Max(),
                    ["FinalClusters"] = evaluation.NumClusters
                },
                Labels = labels,
                Evaluation = evaluation,
                CompositeScore = ComputeCompositeScore(evaluation)
            });
        }
        catch
        {
            // Skip if fails
        }

        return results;
    }

    private double[] EstimateEpsRange(Matrix<T> data)
    {
        int n = data.Rows;
        int sampleSize = Math.Min(100, n);

        // Sample distances to estimate scale
        var distances = new List<double>();
        var random = RandomHelper.CreateSeededRandom(_options.RandomSeed ?? 42);

        for (int i = 0; i < sampleSize; i++)
        {
            int idx1 = random.Next(n);
            int idx2 = random.Next(n);
            if (idx1 == idx2) continue;

            double dist = ComputeDistance(data, idx1, idx2);
            distances.Add(dist);
        }

        if (distances.Count == 0)
        {
            return new[] { 0.1, 0.5, 1.0 };
        }

        distances.Sort();

        // Use percentiles to estimate eps range
        double p10 = distances[(int)(distances.Count * 0.1)];
        double p25 = distances[(int)(distances.Count * 0.25)];
        double p50 = distances[(int)(distances.Count * 0.5)];

        return new[] { p10, p25, p50 };
    }

    private double ComputeDistance(Matrix<T> data, int i, int j)
    {
        double sum = 0;
        for (int k = 0; k < data.Columns; k++)
        {
            double diff = _numOps.ToDouble(data[i, k]) - _numOps.ToDouble(data[j, k]);
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private int CountClusters(Vector<T> labels)
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

    private double ComputeCompositeScore(ClusteringEvaluationResult evaluation)
    {
        double score = 0;
        int count = 0;

        // Silhouette: -1 to 1, higher is better
        if (evaluation.InternalMetrics.TryGetValue("Silhouette Score", out double silhouette) && !double.IsNaN(silhouette))
        {
            score += (silhouette + 1) / 2; // Normalize to 0-1
            count++;
        }

        // Davies-Bouldin: 0+, lower is better
        if (evaluation.InternalMetrics.TryGetValue("Davies-Bouldin Index", out double db) && !double.IsNaN(db))
        {
            score += 1 / (1 + db); // Invert and normalize
            count++;
        }

        // Calinski-Harabasz: 0+, higher is better
        if (evaluation.InternalMetrics.TryGetValue("Calinski-Harabasz Index", out double ch) && !double.IsNaN(ch))
        {
            score += Math.Min(ch / 1000, 1); // Cap at 1000 and normalize
            count++;
        }

        // Dunn Index: 0+, higher is better
        if (evaluation.InternalMetrics.TryGetValue("Dunn Index", out double dunn) && !double.IsNaN(dunn))
        {
            score += Math.Min(dunn, 1); // Cap at 1
            count++;
        }

        return count > 0 ? score / count : 0;
    }
}

/// <summary>
/// Configuration options for ClusteringAutoML.
/// </summary>
public class ClusteringAutoMLOptions
{
    /// <summary>
    /// Range of K values to try for K-based algorithms.
    /// </summary>
    public IEnumerable<int> KRange { get; set; } = Enumerable.Range(2, 9); // 2-10

    /// <summary>
    /// Range of MinPoints values for DBSCAN.
    /// </summary>
    public IEnumerable<int> MinPtsRange { get; set; } = new[] { 3, 5, 7 };

    /// <summary>
    /// Range of fuzziness values for Fuzzy C-Means.
    /// </summary>
    public IEnumerable<double> FuzzinessRange { get; set; } = new[] { 1.5, 2.0, 2.5 };

    /// <summary>
    /// Whether to try KMeans algorithm.
    /// </summary>
    public bool TryKMeans { get; set; } = true;

    /// <summary>
    /// Whether to try DBSCAN algorithm.
    /// </summary>
    public bool TryDBSCAN { get; set; } = true;

    /// <summary>
    /// Whether to try Gaussian Mixture algorithm.
    /// </summary>
    public bool TryGaussianMixture { get; set; } = true;

    /// <summary>
    /// Whether to try Fuzzy C-Means algorithm.
    /// </summary>
    public bool TryFuzzyCMeans { get; set; } = true;

    /// <summary>
    /// Whether to try Auto-K algorithms (X-Means, G-Means).
    /// </summary>
    public bool TryAutoK { get; set; } = true;

    /// <summary>
    /// Maximum iterations per trial.
    /// </summary>
    public int MaxIterationsPerTrial { get; set; } = 100;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Result from a single AutoML trial.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ClusteringTrialResult<T>
{
    /// <summary>
    /// Name of the algorithm used.
    /// </summary>
    public string AlgorithmName { get; set; } = string.Empty;

    /// <summary>
    /// Parameters used for this trial.
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; } = new();

    /// <summary>
    /// The cluster labels produced.
    /// </summary>
    public Vector<T> Labels { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Full evaluation results.
    /// </summary>
    public ClusteringEvaluationResult Evaluation { get; set; } = new();

    /// <summary>
    /// Composite score for ranking.
    /// </summary>
    public double CompositeScore { get; set; }
}

/// <summary>
/// Complete result from ClusteringAutoML.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ClusteringAutoMLResult<T>
{
    /// <summary>
    /// The best result found.
    /// </summary>
    public ClusteringTrialResult<T>? BestResult { get; set; }

    /// <summary>
    /// All successful trials, ranked by score.
    /// </summary>
    public List<ClusteringTrialResult<T>> AllTrials { get; set; } = new();

    /// <summary>
    /// Total number of trials attempted.
    /// </summary>
    public int TotalTrials { get; set; }

    /// <summary>
    /// Number of successful trials.
    /// </summary>
    public int SuccessfulTrials { get; set; }

    /// <summary>
    /// Gets a summary of the best result.
    /// </summary>
    public string GetSummary()
    {
        if (BestResult is null)
        {
            return "No successful clustering found.";
        }

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Best Algorithm: {BestResult.AlgorithmName}");
        sb.AppendLine($"Number of Clusters: {BestResult.Evaluation.NumClusters}");
        sb.AppendLine($"Composite Score: {BestResult.CompositeScore:F4}");
        sb.AppendLine("Parameters:");
        foreach (var kvp in BestResult.Parameters)
        {
            sb.AppendLine($"  {kvp.Key}: {kvp.Value}");
        }
        sb.AppendLine("Metrics:");
        foreach (var kvp in BestResult.Evaluation.InternalMetrics)
        {
            sb.AppendLine($"  {kvp.Key}: {kvp.Value:F4}");
        }

        return sb.ToString();
    }
}
