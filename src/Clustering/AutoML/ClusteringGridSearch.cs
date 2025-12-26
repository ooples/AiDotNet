using AiDotNet.Clustering.Evaluation;
using AiDotNet.Clustering.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Clustering.AutoML;

/// <summary>
/// Grid search for clustering hyperparameter optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ClusteringGridSearch systematically searches through a specified
/// parameter grid to find the optimal hyperparameters for a given
/// clustering algorithm.
/// </para>
/// <para><b>For Beginners:</b> Grid Search tries every combination of parameters.
///
/// Example: For KMeans with K={2,3,4} and init={"random","kmeans++"}
/// Grid Search will try:
/// - K=2, init=random
/// - K=2, init=kmeans++
/// - K=3, init=random
/// - K=3, init=kmeans++
/// - K=4, init=random
/// - K=4, init=kmeans++
///
/// It evaluates each combination and returns the best one.
/// </para>
/// </remarks>
public class ClusteringGridSearch<T>
{
    private readonly ClusteringEvaluator<T> _evaluator;
    private readonly string _primaryMetric;
    private readonly bool _higherIsBetter;

    /// <summary>
    /// Initializes a new ClusteringGridSearch instance.
    /// </summary>
    /// <param name="primaryMetric">The metric to optimize. Default is Silhouette Score.</param>
    /// <param name="higherIsBetter">Whether higher metric values are better. Default is true.</param>
    public ClusteringGridSearch(string primaryMetric = "Silhouette Score", bool higherIsBetter = true)
    {
        _evaluator = new ClusteringEvaluator<T>();
        _primaryMetric = primaryMetric;
        _higherIsBetter = higherIsBetter;
    }

    /// <summary>
    /// Performs grid search over parameter combinations.
    /// </summary>
    /// <param name="data">The data matrix to cluster.</param>
    /// <param name="clusteringFactory">Factory function that creates a clustering algorithm from parameters.</param>
    /// <param name="parameterGrid">Dictionary of parameter names to possible values.</param>
    /// <returns>Grid search results including best parameters.</returns>
    public GridSearchResult<T> Search(
        Matrix<T> data,
        Func<Dictionary<string, object>, IClustering<T>> clusteringFactory,
        Dictionary<string, object[]> parameterGrid)
    {
        var allCombinations = GenerateCombinations(parameterGrid);
        var results = new List<GridSearchTrialResult<T>>();

        foreach (var parameters in allCombinations)
        {
            try
            {
                var algorithm = clusteringFactory(parameters);
                var labels = algorithm.FitPredict(data);

                int numClusters = CountClusters(labels);
                if (numClusters < 2)
                {
                    continue; // Skip trivial clusterings
                }

                var evaluation = _evaluator.EvaluateAll(data, labels);

                double primaryScore = 0;
                if (evaluation.InternalMetrics.TryGetValue(_primaryMetric, out double score))
                {
                    primaryScore = score;
                }

                results.Add(new GridSearchTrialResult<T>
                {
                    Parameters = new Dictionary<string, object>(parameters),
                    Labels = labels,
                    Evaluation = evaluation,
                    PrimaryScore = primaryScore
                });
            }
            catch
            {
                // Skip failed trials
            }
        }

        // Sort by primary metric
        var sortedResults = _higherIsBetter
            ? results.OrderByDescending(r => r.PrimaryScore).ToList()
            : results.OrderBy(r => r.PrimaryScore).ToList();

        return new GridSearchResult<T>
        {
            BestResult = sortedResults.FirstOrDefault(),
            AllTrials = sortedResults,
            TotalCombinations = allCombinations.Count,
            SuccessfulTrials = results.Count,
            PrimaryMetric = _primaryMetric,
            HigherIsBetter = _higherIsBetter
        };
    }

    /// <summary>
    /// Performs grid search with cross-validation for more robust results.
    /// </summary>
    /// <param name="data">The data matrix to cluster.</param>
    /// <param name="clusteringFactory">Factory function that creates a clustering algorithm from parameters.</param>
    /// <param name="parameterGrid">Dictionary of parameter names to possible values.</param>
    /// <param name="numFolds">Number of cross-validation folds.</param>
    /// <returns>Grid search results with cross-validation scores.</returns>
    public GridSearchCVResult<T> SearchCV(
        Matrix<T> data,
        Func<Dictionary<string, object>, IClustering<T>> clusteringFactory,
        Dictionary<string, object[]> parameterGrid,
        int numFolds = 5)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var allCombinations = GenerateCombinations(parameterGrid);
        var results = new List<GridSearchCVTrialResult<T>>();

        // Create fold indices
        int n = data.Rows;
        var foldIndices = CreateFoldIndices(n, numFolds);

        foreach (var parameters in allCombinations)
        {
            var foldScores = new List<double>();

            for (int fold = 0; fold < numFolds; fold++)
            {
                try
                {
                    // Get train indices (all except current fold)
                    var trainIndices = foldIndices
                        .Where((_, i) => i != fold)
                        .SelectMany(x => x)
                        .ToList();

                    // Create training subset
                    var trainData = CreateSubset(data, trainIndices, numOps);

                    var algorithm = clusteringFactory(parameters);
                    var labels = algorithm.FitPredict(trainData);

                    int numClusters = CountClusters(labels);
                    if (numClusters < 2)
                    {
                        continue;
                    }

                    var evaluation = _evaluator.EvaluateAll(trainData, labels);

                    if (evaluation.InternalMetrics.TryGetValue(_primaryMetric, out double score))
                    {
                        foldScores.Add(score);
                    }
                }
                catch
                {
                    // Skip failed folds
                }
            }

            if (foldScores.Count > 0)
            {
                results.Add(new GridSearchCVTrialResult<T>
                {
                    Parameters = new Dictionary<string, object>(parameters),
                    FoldScores = foldScores.ToArray(),
                    MeanScore = foldScores.Average(),
                    StdScore = ComputeStd(foldScores)
                });
            }
        }

        // Sort by mean score
        var sortedResults = _higherIsBetter
            ? results.OrderByDescending(r => r.MeanScore).ToList()
            : results.OrderBy(r => r.MeanScore).ToList();

        return new GridSearchCVResult<T>
        {
            BestResult = sortedResults.FirstOrDefault(),
            AllTrials = sortedResults,
            TotalCombinations = allCombinations.Count,
            SuccessfulTrials = results.Count,
            NumFolds = numFolds,
            PrimaryMetric = _primaryMetric,
            HigherIsBetter = _higherIsBetter
        };
    }

    private List<Dictionary<string, object>> GenerateCombinations(Dictionary<string, object[]> grid)
    {
        var keys = grid.Keys.ToList();
        var combinations = new List<Dictionary<string, object>>();

        GenerateCombinationsRecursive(grid, keys, 0, new Dictionary<string, object>(), combinations);

        return combinations;
    }

    private void GenerateCombinationsRecursive(
        Dictionary<string, object[]> grid,
        List<string> keys,
        int keyIndex,
        Dictionary<string, object> current,
        List<Dictionary<string, object>> results)
    {
        if (keyIndex >= keys.Count)
        {
            results.Add(new Dictionary<string, object>(current));
            return;
        }

        string key = keys[keyIndex];
        foreach (object value in grid[key])
        {
            current[key] = value;
            GenerateCombinationsRecursive(grid, keys, keyIndex + 1, current, results);
        }
    }

    private int CountClusters(Vector<T> labels)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var unique = new HashSet<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            int label = (int)numOps.ToDouble(labels[i]);
            if (label >= 0)
            {
                unique.Add(label);
            }
        }
        return unique.Count;
    }

    private List<List<int>> CreateFoldIndices(int n, int numFolds)
    {
        var folds = new List<List<int>>();
        for (int i = 0; i < numFolds; i++)
        {
            folds.Add(new List<int>());
        }

        for (int i = 0; i < n; i++)
        {
            folds[i % numFolds].Add(i);
        }

        return folds;
    }

    private Matrix<T> CreateSubset(Matrix<T> data, List<int> indices, INumericOperations<T> numOps)
    {
        var subset = new Matrix<T>(indices.Count, data.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                subset[i, j] = data[indices[i], j];
            }
        }
        return subset;
    }

    private double ComputeStd(List<double> values)
    {
        if (values.Count < 2) return 0;

        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / (values.Count - 1));
    }
}

/// <summary>
/// Result from a single grid search trial.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GridSearchTrialResult<T>
{
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
    /// Score on the primary metric.
    /// </summary>
    public double PrimaryScore { get; set; }
}

/// <summary>
/// Complete result from grid search.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GridSearchResult<T>
{
    /// <summary>
    /// The best result found.
    /// </summary>
    public GridSearchTrialResult<T>? BestResult { get; set; }

    /// <summary>
    /// All successful trials, ranked by score.
    /// </summary>
    public List<GridSearchTrialResult<T>> AllTrials { get; set; } = new();

    /// <summary>
    /// Total number of parameter combinations tried.
    /// </summary>
    public int TotalCombinations { get; set; }

    /// <summary>
    /// Number of successful trials.
    /// </summary>
    public int SuccessfulTrials { get; set; }

    /// <summary>
    /// The metric used for optimization.
    /// </summary>
    public string PrimaryMetric { get; set; } = string.Empty;

    /// <summary>
    /// Whether higher metric values are better.
    /// </summary>
    public bool HigherIsBetter { get; set; }
}

/// <summary>
/// Result from a single cross-validation grid search trial.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GridSearchCVTrialResult<T>
{
    /// <summary>
    /// Parameters used for this trial.
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; } = new();

    /// <summary>
    /// Scores for each fold.
    /// </summary>
    public double[] FoldScores { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Mean score across folds.
    /// </summary>
    public double MeanScore { get; set; }

    /// <summary>
    /// Standard deviation of scores across folds.
    /// </summary>
    public double StdScore { get; set; }
}

/// <summary>
/// Complete result from cross-validation grid search.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GridSearchCVResult<T>
{
    /// <summary>
    /// The best result found.
    /// </summary>
    public GridSearchCVTrialResult<T>? BestResult { get; set; }

    /// <summary>
    /// All successful trials, ranked by mean score.
    /// </summary>
    public List<GridSearchCVTrialResult<T>> AllTrials { get; set; } = new();

    /// <summary>
    /// Total number of parameter combinations tried.
    /// </summary>
    public int TotalCombinations { get; set; }

    /// <summary>
    /// Number of successful trials.
    /// </summary>
    public int SuccessfulTrials { get; set; }

    /// <summary>
    /// Number of cross-validation folds.
    /// </summary>
    public int NumFolds { get; set; }

    /// <summary>
    /// The metric used for optimization.
    /// </summary>
    public string PrimaryMetric { get; set; } = string.Empty;

    /// <summary>
    /// Whether higher metric values are better.
    /// </summary>
    public bool HigherIsBetter { get; set; }

    /// <summary>
    /// Gets a summary of the best result.
    /// </summary>
    public string GetSummary()
    {
        if (BestResult is null)
        {
            return "No successful results found.";
        }

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Best Mean {PrimaryMetric}: {BestResult.MeanScore:F4} (+/- {BestResult.StdScore:F4})");
        sb.AppendLine("Best Parameters:");
        foreach (var kvp in BestResult.Parameters)
        {
            sb.AppendLine($"  {kvp.Key}: {kvp.Value}");
        }
        sb.AppendLine($"Tested {SuccessfulTrials}/{TotalCombinations} combinations with {NumFolds}-fold CV");

        return sb.ToString();
    }
}
