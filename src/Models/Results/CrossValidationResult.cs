namespace AiDotNet.Models.Results;


/// <summary>
/// Aggregates results from all folds in a cross-validation procedure.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cross-validation helps you understand how well your model will perform
/// on new data by testing it on several different train/test splits. This class combines
/// the results from all those tests to give you an overall picture of your model's performance.
/// </para>
/// </remarks>
public class CrossValidationResult<T>
{
    /// <summary>
    /// Gets the individual results for each fold.
    /// </summary>
    public List<FoldResult<T>> FoldResults { get; }

    /// <summary>
    /// Gets the number of folds used in cross-validation.
    /// </summary>
    public int FoldCount => FoldResults.Count;

    /// <summary>
    /// Gets basic statistics (mean, standard deviation, etc.) for R² values across folds.
    /// </summary>
    public BasicStats<T> R2Stats { get; }

    /// <summary>
    /// Gets basic statistics for RMSE values across folds.
    /// </summary>
    public BasicStats<T> RMSEStats { get; }

    /// <summary>
    /// Gets basic statistics for MAE values across folds.
    /// </summary>
    public BasicStats<T> MAEStats { get; }

    /// <summary>
    /// Gets a dictionary of feature importance scores aggregated across all folds.
    /// </summary>
    public Dictionary<string, BasicStats<T>> FeatureImportanceStats { get; }

    /// <summary>
    /// Gets the average time taken to train the model across all folds.
    /// </summary>
    public TimeSpan AverageTrainingTime { get; }

    /// <summary>
    /// Gets the total time taken for the entire cross-validation process.
    /// </summary>
    public TimeSpan TotalTime { get; }

    private readonly ModelType _modelType = default!;

    /// <summary>
    /// Creates a new instance of the CrossValidationResult class.
    /// </summary>
    /// <param name="foldResults">The results from each individual fold.</param>
    /// <param name="totalTime">The total time taken for the entire cross-validation process.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes the results from each fold of cross-validation
    /// and calculates summary statistics to give you an overall view of how well your model performs.
    /// It helps you understand both the average performance and how consistent that performance is
    /// across different subsets of your data.
    /// </para>
    /// </remarks>
    public CrossValidationResult(List<FoldResult<T>> foldResults, TimeSpan totalTime, ModelType modelType)
    {
        FoldResults = foldResults;
        TotalTime = totalTime;
        _modelType = modelType;

        // Calculate average training time
        AverageTrainingTime = TimeSpan.FromTicks(
            (long)foldResults.Average(r => r.TrainingTime.Ticks)
        );

        // Gather metrics from all folds
        var r2Values = new Vector<T>([.. foldResults.Select(r => r.ValidationPredictionStats.R2)]);
        var rmseValues = new Vector<T>([.. foldResults.Select(r => r.ValidationErrors.RMSE)]);
        var maeValues = new Vector<T>([.. foldResults.Select(r => r.ValidationErrors.MAE)]);

        // Use our existing BasicStats class to calculate statistics across folds
        R2Stats = new BasicStats<T>(r2Values, _modelType);
        RMSEStats = new BasicStats<T>(rmseValues, _modelType);
        MAEStats = new BasicStats<T>(maeValues, _modelType);

        // Aggregate feature importance scores across folds
        FeatureImportanceStats = AggregateFeatureImportance(foldResults);
    }

    /// <summary>
    /// Combines feature importance scores from all folds and calculates statistics.
    /// </summary>
    /// <param name="foldResults">The results from each individual fold.</param>
    /// <returns>A dictionary mapping feature names to their importance statistics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes the feature importance scores from each fold
    /// and calculates statistics like mean and standard deviation. This helps you understand
    /// which features are consistently important across different subsets of your data.
    /// </para>
    /// </remarks>
    private Dictionary<string, BasicStats<T>> AggregateFeatureImportance(List<FoldResult<T>> foldResults)
    {
        var result = new Dictionary<string, BasicStats<T>>();

        // Get all unique feature names from all folds
        var allFeatures = foldResults
            .SelectMany(r => r.FeatureImportance.Keys)
            .Distinct()
            .ToList();

        // For each feature, gather its importance scores from all folds
        foreach (var feature in allFeatures)
        {
            var values = foldResults
                .Where(r => r.FeatureImportance.ContainsKey(feature))
                .Select(r => r.FeatureImportance[feature])
                .ToArray();

            // Calculate statistics for this feature using our existing BasicStats class
            result[feature] = new BasicStats<T>(values, _modelType);
        }

        return result;
    }

    /// <summary>
    /// Gets summary statistics for a specific metric across all folds.
    /// </summary>
    /// <param name="metricType">The type of the metric to analyze.</param>
    /// <returns>Basic statistics for the specified metric.</returns>
    /// <exception cref="ArgumentException">Thrown when an invalid metric type is provided.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you get statistics (mean, standard deviation, etc.)
    /// for any metric across all folds. It helps you understand not just the average performance,
    /// but also how consistent that performance is across different data splits.
    /// </para>
    /// </remarks>
    public BasicStats<T> GetMetricStats(MetricType metricType)
    {
        try
        {
            var values = FoldResults.Select(fold => 
            {
                // Try to get the metric from ValidationPredictionStats
                if (fold.ValidationPredictionStats.IsValidMetric(metricType))
                    return fold.ValidationPredictionStats.GetMetric(metricType);
            
                // If not found, try to get it from ValidationErrors
                if (fold.ValidationErrors.IsValidMetric(metricType))
                    return fold.ValidationErrors.GetMetric(metricType);
            
                // If still not found, throw an exception
                throw new ArgumentException($"Metric '{metricType}' not found in fold results", nameof(metricType));
            }).ToArray();

            return new BasicStats<T>(values, _modelType);
        }
        catch (ArgumentException ex)
        {
            throw new ArgumentException($"Error retrieving metric '{metricType}': {ex.Message}", nameof(metricType), ex);
        }
    }

    /// <summary>
    /// Generates a comprehensive summary report of the cross-validation results.
    /// </summary>
    /// <returns>A string containing the summary report.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a human-readable report that summarizes
    /// how well your model performed across all folds of cross-validation. It includes
    /// key metrics, their variability, and feature importance information.
    /// </para>
    /// </remarks>
    public string GenerateReport()
    {
        var report = new StringBuilder();

        report.AppendLine("Cross-Validation Summary Report");
        report.AppendLine("==============================");
        report.AppendLine($"Number of Folds: {FoldCount}");
        report.AppendLine($"Total Time: {TotalTime.TotalSeconds:F2} seconds");
        report.AppendLine($"Average Training Time: {AverageTrainingTime.TotalSeconds:F2} seconds");
        report.AppendLine();

        report.AppendLine("Performance Metrics (Mean ± Standard Deviation):");
        report.AppendLine($"R² Score: {R2Stats.Mean} ± {R2Stats.StandardDeviation}");
        report.AppendLine($"RMSE: {RMSEStats.Mean} ± {RMSEStats.StandardDeviation}");
        report.AppendLine($"MAE: {MAEStats.Mean} ± {MAEStats.StandardDeviation}");
        report.AppendLine();

        // Add other metrics that might be of interest
        try
        {
            var mapeStats = GetMetricStats(MetricType.MAPE);
            report.AppendLine($"MAPE: {mapeStats.Mean} ± {mapeStats.StandardDeviation}");
        }
        catch (ArgumentException)
        {
            // Metric not available, skip
        }

        // Add feature importance if available
        if (FeatureImportanceStats.Count > 0)
        {
            report.AppendLine("Feature Importance (Mean ± Standard Deviation):");

            // Sort features by mean importance (descending)
            var sortedFeatures = FeatureImportanceStats
                .OrderByDescending(kv => kv.Value.Mean)
                .ToList();

            foreach (var kvp in sortedFeatures)
            {
                var feature = kvp.Key;
                var stats = kvp.Value;
                report.AppendLine($"- {feature}: {stats.Mean:F4} ± {stats.StandardDeviation:F4}");
            }
        }

        return report.ToString();
    }
}