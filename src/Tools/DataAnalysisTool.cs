using System.Linq;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Tools;

/// <summary>
/// A specialized tool that performs comprehensive statistical analysis on datasets to identify patterns,
/// anomalies, distributions, correlations, and data quality issues.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with the ability to analyze datasets and generate detailed statistical insights.
/// It examines data distributions, detects outliers, identifies missing values, calculates feature correlations,
/// checks for class imbalances, and assesses overall data quality. The analysis helps agents make informed
/// recommendations about preprocessing steps, model selection, and potential issues that could affect model performance.
/// </para>
/// <para><b>For Beginners:</b> This tool is like having a data scientist examine your dataset and tell you everything
/// important about it.
///
/// What it analyzes:
/// - **Distributions**: How your data values are spread out (normal, skewed, uniform, etc.)
/// - **Outliers**: Data points that are unusually high or low compared to the rest
/// - **Missing Values**: Gaps in your data where values are absent
/// - **Correlations**: How strongly different features relate to each other
/// - **Class Balance**: Whether you have roughly equal amounts of each category (for classification)
/// - **Data Quality**: Overall assessment of potential issues
///
/// Why this matters:
/// - Outliers can skew your model's predictions
/// - Missing values need to be handled before training
/// - Highly correlated features might cause multicollinearity problems
/// - Imbalanced classes require special handling techniques
/// - Understanding distributions helps choose appropriate preprocessing
///
/// Example input (JSON format):
/// <code>
/// {
///   "dataset_info": {
///     "n_samples": 1000,
///     "n_features": 10,
///     "feature_names": ["age", "income", "score"],
///     "target_type": "continuous"
///   },
///   "statistics": {
///     "age": {"mean": 35.5, "std": 12.3, "min": 18, "max": 75, "missing_pct": 0.05},
///     "income": {"mean": 65000, "std": 25000, "min": 20000, "max": 200000, "missing_pct": 0.0},
///     "score": {"mean": 7.2, "std": 1.8, "min": 1, "max": 10, "missing_pct": 0.02}
///   }
/// }
/// </code>
///
/// Example output:
/// "Dataset Analysis:\n\n" +
/// "Sample Size: 1,000 samples with 10 features - adequate for basic models, may be limited for deep learning.\n\n" +
/// "Missing Values: 'age' has 5% missing values - consider imputation or removal. 'score' has 2% missing.\n\n" +
/// "Distributions: 'income' shows right skew with potential outliers above $200k - consider log transformation.\n\n" +
/// "Recommendations: Handle missing values before training. Apply log transform to 'income'. Check for outliers in high-value income records."
///
/// This analysis gives the AI agent (and you) crucial insights about data quality and preprocessing needs.
/// </para>
/// </remarks>
public class DataAnalysisTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "DataAnalysisTool";

    /// <inheritdoc/>
    public override string Description =>
        "Performs comprehensive statistical analysis on datasets. " +
        "Input should be a JSON object containing dataset information and statistics: " +
        "{ \"dataset_info\": { \"n_samples\": number, \"n_features\": number, \"feature_names\": [strings], " +
        "\"target_type\": \"continuous|categorical\" }, \"statistics\": { \"feature_name\": { \"mean\": number, " +
        "\"std\": number, \"min\": number, \"max\": number, \"missing_pct\": number } }, " +
        "\"correlations\" (optional): { \"feature1\": { \"feature2\": correlation_value } }, " +
        "\"class_distribution\" (optional): { \"class_name\": sample_count } }. " +
        "Returns detailed analysis including distribution characteristics, outlier detection, missing value assessment, " +
        "correlation insights, class imbalance detection, and data quality recommendations.";

    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        try
        {
            // Parse the input JSON
            var root = JObject.Parse(input);

            // Extract dataset information
            var datasetInfo = root["dataset_info"];
            if (datasetInfo == null)
            {
                return "Error: Missing 'dataset_info' property in input JSON.";
            }

            int nSamples = TryGetInt(datasetInfo, "n_samples", 0);
            int nFeatures = TryGetInt(datasetInfo, "n_features", 0);
            string targetType = TryGetString(datasetInfo, "target_type", "unknown");

            // Build analysis report
            var analysis = new System.Text.StringBuilder();
            analysis.AppendLine("=== DATASET ANALYSIS REPORT ===\n");

            // Sample size analysis
            analysis.AppendLine($"**Sample Size Analysis:**");
            analysis.AppendLine($"- Total samples: {nSamples:N0}");
            analysis.AppendLine($"- Total features: {nFeatures}");
            analysis.AppendLine($"- Target type: {targetType}");

            if (nSamples < 100)
            {
                analysis.AppendLine($"- ⚠️ WARNING: Very small dataset ({nSamples} samples). Risk of overfitting is HIGH.");
                analysis.AppendLine($"  Recommendation: Use simple models (linear/logistic regression), apply strong regularization, or collect more data.");
            }
            else if (nSamples < 1000)
            {
                analysis.AppendLine($"- NOTICE: Small dataset ({nSamples} samples). Suitable for basic models.");
                analysis.AppendLine($"  Recommendation: Avoid complex models like deep neural networks. Consider ensemble methods with regularization.");
            }
            else if (nSamples < 10000)
            {
                analysis.AppendLine($"- GOOD: Moderate dataset size ({nSamples} samples). Suitable for most standard ML algorithms.");
                analysis.AppendLine($"  Recommendation: Tree-based ensembles (Random Forest, Gradient Boosting) should work well.");
            }
            else
            {
                analysis.AppendLine($"- EXCELLENT: Large dataset ({nSamples} samples). Can support complex models.");
                analysis.AppendLine($"  Recommendation: Consider neural networks, deep learning, or large ensemble models.");
            }

            analysis.AppendLine();

            // Feature statistics analysis
            var statistics = root["statistics"] as JObject;
            if (statistics != null)
            {
                analysis.AppendLine("**Feature-Level Analysis:**");

                var missingFeatures = new List<string>();
                var skewedFeatures = new List<string>();
                var outlierFeatures = new List<string>();

                foreach (var feature in statistics.Properties())
                {
                    string featureName = feature.Name;
                    var stats = feature.Value as JObject;
                    if (stats == null) continue;

                    // Missing values check
                    double? missingPct = stats["missing_pct"]?.ToObject<double>();
                    if (missingPct.HasValue && missingPct.Value > 0)
                    {
                        missingFeatures.Add($"{featureName} ({missingPct:P1})");

                        if (missingPct.Value > 0.3)
                        {
                            analysis.AppendLine($"- ⚠️ CRITICAL: '{featureName}' has {missingPct:P1} missing values!");
                            analysis.AppendLine($"  Recommendation: Consider removing this feature or using advanced imputation techniques.");
                        }
                        else if (missingPct.Value > 0.05)
                        {
                            analysis.AppendLine($"- ⚠️ WARNING: '{featureName}' has {missingPct:P1} missing values.");
                            analysis.AppendLine($"  Recommendation: Apply imputation (mean/median for numeric, mode for categorical).");
                        }
                    }

                    // Skewness and outlier detection (simple heuristic)
                    double? mean = stats["mean"]?.ToObject<double>();
                    double? std = stats["std"]?.ToObject<double>();
                    double? min = stats["min"]?.ToObject<double>();
                    double? max = stats["max"]?.ToObject<double>();

                    if (mean.HasValue && std.HasValue && min.HasValue && max.HasValue)
                    {
                        // Check for potential outliers (values beyond 3 standard deviations)
                        double lowerBound = mean.Value - 3 * std.Value;
                        double upperBound = mean.Value + 3 * std.Value;

                        if (min.Value < lowerBound || max.Value > upperBound)
                        {
                            outlierFeatures.Add(featureName);
                            analysis.AppendLine($"- NOTICE: '{featureName}' may contain outliers (range: {min:F2} to {max:F2}, mean: {mean:F2}, std: {std:F2}).");
                            analysis.AppendLine($"  Recommendation: Investigate extreme values, consider clipping or robust scaling.");
                        }

                        // Check for skewness (simple heuristic: if median << mean or median >> mean)
                        // Since we don't have median, use a rough approximation
                        double range = max.Value - min.Value;
                        if (range > 0)
                        {
                            double normalizedMean = (mean.Value - min.Value) / range;
                            if (normalizedMean < 0.3 || normalizedMean > 0.7)
                            {
                                skewedFeatures.Add(featureName);
                                analysis.AppendLine($"- NOTICE: '{featureName}' appears skewed (may benefit from transformation).");
                                analysis.AppendLine($"  Recommendation: Try log transformation, square root, or Box-Cox transformation.");
                            }
                        }
                    }
                }

                analysis.AppendLine();
            }
            else
            {
                analysis.AppendLine("**Feature-Level Analysis:**");
                analysis.AppendLine("No detailed feature statistics provided. Cannot perform detailed analysis.");
                analysis.AppendLine();
            }

            // Correlation analysis
            var correlations = root["correlations"] as JObject;
            if (correlations != null)
            {
                analysis.AppendLine("**Correlation Analysis:**");

                var highCorrelations = new List<(string Feature1, string Feature2, double Correlation)>();
                var moderateCorrelations = new List<(string Feature1, string Feature2, double Correlation)>();

                foreach (var featureProp in correlations.Properties())
                {
                    string feature1 = featureProp.Name;
                    var corrPairs = featureProp.Value as JObject;
                    if (corrPairs == null) continue;

                    foreach (var corrProp in corrPairs.Properties())
                    {
                        string feature2 = corrProp.Name;
                        double? correlation = corrProp.Value?.ToObject<double>();

                        if (!correlation.HasValue) continue;

                        double absCorr = Math.Abs(correlation.Value);

                        // Avoid duplicate pairs (only add if feature1 < feature2 alphabetically)
                        if (string.CompareOrdinal(feature1, feature2) < 0)
                        {
                            if (absCorr >= 0.7)
                            {
                                highCorrelations.Add((feature1, feature2, correlation.Value));
                            }
                            else if (absCorr >= 0.5)
                            {
                                moderateCorrelations.Add((feature1, feature2, correlation.Value));
                            }
                        }
                    }
                }

                if (highCorrelations.Count > 0)
                {
                    analysis.AppendLine("⚠️ **HIGH CORRELATIONS DETECTED** (|r| >= 0.7, may cause multicollinearity):");
                    foreach (var (f1, f2, corr) in highCorrelations.OrderByDescending(x => Math.Abs(x.Correlation)))
                    {
                        analysis.AppendLine($"  • **{f1}** ↔ **{f2}**: r = {corr:F3}");
                        analysis.AppendLine($"    → Recommendation: Consider removing one feature or using dimensionality reduction (PCA)");
                    }
                    analysis.AppendLine();
                }

                if (moderateCorrelations.Count > 0)
                {
                    analysis.AppendLine("MODERATE CORRELATIONS (0.5 <= |r| < 0.7):");
                    foreach (var (f1, f2, corr) in moderateCorrelations.OrderByDescending(x => Math.Abs(x.Correlation)))
                    {
                        analysis.AppendLine($"  • {f1} ↔ {f2}: r = {corr:F3}");
                    }
                    analysis.AppendLine();
                }

                if (highCorrelations.Count == 0 && moderateCorrelations.Count == 0)
                {
                    analysis.AppendLine("✓ No significant feature correlations detected (all |r| < 0.5).");
                    analysis.AppendLine("  → Features appear independent, good for linear models.");
                    analysis.AppendLine();
                }
            }

            // Class imbalance analysis
            var classDistribution = root["class_distribution"] as JObject;
            if (classDistribution != null && targetType.Equals("categorical", StringComparison.OrdinalIgnoreCase))
            {
                analysis.AppendLine("**Class Imbalance Analysis:**");

                var classCounts = new Dictionary<string, int>();
                int totalSamples = 0;

                foreach (var classProp in classDistribution.Properties())
                {
                    string className = classProp.Name;
                    int? count = classProp.Value?.ToObject<int>();
                    if (count.HasValue)
                    {
                        classCounts[className] = count.Value;
                        totalSamples += count.Value;
                    }
                }

                if (classCounts.Count > 0)
                {
                    analysis.AppendLine($"- Total classes: {classCounts.Count}");
                    analysis.AppendLine($"- Total samples: {totalSamples:N0}");
                    analysis.AppendLine();
                    analysis.AppendLine("Class distribution:");

                    int maxCount = classCounts.Values.Max();
                    int minCount = classCounts.Values.Min();

                    foreach (var kvp in classCounts.OrderByDescending(x => x.Value))
                    {
                        double percentage = (double)kvp.Value / totalSamples * 100;
                        analysis.AppendLine($"  • {kvp.Key}: {kvp.Value:N0} samples ({percentage:F1}%)");
                    }

                    analysis.AppendLine();

                    // Guard against division by zero if a class has 0 samples
                    if (minCount == 0)
                    {
                        analysis.AppendLine("⚠️ **CRITICAL: One or more classes have 0 samples!**");
                        analysis.AppendLine("  → This indicates a data issue - all classes must have at least one sample");
                        analysis.AppendLine("  → Remove classes with 0 samples before training");
                    }
                    else
                    {
                        // Calculate and analyze imbalance ratio
                        double imbalanceRatio = (double)maxCount / minCount;

                        if (imbalanceRatio >= 10.0)
                        {
                            analysis.AppendLine($"⚠️ **SEVERE CLASS IMBALANCE** (ratio: {imbalanceRatio:F1}:1)");
                            analysis.AppendLine("  → Recommendation: Use SMOTE, class weights, or undersampling/oversampling techniques");
                            analysis.AppendLine("  → Consider using stratified cross-validation");
                            analysis.AppendLine("  → Evaluation metrics: Use F1-score, precision, recall instead of accuracy");
                        }
                        else if (imbalanceRatio >= 3.0)
                        {
                            analysis.AppendLine($"⚠️ **MODERATE CLASS IMBALANCE** (ratio: {imbalanceRatio:F1}:1)");
                            analysis.AppendLine("  → Recommendation: Consider using class weights in your model");
                            analysis.AppendLine("  → Use stratified cross-validation");
                        }
                        else
                        {
                            analysis.AppendLine($"✓ Classes are reasonably balanced (ratio: {imbalanceRatio:F1}:1)");
                        }
                    }

                    analysis.AppendLine();
                }
            }

            if (statistics != null)
            {
                // Summary recommendations
                var missingFeatures = new List<string>();
                var outlierFeatures = new List<string>();
                var skewedFeatures = new List<string>();

                // Recalculate these for summary (they were calculated earlier but may be out of scope)
                foreach (var feature in statistics.Properties())
                {
                    var stats = feature.Value as JObject;
                    if (stats == null) continue;

                    double? missingPct = stats["missing_pct"]?.ToObject<double>();
                    if (missingPct.HasValue && missingPct.Value > 0)
                    {
                        missingFeatures.Add($"{feature.Name} ({missingPct:P1})");
                    }

                    double? mean = stats["mean"]?.ToObject<double>();
                    double? std = stats["std"]?.ToObject<double>();
                    double? min = stats["min"]?.ToObject<double>();
                    double? max = stats["max"]?.ToObject<double>();

                    if (mean.HasValue && std.HasValue && min.HasValue && max.HasValue)
                    {
                        double lowerBound = mean.Value - 3 * std.Value;
                        double upperBound = mean.Value + 3 * std.Value;

                        if (min.Value < lowerBound || max.Value > upperBound)
                        {
                            outlierFeatures.Add(feature.Name);
                        }

                        double range = max.Value - min.Value;
                        if (range > 0)
                        {
                            double normalizedMean = (mean.Value - min.Value) / range;
                            if (normalizedMean < 0.3 || normalizedMean > 0.7)
                            {
                                skewedFeatures.Add(feature.Name);
                            }
                        }
                    }
                }

                analysis.AppendLine("**Data Quality Summary:**");
                if (missingFeatures.Count > 0)
                {
                    analysis.AppendLine($"- Features with missing values: {string.Join(", ", missingFeatures)}");
                }
                else
                {
                    analysis.AppendLine("- ✓ No missing values detected - data is complete!");
                }

                if (outlierFeatures.Count > 0)
                {
                    analysis.AppendLine($"- Features with potential outliers: {string.Join(", ", outlierFeatures)}");
                }

                if (skewedFeatures.Count > 0)
                {
                    analysis.AppendLine($"- Features with potential skewness: {string.Join(", ", skewedFeatures)}");
                }

                analysis.AppendLine();
                analysis.AppendLine("**Overall Recommendations:**");

                if (missingFeatures.Count > 0 || outlierFeatures.Count > 0 || skewedFeatures.Count > 0)
                {
                    analysis.AppendLine("1. Perform data cleaning: handle missing values, outliers, and skewed distributions");
                    analysis.AppendLine("2. Apply feature scaling (StandardScaler or RobustScaler) after cleaning");
                    analysis.AppendLine("3. Consider feature engineering to create more robust predictors");
                    analysis.AppendLine("4. Use cross-validation to ensure model generalizes well");
                }
                else
                {
                    analysis.AppendLine("✓ Data quality looks good! Ready for model training.");
                    analysis.AppendLine("- Still recommend: feature scaling and cross-validation");
                }
            }

            return analysis.ToString();
        }
        catch (JsonReaderException)
        {
            throw; // Let base class handle JSON errors
        }
        catch (Exception)
        {
            throw; // Let base class handle generic errors
        }
    }

    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"dataset_info\": { \"n_samples\": number, \"n_features\": number, ... }, " +
               "\"statistics\": { \"feature_name\": { \"mean\": number, ... } }, " +
               "\"correlations\" (optional): { \"feature1\": { \"feature2\": number } }, " +
               "\"class_distribution\" (optional): { \"class_name\": count } }";
    }
}
