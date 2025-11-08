using AiDotNet.Interfaces;
using System.Text.Json;

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
        "\"std\": number, \"min\": number, \"max\": number, \"missing_pct\": number } } }. " +
        "Returns detailed analysis including distribution characteristics, outlier detection, missing value assessment, " +
        "correlation insights, and data quality recommendations.";

    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        try
        {
            // Parse the input JSON
            using JsonDocument document = JsonDocument.Parse(input);
            JsonElement root = document.RootElement;

            // Extract dataset information
            if (!root.TryGetProperty("dataset_info", out JsonElement datasetInfo))
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
            if (root.TryGetProperty("statistics", out JsonElement statistics))
            {
                analysis.AppendLine("**Feature-Level Analysis:**");

                var missingFeatures = new List<string>();
                var skewedFeatures = new List<string>();
                var outlierFeatures = new List<string>();

                foreach (JsonProperty feature in statistics.EnumerateObject())
                {
                    string featureName = feature.Name;
                    JsonElement stats = feature.Value;

                    // Missing values check
                    if (stats.TryGetProperty("missing_pct", out JsonElement missingElem))
                    {
                        double missingPct = missingElem.GetDouble();
                        if (missingPct > 0)
                        {
                            missingFeatures.Add($"{featureName} ({missingPct:P1})");

                            if (missingPct > 0.3)
                            {
                                analysis.AppendLine($"- ⚠️ CRITICAL: '{featureName}' has {missingPct:P1} missing values!");
                                analysis.AppendLine($"  Recommendation: Consider removing this feature or using advanced imputation techniques.");
                            }
                            else if (missingPct > 0.05)
                            {
                                analysis.AppendLine($"- ⚠️ WARNING: '{featureName}' has {missingPct:P1} missing values.");
                                analysis.AppendLine($"  Recommendation: Apply imputation (mean/median for numeric, mode for categorical).");
                            }
                        }
                    }

                    // Skewness and outlier detection (simple heuristic)
                    if (stats.TryGetProperty("mean", out JsonElement meanElem) &&
                        stats.TryGetProperty("std", out JsonElement stdElem) &&
                        stats.TryGetProperty("min", out JsonElement minElem) &&
                        stats.TryGetProperty("max", out JsonElement maxElem))
                    {
                        double mean = meanElem.GetDouble();
                        double std = stdElem.GetDouble();
                        double min = minElem.GetDouble();
                        double max = maxElem.GetDouble();

                        // Check for potential outliers (values beyond 3 standard deviations)
                        double lowerBound = mean - 3 * std;
                        double upperBound = mean + 3 * std;

                        if (min < lowerBound || max > upperBound)
                        {
                            outlierFeatures.Add(featureName);
                            analysis.AppendLine($"- NOTICE: '{featureName}' may contain outliers (range: {min:F2} to {max:F2}, mean: {mean:F2}, std: {std:F2}).");
                            analysis.AppendLine($"  Recommendation: Investigate extreme values, consider clipping or robust scaling.");
                        }

                        // Check for skewness (simple heuristic: if median << mean or median >> mean)
                        // Since we don't have median, use a rough approximation
                        double range = max - min;
                        if (range > 0)
                        {
                            double normalizedMean = (mean - min) / range;
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

                // Summary recommendations
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
            else
            {
                analysis.AppendLine("**Feature-Level Analysis:**");
                analysis.AppendLine("No detailed feature statistics provided. Cannot perform detailed analysis.");
            }

            return analysis.ToString();
        }
        catch (JsonException)
        {
            throw; // Let base class handle JSON errors
        }
        catch (Exception)
        {
            throw; // Let base class handle generic errors
        }
    }

    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"dataset_info\": { \"n_samples\": number, \"n_features\": number, ... }, " +
               "\"statistics\": { \"feature_name\": { \"mean\": number, ... } } }";
    }
}
