using AiDotNet.Interfaces;
using Newtonsoft.Json.Linq;
namespace AiDotNet.Tools;
/// <summary>
/// A specialized tool that analyzes feature importance, identifies redundant features, detects multicollinearity,
/// and suggests feature engineering improvements to enhance model performance.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with advanced feature analysis capabilities. It examines relationships between
/// features and the target variable, identifies features with low predictive power, detects highly correlated
/// features that may cause multicollinearity issues, and suggests feature engineering techniques such as
/// transformations, interactions, and dimensionality reduction. The tool helps agents recommend which features
/// to keep, remove, transform, or combine to improve model performance and interpretability.
/// </para>
/// <para><b>For Beginners:</b> This tool helps you understand which input variables (features) actually matter
/// for your predictions and how to improve them.
///
/// Why feature analysis matters:
/// - **Not all features are useful**: Some features add noise without helping predictions
/// - **Some features are redundant**: Having age and birth_year is redundant - they contain the same information
/// - **Feature quality affects model performance**: Good features → accurate models; bad features → poor predictions
/// - **More features ≠ better**: Too many features can actually hurt performance (curse of dimensionality)
///
/// What this tool analyzes:
/// - **Feature Importance**: Which features most strongly predict the target variable
/// - **Redundancy**: Features that provide duplicate information (high correlation with each other)
/// - **Multicollinearity**: When features are so correlated they confuse the model
/// - **Low-value Features**: Features with minimal predictive power
/// - **Feature Engineering Opportunities**: Ways to create better features from existing ones
///
/// Example input (JSON format):
/// <code>
/// {
///   "features": {
///     "square_feet": {
///       "target_correlation": 0.82,
///       "importance_score": 0.45,
///       "missing_pct": 0.0,
///       "correlations": { "bedrooms": 0.71, "price": 0.82 }
///     },
///     "bedrooms": {
///       "target_correlation": 0.65,
///       "importance_score": 0.23,
///       "missing_pct": 0.0,
///       "correlations": { "square_feet": 0.71, "price": 0.65 }
///     },
///     "exterior_color": {
///       "target_correlation": 0.02,
///       "importance_score": 0.01,
///       "missing_pct": 0.05,
///       "correlations": {}
///     }
///   },
///   "target_name": "price",
///   "n_samples": 5000
/// }
/// </code>
///
/// Example output:
/// "Feature Importance Analysis:\n\n" +
/// "HIGH IMPORTANCE FEATURES (keep these!):\n" +
/// "  • square_feet (importance: 0.45, correlation: 0.82) - Strong predictor, essential\n" +
/// "  • bedrooms (importance: 0.23, correlation: 0.65) - Moderate predictor, useful\n\n" +
/// "LOW IMPORTANCE FEATURES (consider removing):\n" +
/// "  • exterior_color (importance: 0.01) - Very weak predictor, adds noise\n\n" +
/// "MULTICOLLINEARITY DETECTED:\n" +
/// "  • square_feet ↔ bedrooms (r=0.71) - High correlation\n" +
/// "  Recommendation: Consider using only square_feet or creating price_per_bedroom ratio\n\n" +
/// "FEATURE ENGINEERING SUGGESTIONS:\n" +
/// "  • Create 'price_per_sqft' = price / square_feet (normalization)\n" +
/// "  • Create 'bedroom_density' = bedrooms / square_feet (interaction feature)"
///
/// This analysis helps you build better models by focusing on the right features and creating more informative ones.
/// </para>
/// </remarks>
public class FeatureImportanceTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "FeatureImportanceTool";
    /// <inheritdoc/>
    public override string Description =>
        "Analyzes feature importance, detects redundancy, and suggests feature engineering improvements. " +
        "Input should be a JSON object: { \"features\": { \"feature_name\": { \"target_correlation\": number, " +
        "\"importance_score\": number, \"missing_pct\": number, \"correlations\": { \"other_feature\": number } } }, " +
        "\"target_name\": \"string\", \"n_samples\": number }. " +
        "Returns importance rankings, redundancy detection, multicollinearity warnings, and feature engineering suggestions.";
    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        try
        {
            var root = JObject.Parse(input);
            if (!root.TryGetProperty("features", out JsonElement featuresElem))
            {
                return "Error: Missing 'features' property in input JSON.";
            }
            string targetName = TryGetString(root, "target_name", "target");
            int nSamples = TryGetInt(root, "n_samples", 1000);
            var analysis = new System.Text.StringBuilder();
            analysis.AppendLine("=== FEATURE IMPORTANCE & ENGINEERING ANALYSIS ===\n");
            // Parse all features
            var features = new List<FeatureInfo>();
            foreach (JsonProperty featureProp in featuresElem.EnumerateObject())
            {
                var feature = new FeatureInfo
                {
                    Name = featureProp.Name
                };
                JsonElement stats = featureProp.Value;
                if (stats.TryGetProperty("target_correlation", out JsonElement corrElem))
                    feature.TargetCorrelation = Math.Abs(corrElem.GetDouble());
                if (stats.TryGetProperty("importance_score", out JsonElement impElem))
                    feature.ImportanceScore = impElem.GetDouble();
                if (stats.TryGetProperty("missing_pct", out JsonElement missingElem))
                    feature.MissingPct = missingElem.GetDouble();
                if (stats.TryGetProperty("correlations", out JsonElement correlsElem))
                {
                    foreach (JsonProperty corrProp in correlsElem.EnumerateObject())
                    {
                        feature.Correlations[corrProp.Name] = Math.Abs(corrProp.Value.GetDouble());
                    }
                }
                features.Add(feature);
            }
            // Sort by importance
            var sortedByImportance = features.OrderByDescending(f => f.ImportanceScore).ToList();
            // Categorize features
            var highImportance = sortedByImportance.Where(f => f.ImportanceScore > 0.15).ToList();
            var moderateImportance = sortedByImportance.Where(f => f.ImportanceScore > 0.05 && f.ImportanceScore <= 0.15).ToList();
            var lowImportance = sortedByImportance.Where(f => f.ImportanceScore <= 0.05).ToList();
            // === IMPORTANCE RANKINGS ===
            analysis.AppendLine("**Feature Importance Rankings:**\n");
            if (highImportance.Count > 0)
            {
                analysis.AppendLine("**HIGH IMPORTANCE** (Essential features - definitely keep):");
                foreach (var feature in highImportance)
                {
                    analysis.AppendLine($"  ✓ **{feature.Name}**");
                    analysis.AppendLine($"    - Importance Score: {feature.ImportanceScore:P1}");
                    analysis.AppendLine($"    - Target Correlation: {feature.TargetCorrelation:F3}");
                    if (feature.MissingPct > 0)
                        analysis.AppendLine($"    - Missing: {feature.MissingPct:P1}");
                    analysis.AppendLine($"    - Impact: Strong predictor, removing would significantly hurt performance");
                    analysis.AppendLine();
                }
            }
            if (moderateImportance.Count > 0)
            {
                analysis.AppendLine("**MODERATE IMPORTANCE** (Useful features - likely keep):");
                foreach (var feature in moderateImportance)
                {
                    analysis.AppendLine($"  • {feature.Name}");
                    analysis.AppendLine($"    - Importance: {feature.ImportanceScore:P1}, Correlation: {feature.TargetCorrelation:F3}");
                    analysis.AppendLine($"    - Impact: Moderate predictor, provides some value");
                    analysis.AppendLine();
                }
            }
            if (lowImportance.Count > 0)
            {
                analysis.AppendLine("**LOW IMPORTANCE** (Consider removing to reduce noise and complexity):");
                foreach (var feature in lowImportance)
                {
                    analysis.AppendLine($"  ⚠️ {feature.Name}");
                    analysis.AppendLine($"    - Importance: {feature.ImportanceScore:P1}, Correlation: {feature.TargetCorrelation:F3}");
                    analysis.AppendLine($"    - Recommendation: Weak predictor - removing may improve model generalization");
                    analysis.AppendLine();
                }
            }
            // === MULTICOLLINEARITY DETECTION ===
            analysis.AppendLine("**Multicollinearity Analysis:**\n");
            var highCorrelations = new List<(string Feature1, string Feature2, double Correlation)>();
            foreach (var feature in features)
            {
                foreach (var (otherFeature, correlation) in feature.Correlations)
                {
                    if (correlation > 0.7 && otherFeature != targetName)
                    {
                        // Avoid duplicates by only adding if feature1 < feature2 alphabetically
                        if (string.CompareOrdinal(feature.Name, otherFeature) < 0)
                        {
                            highCorrelations.Add((feature.Name, otherFeature, correlation));
                        }
                    }
                }
            }
            if (highCorrelations.Count > 0)
            {
                analysis.AppendLine("⚠️ **HIGH CORRELATION DETECTED** (May cause multicollinearity issues):\n");
                foreach (var (f1, f2, corr) in highCorrelations.OrderByDescending(x => x.Correlation))
                {
                    analysis.AppendLine($"  • **{f1}** ↔ **{f2}** (correlation: {corr:F3})");
                    // Find importance of each
                    var feat1 = features.First(f => f.Name == f1);
                    var feat2 = features.First(f => f.Name == f2);
                    if (feat1.ImportanceScore > feat2.ImportanceScore * 1.5)
                    {
                        analysis.AppendLine($"    → Recommendation: Keep '{f1}' (importance: {feat1.ImportanceScore:P1}), " +
                                          $"consider removing '{f2}' (importance: {feat2.ImportanceScore:P1})");
                    }
                    else if (feat2.ImportanceScore > feat1.ImportanceScore * 1.5)
                    {
                        analysis.AppendLine($"    → Recommendation: Keep '{f2}' (importance: {feat2.ImportanceScore:P1}), " +
                                          $"consider removing '{f1}' (importance: {feat1.ImportanceScore:P1})");
                    }
                    else
                    {
                        analysis.AppendLine($"    → Recommendation: Similar importance - consider creating interaction feature or using PCA");
                    }
                    analysis.AppendLine();
                }
                analysis.AppendLine("**What is multicollinearity?** When features are highly correlated, they provide");
                analysis.AppendLine("redundant information. This can make models unstable and coefficients unreliable.");
                analysis.AppendLine("Solution: Remove one feature, combine them, or use dimensionality reduction.\n");
            }
            else
            {
                analysis.AppendLine("✓ No significant multicollinearity detected (all feature correlations < 0.7)\n");
            }
            // === FEATURE ENGINEERING SUGGESTIONS ===
            analysis.AppendLine("**Feature Engineering Suggestions:**\n");
            var suggestions = new List<string>();
            // Suggest removing low importance features
            if (lowImportance.Count > 0)
            {
                suggestions.Add($"**Remove low-importance features:** {string.Join(", ", lowImportance.Select(f => f.Name))}");
                suggestions.Add($"  Benefit: Reduces noise, prevents overfitting, speeds up training");
            }
            // Suggest interaction features for highly correlated important features
            foreach (var (f1, f2, corr) in highCorrelations)
            {
                var feat1 = features.First(f => f.Name == f1);
                var feat2 = features.First(f => f.Name == f2);
                if (feat1.ImportanceScore > 0.1 && feat2.ImportanceScore > 0.1)
                {
                    suggestions.Add($"**Create interaction feature:** '{f1}_x_{f2}' = {f1} * {f2}");
                    suggestions.Add($"  Benefit: May capture non-linear interactions between correlated features");
                    suggestions.Add($"**Create ratio feature:** '{f1}_per_{f2}' = {f1} / {f2}");
                    suggestions.Add($"  Benefit: Normalizes one feature by another, often reveals hidden patterns");
                }
            }
            // Suggest polynomial features for highly important features
            if (highImportance.Count > 0 && highImportance.Count <= 3)
            {
                foreach (var feature in highImportance)
                {
                    if (feature.TargetCorrelation > 0.5)
                    {
                        suggestions.Add($"**Consider polynomial features for '{feature.Name}':** squared, cubed terms");
                        suggestions.Add($"  Benefit: Captures non-linear relationships with target variable");
                    }
                }
            }
            // Suggest dimensionality reduction if many features
            if (features.Count > 20)
            {
                suggestions.Add("**Consider dimensionality reduction:** PCA, t-SNE, or feature selection algorithms");
                suggestions.Add($"  Benefit: Reduce {features.Count} features while retaining most information");
            }
            // Suggest binning/discretization for specific patterns
            var highSkewFeatures = features.Where(f => f.ImportanceScore > 0.1 && f.Correlations.Values.Any(c => c > 0.8)).ToList();
            if (highSkewFeatures.Count > 0)
            {
                foreach (var feature in highSkewFeatures)
                {
                    suggestions.Add($"**Try log/sqrt transformation on '{feature.Name}'**");
                    suggestions.Add("  Benefit: Handle skewness and outliers in important features");
                }
            }
            if (suggestions.Count > 0)
            {
                foreach (var suggestion in suggestions)
                {
                    analysis.AppendLine($"  • {suggestion}");
                }
            }
            else
            {
                analysis.AppendLine("  • Current feature set looks reasonable - focus on model selection and tuning");
            }
            analysis.AppendLine();
            // === SUMMARY ===
            analysis.AppendLine("**Summary:**");
            analysis.AppendLine($"  • Total features analyzed: {features.Count}");
            analysis.AppendLine($"  • High importance: {highImportance.Count}");
            analysis.AppendLine($"  • Moderate importance: {moderateImportance.Count}");
            analysis.AppendLine($"  • Low importance: {lowImportance.Count}");
            analysis.AppendLine($"  • Multicollinearity issues: {highCorrelations.Count}");
            analysis.AppendLine();
            analysis.AppendLine("**Next Steps:**");
            analysis.AppendLine("  1. Remove or investigate low-importance features");
            analysis.AppendLine("  2. Address multicollinearity (remove redundant features or create combinations)");
            analysis.AppendLine("  3. Try suggested feature engineering techniques");
            analysis.AppendLine("  4. Re-evaluate feature importance after changes");
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
    protected override string GetJsonErrorMessage(Newtonsoft.Json.JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"features\": { \"feature_name\": { \"target_correlation\": number, ... } }, ... }";
    }
    private class FeatureInfo
    {
        public string Name { get; set; } = string.Empty;
        public double TargetCorrelation { get; set; }
        public double ImportanceScore { get; set; }
        public double MissingPct { get; set; }
        public Dictionary<string, double> Correlations { get; set; } = new();
    }
}
