using AiDotNet.Interfaces;
using Newtonsoft.Json.Linq;
namespace AiDotNet.Tools;
/// <summary>
/// A specialized tool that recommends optimal machine learning model types based on dataset characteristics,
/// problem type, and computational constraints.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with expert-level model selection capabilities. It analyzes dataset properties
/// such as size, dimensionality, linearity, presence of outliers, and feature interactions to recommend the
/// most appropriate machine learning algorithms. The tool considers trade-offs between model complexity,
/// interpretability, training time, and expected performance to suggest models that are well-suited to the
/// specific problem at hand.
/// </para>
/// <para><b>For Beginners:</b> This tool is like consulting an experienced data scientist about which machine
/// learning algorithm to use for your problem.
///
/// Choosing the right model is crucial because:
/// - Simple datasets work better with simple models (avoid overfitting)
/// - Complex patterns need powerful models (avoid underfitting)
/// - Small datasets can't support very complex models
/// - Different models have different strengths and weaknesses
///
/// What it considers:
/// - **Dataset size**: Small datasets need simple models; large datasets can handle complexity
/// - **Feature count**: High-dimensional data may need dimensionality reduction or specific models
/// - **Linearity**: Linear relationships suggest linear models; non-linear patterns need flexible models
/// - **Outliers**: Some models are sensitive to outliers, others are robust
/// - **Interpretability**: Some applications require explainable models
/// - **Training time**: Production systems may have time constraints
///
/// Example input (JSON format):
/// <code>
/// {
///   "problem_type": "regression",
///   "n_samples": 5000,
///   "n_features": 20,
///   "is_linear": false,
///   "has_outliers": true,
///   "has_missing_values": false,
///   "requires_interpretability": false,
///   "computational_constraints": "moderate"
/// }
/// </code>
///
/// Example output:
/// "Recommended Model: Random Forest Regression\n\n" +
/// "Reasoning:\n" +
/// "- Dataset size (5,000 samples) is sufficient for ensemble methods\n" +
/// "- Non-linear relationships detected - need flexible model\n" +
/// "- Outliers present - Random Forest is robust to outliers\n" +
/// "- 20 features - Random Forest handles moderate dimensionality well\n" +
/// "- No interpretability requirement - can use complex model\n\n" +
/// "Alternative Models:\n" +
/// "- Gradient Boosting: May achieve slightly better performance but more sensitive to outliers\n" +
/// "- Support Vector Regression: Good for non-linear patterns but slower on this dataset size"
///
/// This guidance helps ensure you use the right algorithm for your specific situation.
/// </para>
/// </remarks>
public class ModelSelectionTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "ModelSelectionTool";
    /// <inheritdoc/>
    public override string Description =>
        "Recommends optimal machine learning model types based on dataset characteristics. " +
        "Input should be a JSON object: { \"problem_type\": \"regression|classification\", " +
        "\"n_samples\": number, \"n_features\": number, \"is_linear\": boolean, " +
        "\"has_outliers\": boolean, \"has_missing_values\": boolean, " +
        "\"requires_interpretability\": boolean, \"computational_constraints\": \"low|moderate|high\" }. " +
        "Returns recommended model type with detailed reasoning and alternative suggestions.";
    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        try
        {
            var root = JObject.Parse(input);
            // Extract problem characteristics
            string problemType = TryGetString(root, "problem_type", "regression");
            int nSamples = TryGetInt(root, "n_samples", 1000);
            int nFeatures = TryGetInt(root, "n_features", 10);
            bool isLinear = TryGetBool(root, "is_linear", false);
            bool hasOutliers = TryGetBool(root, "has_outliers", false);
            bool hasMissingValues = TryGetBool(root, "has_missing_values", false);
            bool requiresInterpretability = TryGetBool(root, "requires_interpretability", false);
            string computationalConstraints = TryGetString(root, "computational_constraints", "moderate");
            // Build recommendation
            var recommendation = new System.Text.StringBuilder();
            recommendation.AppendLine("=== MODEL SELECTION RECOMMENDATION ===\n");
            // Determine recommended model based on characteristics
            string recommendedModel;
            var reasoning = new List<string>();
            var alternatives = new List<(string Model, string Reason)>();
            // Decision logic
            if (nSamples < 100)
            {
                // Very small dataset - use simple models
                if (isLinear || requiresInterpretability)
                {
                    recommendedModel = problemType.ToLowerInvariant() == "classification"
                        ? "Logistic Regression"
                        : "Linear Regression";
                    reasoning.Add($"Very small dataset ({nSamples} samples) - simple linear model prevents overfitting");
                    reasoning.Add("Linear model provides interpretability and requires few samples to train");
                    if (hasOutliers)
                    {
                        reasoning.Add("⚠️ Note: Linear models sensitive to outliers - consider robust regression or outlier removal");
                    }
                    alternatives.Add(("Ridge/Lasso Regression", "Regularization helps prevent overfitting on small datasets"));
                }
                else
                {
                    recommendedModel = "k-Nearest Neighbors (k-NN)";
                    reasoning.Add($"Small dataset ({nSamples} samples) with non-linear patterns");
                    reasoning.Add("k-NN is non-parametric and can capture non-linear relationships");
                    reasoning.Add("No training required - entire dataset used for predictions");
                    if (hasOutliers)
                    {
                        reasoning.Add("⚠️ Note: k-NN sensitive to outliers - consider data cleaning");
                    }
                    alternatives.Add(("Decision Tree", "Simple non-linear model, easy to interpret"));
                    alternatives.Add(("Support Vector Machine", "Effective in high-dimensional spaces"));
                }
            }
            else if (nSamples < 1000)
            {
                // Small dataset - moderate complexity
                if (requiresInterpretability)
                {
                    recommendedModel = problemType.ToLowerInvariant() == "classification"
                        ? "Decision Tree"
                        : "Decision Tree Regression";
                    reasoning.Add($"Small dataset ({nSamples} samples) with interpretability requirement");
                    reasoning.Add("Decision trees provide clear decision rules that are easy to explain");
                    reasoning.Add("Can handle both linear and non-linear relationships");
                    if (!hasOutliers)
                    {
                        alternatives.Add(("Linear/Logistic Regression with Regularization", "Simpler model, better for linear patterns"));
                    }
                }
                else if (isLinear)
                {
                    recommendedModel = problemType.ToLowerInvariant() == "classification"
                        ? "Logistic Regression with Regularization"
                        : "Ridge Regression";
                    reasoning.Add($"Dataset size ({nSamples} samples) suitable for regularized linear models");
                    reasoning.Add("Linear patterns detected - linear model is appropriate");
                    reasoning.Add("Regularization prevents overfitting");
                    alternatives.Add(("Elastic Net", "Combines L1 and L2 regularization for feature selection"));
                }
                else
                {
                    recommendedModel = hasOutliers
                        ? "Random Forest"
                        : "Gradient Boosting";
                    reasoning.Add($"Dataset size ({nSamples} samples) supports ensemble methods");
                    reasoning.Add("Non-linear patterns require flexible model");
                    if (hasOutliers)
                    {
                        reasoning.Add("Random Forest is robust to outliers and missing values");
                    }
                    else
                    {
                        reasoning.Add("Gradient Boosting typically achieves highest accuracy");
                    }
                    alternatives.Add(("Support Vector Machine", "Effective for non-linear patterns, good generalization"));
                    alternatives.Add(("Neural Network (shallow)", "Can learn complex patterns but may overfit"));
                }
            }
            else if (nSamples < 10000)
            {
                // Moderate dataset - can use more complex models
                if (requiresInterpretability)
                {
                    recommendedModel = "Random Forest with Feature Importance";
                    reasoning.Add($"Moderate dataset ({nSamples} samples) with interpretability needs");
                    reasoning.Add("Random Forest balances performance with explainability");
                    reasoning.Add("Feature importance scores help explain predictions");
                    alternatives.Add(("Gradient Boosting with SHAP", "Better performance, SHAP values for interpretability"));
                    alternatives.Add(("Regularized Linear Model", "Most interpretable but may underfit"));
                }
                else if (hasOutliers)
                {
                    recommendedModel = "Random Forest";
                    reasoning.Add($"Dataset size ({nSamples} samples) ideal for Random Forest");
                    reasoning.Add("Random Forest highly robust to outliers and noise");
                    reasoning.Add("No extensive hyperparameter tuning required");
                    alternatives.Add(("Gradient Boosting with robust loss", "Higher accuracy potential, more tuning needed"));
                    alternatives.Add(("Isolation Forest preprocessing + Gradient Boosting", "Remove outliers then use powerful model"));
                }
                else
                {
                    recommendedModel = "Gradient Boosting (XGBoost/LightGBM)";
                    reasoning.Add($"Dataset size ({nSamples} samples) optimal for gradient boosting");
                    reasoning.Add("Clean data without outliers - can use sensitive but powerful algorithm");
                    reasoning.Add("Gradient boosting typically achieves state-of-the-art results");
                    alternatives.Add(("Random Forest", "Faster training, less hyperparameter tuning"));
                    alternatives.Add(("Neural Network", "Can learn very complex patterns with proper regularization"));
                }
            }
            else
            {
                // Large dataset - can use complex models including deep learning
                if (requiresInterpretability)
                {
                    recommendedModel = "Gradient Boosting with Explainability Tools";
                    reasoning.Add($"Large dataset ({nSamples} samples) supports complex models");
                    reasoning.Add("Gradient boosting achieves excellent performance");
                    reasoning.Add("Use SHAP or LIME for post-hoc interpretability");
                    alternatives.Add(("Random Forest", "Inherently more interpretable, slightly lower performance"));
                    alternatives.Add(("GAM (Generalized Additive Models)", "High interpretability with non-linear capabilities"));
                }
                else if (nFeatures > 100)
                {
                    recommendedModel = "Deep Neural Network";
                    reasoning.Add($"Large dataset ({nSamples} samples) and high dimensionality ({nFeatures} features)");
                    reasoning.Add("Deep learning excels with large datasets and many features");
                    reasoning.Add("Can automatically learn feature interactions and representations");
                    if (computationalConstraints == "low")
                    {
                        reasoning.Add("⚠️ Note: Neural networks require significant computational resources");
                    }
                    alternatives.Add(("Gradient Boosting", "Faster training, often competitive performance"));
                    alternatives.Add(("AutoML ensemble", "Combines multiple models for best results"));
                }
                else
                {
                    recommendedModel = "Gradient Boosting (XGBoost/LightGBM/CatBoost)";
                    reasoning.Add($"Large dataset ({nSamples} samples) with moderate features ({nFeatures})");
                    reasoning.Add("Gradient boosting: state-of-the-art performance for tabular data");
                    reasoning.Add("Handles missing values, categorical features, and complex interactions");
                    alternatives.Add(("Deep Neural Network", "May achieve better results with proper architecture"));
                    alternatives.Add(("Ensemble of models", "Stack multiple models for maximum performance"));
                }
            }
            // Output recommendation
            recommendation.AppendLine($"**Primary Recommendation: {recommendedModel}**\n");
            recommendation.AppendLine("**Reasoning:**");
            foreach (var reason in reasoning)
            {
                recommendation.AppendLine($"  • {reason}");
            }
            recommendation.AppendLine();
            recommendation.AppendLine("**Problem Characteristics:**");
            recommendation.AppendLine($"  • Problem type: {problemType}");
            recommendation.AppendLine($"  • Dataset size: {nSamples:N0} samples × {nFeatures} features");
            recommendation.AppendLine($"  • Relationship: {(isLinear ? "Linear" : "Non-linear")}");
            recommendation.AppendLine($"  • Outliers: {(hasOutliers ? "Present" : "Minimal/None")}");
            recommendation.AppendLine($"  • Missing values: {(hasMissingValues ? "Present" : "None")}");
            recommendation.AppendLine($"  • Interpretability required: {(requiresInterpretability ? "Yes" : "No")}");
            recommendation.AppendLine($"  • Computational constraints: {computationalConstraints}");
            if (alternatives.Count > 0)
            {
                recommendation.AppendLine();
                recommendation.AppendLine("**Alternative Models to Consider:**");
                foreach (var (model, reason) in alternatives)
                {
                    recommendation.AppendLine($"  • {model}: {reason}");
                }
            }
            recommendation.AppendLine();
            recommendation.AppendLine("**Next Steps:**");
            recommendation.AppendLine("  1. Start with the primary recommendation");
            recommendation.AppendLine("  2. Use cross-validation to evaluate performance");
            recommendation.AppendLine("  3. Try alternatives if results are unsatisfactory");
            recommendation.AppendLine("  4. Consider ensemble methods combining multiple models");
            return recommendation.ToString();
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
               "Expected format: { \"problem_type\": \"regression|classification\", \"n_samples\": number, " +
               "\"n_features\": number, \"is_linear\": boolean, ... }";
    }
}
