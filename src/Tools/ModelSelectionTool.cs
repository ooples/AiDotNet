using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.NeuralNetworks;
using Newtonsoft.Json;
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
    /// <summary>
    /// Returns the recommended model <see cref="Type"/> based on dataset characteristics.
    /// This provides a type-safe way to get the recommendation without parsing text output.
    /// </summary>
    /// <param name="problemType">The problem type ("regression" or "classification").</param>
    /// <param name="nSamples">Number of samples in the dataset.</param>
    /// <param name="nFeatures">Number of features in the dataset.</param>
    /// <param name="isLinear">Whether the relationship appears linear.</param>
    /// <param name="hasOutliers">Whether the dataset contains outliers.</param>
    /// <param name="computationalConstraints">Computational constraints ("low", "moderate", "high").</param>
    /// <param name="requiresInterpretability">Whether interpretability is required.</param>
    /// <returns>The open generic type definition for the recommended model.</returns>
    public static Type RecommendModelType(
        string problemType,
        int nSamples,
        int nFeatures,
        bool isLinear,
        bool hasOutliers,
        string computationalConstraints,
        bool requiresInterpretability)
    {
        bool isClassification = string.Equals(problemType, "classification", StringComparison.OrdinalIgnoreCase);

        if (nSamples < 100)
        {
            if (isLinear || requiresInterpretability)
            {
                return isClassification
                    ? typeof(LogisticRegression<>)
                    : typeof(SimpleRegression<>);
            }
            return typeof(KNearestNeighborsRegression<>);
        }

        if (nSamples < 1000)
        {
            if (requiresInterpretability)
            {
                return typeof(DecisionTreeRegression<>);
            }
            if (isLinear)
            {
                return isClassification
                    ? typeof(LogisticRegression<>)
                    : typeof(RidgeRegression<>);
            }
            return hasOutliers
                ? typeof(RandomForestRegression<>)
                : typeof(GradientBoostingRegression<>);
        }

        if (nSamples < 10000)
        {
            if (requiresInterpretability || hasOutliers)
            {
                return typeof(RandomForestRegression<>);
            }
            return typeof(GradientBoostingRegression<>);
        }

        // Large dataset
        if (requiresInterpretability)
        {
            return typeof(GradientBoostingRegression<>);
        }
        if (nFeatures > 100)
        {
            return typeof(NeuralNetworkRegression<>);
        }
        return typeof(GradientBoostingRegression<>);
    }

    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
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

        // Get the type-safe recommendation
        var recommendedType = RecommendModelType(
            problemType, nSamples, nFeatures, isLinear, hasOutliers,
            computationalConstraints, requiresInterpretability);
        var recommendedModel = recommendedType.Name.Replace("`1", string.Empty);

        // Build reasoning based on characteristics
        var reasoning = BuildReasoning(nSamples, nFeatures, isLinear, hasOutliers,
            requiresInterpretability, computationalConstraints);
        var alternatives = BuildAlternatives(nSamples, nFeatures, isLinear, hasOutliers,
            requiresInterpretability);

        // Output recommendation
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== MODEL SELECTION RECOMMENDATION ===\n");
        sb.AppendLine($"**Primary Recommendation: {recommendedModel}**\n");
        sb.AppendLine("**Reasoning:**");
        foreach (var reason in reasoning)
        {
            sb.AppendLine($"  - {reason}");
        }
        sb.AppendLine();
        sb.AppendLine("**Problem Characteristics:**");
        sb.AppendLine($"  - Problem type: {problemType}");
        sb.AppendLine($"  - Dataset size: {nSamples:N0} samples x {nFeatures} features");
        sb.AppendLine($"  - Relationship: {(isLinear ? "Linear" : "Non-linear")}");
        sb.AppendLine($"  - Outliers: {(hasOutliers ? "Present" : "Minimal/None")}");
        sb.AppendLine($"  - Missing values: {(hasMissingValues ? "Present" : "None")}");
        sb.AppendLine($"  - Interpretability required: {(requiresInterpretability ? "Yes" : "No")}");
        sb.AppendLine($"  - Computational constraints: {computationalConstraints}");
        if (alternatives.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("**Alternative Models to Consider:**");
            foreach (var (model, reason) in alternatives)
            {
                sb.AppendLine($"  - {model}: {reason}");
            }
        }
        sb.AppendLine();
        sb.AppendLine("**Next Steps:**");
        sb.AppendLine("  1. Start with the primary recommendation");
        sb.AppendLine("  2. Use cross-validation to evaluate performance");
        sb.AppendLine("  3. Try alternatives if results are unsatisfactory");
        sb.AppendLine("  4. Consider ensemble methods combining multiple models");
        return sb.ToString();
    }

    private static List<string> BuildReasoning(
        int nSamples, int nFeatures, bool isLinear, bool hasOutliers,
        bool requiresInterpretability, string computationalConstraints)
    {
        var reasoning = new List<string>();

        if (nSamples < 100)
        {
            if (isLinear || requiresInterpretability)
            {
                reasoning.Add($"Very small dataset ({nSamples} samples) - simple linear model prevents overfitting");
                reasoning.Add("Linear model provides interpretability and requires few samples to train");
                if (hasOutliers)
                {
                    reasoning.Add("Note: Linear models sensitive to outliers - consider robust regression or outlier removal");
                }
            }
            else
            {
                reasoning.Add($"Small dataset ({nSamples} samples) with non-linear patterns");
                reasoning.Add("k-NN is non-parametric and can capture non-linear relationships");
                reasoning.Add("No training required - entire dataset used for predictions");
                if (hasOutliers)
                {
                    reasoning.Add("Note: k-NN sensitive to outliers - consider data cleaning");
                }
            }
        }
        else if (nSamples < 1000)
        {
            if (requiresInterpretability)
            {
                reasoning.Add($"Small dataset ({nSamples} samples) with interpretability requirement");
                reasoning.Add("Decision trees provide clear decision rules that are easy to explain");
            }
            else if (isLinear)
            {
                reasoning.Add($"Dataset size ({nSamples} samples) suitable for regularized linear models");
                reasoning.Add("Linear patterns detected - linear model is appropriate");
                reasoning.Add("Regularization prevents overfitting");
            }
            else
            {
                reasoning.Add($"Dataset size ({nSamples} samples) supports ensemble methods");
                reasoning.Add("Non-linear patterns require flexible model");
                if (hasOutliers)
                    reasoning.Add("Random Forest is robust to outliers and missing values");
                else
                    reasoning.Add("Gradient Boosting typically achieves highest accuracy");
            }
        }
        else if (nSamples < 10000)
        {
            if (requiresInterpretability)
            {
                reasoning.Add($"Moderate dataset ({nSamples} samples) with interpretability needs");
                reasoning.Add("Random Forest balances performance with explainability");
            }
            else if (hasOutliers)
            {
                reasoning.Add($"Dataset size ({nSamples} samples) ideal for Random Forest");
                reasoning.Add("Random Forest highly robust to outliers and noise");
            }
            else
            {
                reasoning.Add($"Dataset size ({nSamples} samples) optimal for gradient boosting");
                reasoning.Add("Gradient boosting typically achieves state-of-the-art results");
            }
        }
        else
        {
            if (requiresInterpretability)
            {
                reasoning.Add($"Large dataset ({nSamples} samples) supports complex models");
                reasoning.Add("Gradient boosting achieves excellent performance");
                reasoning.Add("Use SHAP or LIME for post-hoc interpretability");
            }
            else if (nFeatures > 100)
            {
                reasoning.Add($"Large dataset ({nSamples} samples) and high dimensionality ({nFeatures} features)");
                reasoning.Add("Deep learning excels with large datasets and many features");
                if (string.Equals(computationalConstraints, "low", StringComparison.OrdinalIgnoreCase))
                {
                    reasoning.Add("Note: Neural networks require significant computational resources");
                }
            }
            else
            {
                reasoning.Add($"Large dataset ({nSamples} samples) with moderate features ({nFeatures})");
                reasoning.Add("Gradient boosting: state-of-the-art performance for tabular data");
            }
        }

        return reasoning;
    }

    private static List<(string Model, string Reason)> BuildAlternatives(
        int nSamples, int nFeatures, bool isLinear, bool hasOutliers,
        bool requiresInterpretability)
    {
        var alternatives = new List<(string Model, string Reason)>();

        if (nSamples < 100)
        {
            if (isLinear || requiresInterpretability)
                alternatives.Add(("RidgeRegression", "Regularization helps prevent overfitting on small datasets"));
            else
            {
                alternatives.Add(("DecisionTreeRegression", "Simple non-linear model, easy to interpret"));
                alternatives.Add(("SupportVectorRegression", "Effective in high-dimensional spaces"));
            }
        }
        else if (nSamples < 1000)
        {
            if (!requiresInterpretability && !isLinear)
            {
                alternatives.Add(("SupportVectorRegression", "Effective for non-linear patterns, good generalization"));
                alternatives.Add(("NeuralNetworkRegression", "Can learn complex patterns but may overfit"));
            }
            else if (isLinear)
            {
                alternatives.Add(("ElasticNetRegression", "Combines L1 and L2 regularization for feature selection"));
            }
        }
        else if (nSamples < 10000)
        {
            if (!requiresInterpretability && !hasOutliers)
            {
                alternatives.Add(("RandomForestRegression", "Faster training, less hyperparameter tuning"));
                alternatives.Add(("NeuralNetworkRegression", "Can learn very complex patterns with proper regularization"));
            }
        }
        else
        {
            if (nFeatures > 100)
                alternatives.Add(("GradientBoostingRegression", "Faster training, often competitive performance"));
            else
                alternatives.Add(("NeuralNetworkRegression", "May achieve better results with proper architecture"));
        }

        return alternatives;
    }
    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"problem_type\": \"regression|classification\", \"n_samples\": number, " +
               "\"n_features\": number, \"is_linear\": boolean, ... }";
    }
}
