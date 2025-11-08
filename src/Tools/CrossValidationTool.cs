using AiDotNet.Interfaces;
using System.Text.Json;

namespace AiDotNet.Tools;

/// <summary>
/// A specialized tool that recommends optimal cross-validation strategies based on dataset characteristics,
/// problem type, and computational constraints.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with expert guidance on cross-validation methodology. Cross-validation is a
/// critical technique for assessing model performance and preventing overfitting. The tool analyzes factors
/// such as dataset size, problem type (classification vs regression), class imbalance, temporal structure,
/// and computational resources to recommend the most appropriate CV strategy, number of folds, and validation
/// approach. It helps ensure that model evaluation is robust, unbiased, and computationally feasible.
/// </para>
/// <para><b>For Beginners:</b> This tool helps you choose the right way to test whether your model actually works well.
///
/// What is cross-validation?
/// Cross-validation is a technique for testing your model's performance by:
/// 1. Splitting your data into multiple parts (folds)
/// 2. Training on some parts and testing on others
/// 3. Rotating which parts are used for training vs testing
/// 4. Averaging the results to get a reliable performance estimate
///
/// Why it matters:
/// - **Prevents overfitting**: Ensures your model works on new data, not just training data
/// - **Reliable estimates**: Gives you confidence that reported accuracy is real
/// - **Catches problems**: Identifies if model only works on certain subsets of data
/// - **Guides decisions**: Helps you choose between different models and hyperparameters
///
/// Different CV strategies:
/// - **K-Fold**: Standard approach, splits data into K equal parts
/// - **Stratified K-Fold**: Maintains class proportions in each fold (for classification)
/// - **Time Series Split**: Respects temporal order (for time-based data)
/// - **Leave-One-Out**: Uses each sample once as validation (for very small datasets)
/// - **Hold-Out**: Simple train/test split (fastest but less reliable)
///
/// Example input (JSON format):
/// <code>
/// {
///   "n_samples": 5000,
///   "n_features": 20,
///   "problem_type": "classification",
///   "is_time_series": false,
///   "is_imbalanced": true,
///   "has_groups": false,
///   "computational_budget": "moderate"
/// }
/// </code>
///
/// Example output:
/// "Recommended Cross-Validation Strategy:\n\n" +
/// "Strategy: Stratified K-Fold Cross-Validation\n" +
/// "Number of Folds: 5\n\n" +
/// "Reasoning:\n" +
/// "- Dataset size (5,000 samples) is suitable for 5-fold CV\n" +
/// "- Stratified folding maintains class proportions in imbalanced data\n" +
/// "- Each fold will have ~1,000 samples for validation\n" +
/// "- Provides good balance between reliability and computational cost\n\n" +
/// "Implementation Tips:\n" +
/// "- Use StratifiedKFold from scikit-learn\n" +
/// "- Set random_state for reproducibility\n" +
/// "- Consider stratified sampling if classes are highly imbalanced"
///
/// This guidance ensures your model evaluation is trustworthy and appropriate for your specific situation.
/// </para>
/// </remarks>
public class CrossValidationTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "CrossValidationTool";

    /// <inheritdoc/>
    public override string Description =>
        "Recommends optimal cross-validation strategies based on dataset characteristics. " +
        "Input should be a JSON object: { \"n_samples\": number, \"n_features\": number, " +
        "\"problem_type\": \"regression|classification\", \"is_time_series\": boolean, " +
        "\"is_imbalanced\": boolean, \"has_groups\": boolean, \"computational_budget\": \"low|moderate|high\" }. " +
        "Returns recommended CV strategy, number of folds, and implementation guidance.";

    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(input);
            JsonElement root = document.RootElement;

            // Extract parameters
            int nSamples = TryGetInt(root, "n_samples", 1000);
            int nFeatures = TryGetInt(root, "n_features", 10);
            string problemType = TryGetString(root, "problem_type", "regression");
            bool isTimeSeries = TryGetBool(root, "is_time_series", false);
            bool isImbalanced = TryGetBool(root, "is_imbalanced", false);
            bool hasGroups = TryGetBool(root, "has_groups", false);
            string computationalBudget = TryGetString(root, "computational_budget", "moderate");

            var recommendation = new System.Text.StringBuilder();
            recommendation.AppendLine("=== CROSS-VALIDATION STRATEGY RECOMMENDATION ===\n");

            string cvStrategy;
            int nFolds = 5;
            var reasoning = new List<string>();
            var implementationTips = new List<string>();
            var warnings = new List<string>();

            // Determine CV strategy based on characteristics
            if (isTimeSeries)
            {
                cvStrategy = "Time Series Split Cross-Validation";
                nFolds = Math.Min(5, Math.Max(3, nSamples / 200));

                reasoning.Add("Time series data detected - must respect temporal order");
                reasoning.Add("Time Series Split ensures training always predicts future, never past");
                reasoning.Add($"Using {nFolds} splits to progressively validate on later time periods");
                reasoning.Add("Each split uses increasingly more data for training, simulating real deployment");

                implementationTips.Add("Use TimeSeriesSplit from scikit-learn");
                implementationTips.Add("DO NOT shuffle the data - temporal order is critical");
                implementationTips.Add("Ensure consistent time intervals between samples");
                implementationTips.Add("Consider seasonal patterns when choosing split points");
                implementationTips.Add("Monitor for concept drift between train and validation sets");

                warnings.Add("⚠️ CRITICAL: Never use standard K-Fold with time series - causes data leakage!");
                warnings.Add("⚠️ Shuffling time series data breaks temporal dependencies");
            }
            else if (hasGroups)
            {
                cvStrategy = "Group K-Fold Cross-Validation";
                nFolds = Math.Min(5, Math.Max(3, nSamples / 100));

                reasoning.Add("Grouped data detected - samples within groups are not independent");
                reasoning.Add("Group K-Fold ensures entire groups are in either train or validation, never split");
                reasoning.Add($"Using {nFolds} folds to rotate which groups are held out");
                reasoning.Add("Prevents information leakage from same group appearing in both train and validation");

                implementationTips.Add("Use GroupKFold from scikit-learn");
                implementationTips.Add("Provide group labels to the cv parameter (e.g., patient_id, user_id)");
                implementationTips.Add("Ensure groups have multiple samples to enable meaningful splits");
                implementationTips.Add("Balance group sizes if possible to avoid very unequal folds");

                warnings.Add("⚠️ Group sizes affect fold balance - monitor validation set sizes");
            }
            else if (nSamples < 50)
            {
                cvStrategy = "Leave-One-Out Cross-Validation (LOOCV)";
                nFolds = nSamples;

                reasoning.Add($"Very small dataset ({nSamples} samples) requires maximum data utilization");
                reasoning.Add("LOOCV uses all but one sample for training, giving nearly unbiased estimates");
                reasoning.Add($"Will perform {nSamples} training iterations - one per sample");
                reasoning.Add("High variance but necessary for reliable evaluation with limited data");

                implementationTips.Add("Use LeaveOneOut from scikit-learn");
                implementationTips.Add("⚠️ WARNING: Very computationally expensive - consider model complexity");
                implementationTips.Add("Consider Leave-P-Out as faster alternative if still too slow");
                implementationTips.Add("Results may have high variance - interpret with caution");

                if (nSamples < 20)
                {
                    warnings.Add($"⚠️ CRITICAL: Only {nSamples} samples - consider collecting more data");
                    warnings.Add("⚠️ Model evaluation will be extremely uncertain");
                }
            }
            else if (nSamples < 100)
            {
                nFolds = Math.Min(5, nSamples / 10);
                cvStrategy = problemType.ToLowerInvariant() == "classification" && isImbalanced
                    ? "Stratified K-Fold Cross-Validation"
                    : "K-Fold Cross-Validation";

                reasoning.Add($"Small dataset ({nSamples} samples) - using {nFolds}-fold CV");
                reasoning.Add($"Each fold will have ~{nSamples / nFolds} samples for validation");
                reasoning.Add("Fewer folds maximize training data per iteration");

                if (isImbalanced)
                {
                    reasoning.Add("Stratified folding maintains class proportions despite imbalance");
                }

                implementationTips.Add($"Use {(isImbalanced ? "StratifiedKFold" : "KFold")} from scikit-learn");
                implementationTips.Add("Set random_state for reproducibility");
                implementationTips.Add("Consider repeating CV multiple times with different seeds");

                warnings.Add($"⚠️ Small dataset - consider simpler models to avoid overfitting");
            }
            else if (nSamples < 1000)
            {
                nFolds = 5;
                cvStrategy = problemType.ToLowerInvariant() == "classification" && isImbalanced
                    ? "Stratified K-Fold Cross-Validation"
                    : "K-Fold Cross-Validation";

                reasoning.Add($"Moderate dataset ({nSamples} samples) - standard {nFolds}-fold CV is appropriate");
                reasoning.Add($"Each fold will have ~{nSamples / nFolds} samples for validation");
                reasoning.Add("Good balance between training data and reliable error estimation");

                if (isImbalanced)
                {
                    reasoning.Add("Stratified folding ensures balanced class representation in all folds");
                }

                implementationTips.Add($"Use {(isImbalanced ? "StratifiedKFold" : "KFold")} with n_splits={nFolds}");
                implementationTips.Add("Set shuffle=True and random_state for reproducible shuffling");
                implementationTips.Add("Monitor class distribution in each fold to verify stratification");
            }
            else if (nSamples < 10000)
            {
                nFolds = computationalBudget switch
                {
                    "high" => 10,
                    "moderate" => 5,
                    _ => 3
                };

                cvStrategy = problemType.ToLowerInvariant() == "classification" && isImbalanced
                    ? "Stratified K-Fold Cross-Validation"
                    : "K-Fold Cross-Validation";

                reasoning.Add($"Dataset size ({nSamples} samples) supports {nFolds}-fold CV");
                reasoning.Add($"Each fold will have ~{nSamples / nFolds} samples for validation");
                reasoning.Add($"{nFolds} folds balances reliability with computational cost ({computationalBudget} budget)");

                if (isImbalanced)
                {
                    reasoning.Add("Stratified folding prevents fold-to-fold variation due to class imbalance");
                }

                implementationTips.Add($"Use {(isImbalanced ? "StratifiedKFold" : "KFold")} with n_splits={nFolds}");
                implementationTips.Add("Set shuffle=True for better fold diversity");
                implementationTips.Add("Consider nested CV for hyperparameter tuning");
            }
            else
            {
                // Large dataset
                if (computationalBudget == "low")
                {
                    cvStrategy = "Hold-Out Validation";
                    nFolds = 1;

                    reasoning.Add($"Large dataset ({nSamples} samples) with low computational budget");
                    reasoning.Add("Hold-out validation (single train/test split) is sufficient");
                    reasoning.Add("Recommend 80/20 train/test split");
                    reasoning.Add("Large validation set (~2,000+ samples) provides reliable estimates");

                    implementationTips.Add("Use train_test_split from scikit-learn");
                    implementationTips.Add("Set test_size=0.2 for 80/20 split");
                    implementationTips.Add($"Use stratify parameter if data is imbalanced (stratify=y)");
                    implementationTips.Add("Set random_state for reproducibility");
                    implementationTips.Add("Consider creating separate validation set for hyperparameter tuning");
                }
                else
                {
                    nFolds = computationalBudget == "high" ? 10 : 5;
                    cvStrategy = problemType.ToLowerInvariant() == "classification" && isImbalanced
                        ? "Stratified K-Fold Cross-Validation"
                        : "K-Fold Cross-Validation";

                    reasoning.Add($"Large dataset ({nSamples} samples) with {computationalBudget} computational budget");
                    reasoning.Add($"{nFolds}-fold CV provides robust performance estimates");
                    reasoning.Add($"Each fold will have ~{nSamples / nFolds} samples for validation");
                    reasoning.Add("Large validation sets reduce variance in performance metrics");

                    if (isImbalanced)
                    {
                        reasoning.Add("Stratified sampling ensures consistent class distributions across folds");
                    }

                    implementationTips.Add($"Use {(isImbalanced ? "StratifiedKFold" : "KFold")} with n_splits={nFolds}");
                    implementationTips.Add("Set shuffle=True for better generalization");
                    implementationTips.Add("Consider parallel processing to speed up CV");
                    implementationTips.Add("Use nested CV for unbiased hyperparameter optimization");
                }
            }

            // Output recommendation
            recommendation.AppendLine($"**Recommended Strategy:** {cvStrategy}");
            recommendation.AppendLine($"**Number of Folds/Splits:** {nFolds}");
            recommendation.AppendLine();

            recommendation.AppendLine("**Reasoning:**");
            foreach (var reason in reasoning)
            {
                recommendation.AppendLine($"  • {reason}");
            }

            recommendation.AppendLine();
            recommendation.AppendLine("**Implementation Tips:**");
            foreach (var tip in implementationTips)
            {
                recommendation.AppendLine($"  • {tip}");
            }

            if (warnings.Count > 0)
            {
                recommendation.AppendLine();
                recommendation.AppendLine("**Warnings:**");
                foreach (var warning in warnings)
                {
                    recommendation.AppendLine($"  {warning}");
                }
            }

            recommendation.AppendLine();
            recommendation.AppendLine("**General Best Practices:**");
            recommendation.AppendLine("  • Always set random_state/seed for reproducibility");
            recommendation.AppendLine("  • Report mean AND standard deviation of CV scores");
            recommendation.AppendLine("  • Check for high variance across folds (indicates unstable model)");
            recommendation.AppendLine("  • Use same CV strategy for model selection and final evaluation");
            recommendation.AppendLine("  • Consider stratification for classification, especially with class imbalance");

            if (!isTimeSeries && !hasGroups)
            {
                recommendation.AppendLine("  • Shuffle data before splitting (unless time series or grouped)");
            }

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
    protected override string GetJsonErrorMessage(JsonException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"n_samples\": number, \"problem_type\": \"regression|classification\", " +
               "\"is_time_series\": boolean, \"is_imbalanced\": boolean, ... }";
    }
}
