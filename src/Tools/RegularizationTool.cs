using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
namespace AiDotNet.Tools;
/// <summary>
/// A specialized tool that recommends optimal regularization techniques to prevent overfitting based on
/// dataset characteristics, model type, and observed training behavior.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with expert guidance on regularization strategies. Regularization is a crucial
/// technique for preventing overfitting by adding constraints or penalties that discourage overly complex models.
/// The tool analyzes factors such as dataset size, feature dimensionality, model complexity, and signs of overfitting
/// to recommend appropriate regularization methods, strength parameters, and implementation approaches. It covers
/// L1/L2 regularization, dropout, early stopping, data augmentation, and other techniques tailored to specific
/// model architectures and problem characteristics.
/// </para>
/// <para><b>For Beginners:</b> This tool helps you prevent your model from "memorizing" training data instead of
/// learning real patterns.
///
/// What is overfitting?
/// Overfitting happens when your model learns the training data TOO well:
/// - It memorizes specific examples including noise and errors
/// - It performs great on training data but poorly on new data
/// - It's like a student who memorizes test answers but doesn't understand the concepts
///
/// What is regularization?
/// Regularization adds rules that prevent the model from becoming too complex:
/// - **L1 Regularization (Lasso)**: Pushes some features to zero, performing feature selection
/// - **L2 Regularization (Ridge)**: Shrinks all feature weights, preventing any single feature from dominating
/// - **Dropout**: Randomly ignores some neurons during training (neural networks only)
/// - **Early Stopping**: Stops training when validation performance stops improving
/// - **Data Augmentation**: Creates modified copies of training data to increase diversity
///
/// Why it matters:
/// - Prevents overfitting and improves generalization to new data
/// - Often more important than finding the "perfect" model architecture
/// - Can dramatically improve real-world performance
/// - Essential when you have limited training data
///
/// Example input (JSON format):
/// <code>
/// {
///   "model_type": "NeuralNetwork",
///   "n_samples": 1000,
///   "n_features": 50,
///   "training_score": 0.95,
///   "validation_score": 0.72,
///   "is_overfitting": true,
///   "current_regularization": "none"
/// }
/// </code>
///
/// Example output:
/// "Regularization Recommendations:\n\n" +
/// "⚠️ OVERFITTING DETECTED: Training score (0.95) >> Validation score (0.72)\n" +
/// "Gap of 0.23 indicates model is memorizing training data.\n\n" +
/// "Recommended Techniques:\n" +
/// "1. L2 Regularization (alpha=0.01):\n" +
/// "   - Prevents individual features from having excessive influence\n" +
/// "   - Particularly effective for your feature count (50)\n" +
/// "2. Dropout (rate=0.3-0.5):\n" +
/// "   - Add dropout layers after each hidden layer\n" +
/// "   - Forces network to learn robust features\n" +
/// "3. Early Stopping:\n" +
/// "   - Monitor validation loss and stop when it stops decreasing\n" +
/// "   - Saves best model automatically"
///
/// This guidance helps you build models that work well on new, unseen data, not just your training set.
/// </para>
/// </remarks>
public class RegularizationTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "RegularizationTool";
    /// <inheritdoc/>
    public override string Description =>
        "Recommends regularization techniques to prevent overfitting. " +
        "Input should be a JSON object: { \"model_type\": \"string\", \"n_samples\": number, " +
        "\"n_features\": number, \"training_score\": number, \"validation_score\": number, " +
        "\"is_overfitting\": boolean, \"current_regularization\": \"string\" }. " +
        "Returns recommended regularization techniques with parameters and implementation guidance.";
    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        try
        {
            var root = JObject.Parse(input);
            // Extract parameters
            string modelType = TryGetString(root, "model_type", "Unknown");
            int nSamples = TryGetInt(root, "n_samples", 1000);
            int nFeatures = TryGetInt(root, "n_features", 10);
            double trainingScore = TryGetDouble(root, "training_score", 0.0);
            double validationScore = TryGetDouble(root, "validation_score", 0.0);
            bool isOverfitting = TryGetBool(root, "is_overfitting", false);
            string currentRegularization = TryGetString(root, "current_regularization", "none");
            var recommendations = new System.Text.StringBuilder();
            recommendations.AppendLine("=== REGULARIZATION RECOMMENDATIONS ===\n");
            // Calculate performance gap
            double performanceGap = trainingScore - validationScore;
            bool significantGap = Math.Abs(performanceGap) > 0.1;
            // Assess overfitting severity
            if (isOverfitting || significantGap)
            {
                recommendations.AppendLine("⚠️ **OVERFITTING DETECTED**");
                if (trainingScore > 0 && validationScore > 0)
                {
                    recommendations.AppendLine($"  Training Score: {trainingScore:P1}");
                    recommendations.AppendLine($"  Validation Score: {validationScore:P1}");
                    recommendations.AppendLine($"  Performance Gap: {performanceGap:P1}");
                }
                if (performanceGap > 0.2)
                {
                    recommendations.AppendLine("  **Severity: HIGH** - Model is heavily overfitting");
                }
                else if (performanceGap > 0.1)
                {
                    recommendations.AppendLine("  **Severity: MODERATE** - Some overfitting present");
                }
                else
                {
                    recommendations.AppendLine("  **Severity: MILD** - Minor overfitting");
                }
                recommendations.AppendLine();
            }
            // Analyze data characteristics
            double samplesPerFeature = (double)nSamples / nFeatures;
            recommendations.AppendLine("**Dataset Characteristics:**");
            recommendations.AppendLine($"  • Samples: {nSamples:N0}");
            recommendations.AppendLine($"  • Features: {nFeatures}");
            recommendations.AppendLine($"  • Samples per feature: {samplesPerFeature:F1}");
            if (samplesPerFeature < 10)
            {
                recommendations.AppendLine("  • ⚠️ **HIGH DIMENSIONALITY**: Very few samples per feature - high overfitting risk");
            }
            else if (samplesPerFeature < 50)
            {
                recommendations.AppendLine("  • **MODERATE DIMENSIONALITY**: Regularization strongly recommended");
            }
            else
            {
                recommendations.AppendLine("  • **GOOD RATIO**: Sufficient samples per feature");
            }
            recommendations.AppendLine($"  • Current regularization: {currentRegularization}");
            recommendations.AppendLine();
            // Generate recommendations based on model type
            recommendations.AppendLine("**Recommended Regularization Techniques:**\n");
            var techniques = new List<(int Priority, string Technique, string Description, string Implementation)>();
            switch (modelType.ToLowerInvariant().Replace(" ", ""))
            {
                case "neuralnetwork":
                case "deeplearning":
                case "mlp":
                    GenerateNeuralNetworkRegularization(techniques, nSamples, nFeatures, performanceGap);
                    break;
                case "linearregression":
                case "logisticregression":
                    GenerateLinearModelRegularization(techniques, nSamples, nFeatures, samplesPerFeature);
                    break;
                case "randomforest":
                    GenerateRandomForestRegularization(techniques, nSamples, nFeatures, performanceGap);
                    break;
                case "gradientboosting":
                case "xgboost":
                case "lightgbm":
                    GenerateGradientBoostingRegularization(techniques, nSamples, performanceGap);
                    break;
                case "svm":
                case "supportvectormachine":
                    GenerateSVMRegularization(techniques, nSamples, nFeatures);
                    break;
                case "decisiontree":
                    GenerateDecisionTreeRegularization(techniques, nSamples);
                    break;
                default:
                    // Generic recommendations
                    GenerateGenericRegularization(techniques, nSamples, nFeatures, samplesPerFeature);
                    break;
            }
            // Sort by priority and output
            foreach (var (priority, technique, description, implementation) in techniques.OrderBy(t => t.Priority))
            {
                recommendations.AppendLine($"**{priority}. {technique}**");
                recommendations.AppendLine($"   {description}");
                recommendations.AppendLine($"   Implementation: {implementation}");
                recommendations.AppendLine();
            }
            // General advice
            recommendations.AppendLine("**General Regularization Strategies:**");
            recommendations.AppendLine("  • Start with mild regularization and gradually increase if needed");
            recommendations.AppendLine("  • Monitor both training and validation metrics");
            recommendations.AppendLine("  • Combine multiple techniques for best results (e.g., L2 + Dropout + Early Stopping)");
            recommendations.AppendLine("  • Use cross-validation to tune regularization strength");
            recommendations.AppendLine("  • Early stopping is almost always beneficial - use it!");
            if (nSamples < 1000)
            {
                recommendations.AppendLine();
                recommendations.AppendLine("**Additional Recommendations for Small Datasets:**");
                recommendations.AppendLine("  • Consider collecting more data if possible");
                recommendations.AppendLine("  • Use simpler models to avoid overfitting");
                recommendations.AppendLine("  • Try data augmentation to artificially increase dataset size");
                recommendations.AppendLine("  • Reduce feature count through feature selection");
            }
            return recommendations.ToString();
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
    private void GenerateNeuralNetworkRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples, int nFeatures, double gap)
    {
        int priority = 1;
        techniques.Add((priority++, "Dropout Regularization",
            "Randomly drops neurons during training, forcing network to learn robust features. " +
            $"Recommended rate: {(gap > 0.2 ? "0.5 (aggressive)" : gap > 0.1 ? "0.3-0.4 (moderate)" : "0.2-0.3 (mild)")}",
            "Add Dropout layers after each hidden layer. In Keras: model.add(Dropout(0.3))"));
        techniques.Add((priority++, "L2 Regularization (Weight Decay)",
            $"Penalizes large weights, preventing any single connection from dominating. " +
            $"Recommended alpha: {(nSamples < 1000 ? "0.01-0.1" : "0.001-0.01")}",
            "Add kernel_regularizer to dense layers. In Keras: Dense(units, kernel_regularizer=l2(0.01))"));
        techniques.Add((priority++, "Early Stopping",
            "Monitors validation loss and stops training when it stops improving, preventing overfitting to training data.",
            "Use EarlyStopping callback with patience=10-20 epochs. Monitor validation loss."));
        techniques.Add((priority++, "Batch Normalization",
            "Normalizes layer inputs, has mild regularization effect and stabilizes training.",
            "Add BatchNormalization layers after dense layers but before activation."));
        if (nSamples < 10000)
        {
            techniques.Add((priority++, "Data Augmentation",
                $"Artificially increases dataset size ({nSamples} samples is relatively small). " +
                "Apply random transformations to training samples.",
                "For images: rotation, flipping, cropping. For tabular: add noise, mixup, SMOTE."));
        }
        techniques.Add((priority++, "Learning Rate Scheduling",
            "Reduces learning rate as training progresses, helps convergence and regularization.",
            "Use ReduceLROnPlateau or exponential decay. Start with lr=0.001, reduce by factor of 10."));
        if (gap > 0.15)
        {
            techniques.Add((priority++, "Reduce Model Capacity",
                "Model may be too complex for available data. Consider fewer/smaller layers.",
                "Reduce neurons per layer by 25-50% or remove one hidden layer."));
        }
    }
    private void GenerateLinearModelRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples, int nFeatures, double samplesPerFeature)
    {
        int priority = 1;
        if (samplesPerFeature < 10)
        {
            techniques.Add((priority++, "L1 Regularization (Lasso)",
                $"Performs automatic feature selection by pushing some coefficients to zero. " +
                $"Critical with {nFeatures} features and {nSamples} samples (ratio: {samplesPerFeature:F1}).",
                "Use Lasso regression with alpha=1.0 to 0.001. Use LassoCV for automatic alpha selection."));
            techniques.Add((priority++, "Elastic Net",
                "Combines L1 and L2 regularization, balances feature selection with weight shrinkage.",
                "Use ElasticNet with l1_ratio=0.5 (equal mix). Tune both alpha and l1_ratio."));
        }
        techniques.Add((priority++, "L2 Regularization (Ridge)",
            "Shrinks all coefficients, particularly effective when features are correlated.",
            $"Use Ridge regression with alpha={(samplesPerFeature < 50 ? "10.0 to 0.1" : "1.0 to 0.001")}. Use RidgeCV for CV-based selection."));
        if (nFeatures > 20)
        {
            techniques.Add((priority++, "Feature Selection",
                $"Reduce feature count ({nFeatures}) to most important predictors.",
                "Use SelectKBest, Recursive Feature Elimination, or L1-based feature selection."));
        }
    }
    private void GenerateRandomForestRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples, int nFeatures, double gap)
    {
        int priority = 1;
        techniques.Add((priority++, "Increase min_samples_split",
            "Require more samples before splitting nodes, creates more conservative trees.",
            $"Set min_samples_split={Math.Max(10, nSamples / 100)}. Current default is likely too low."));
        techniques.Add((priority++, "Limit max_depth",
            $"Prevents trees from becoming too complex and memorizing noise. {(gap > 0.15 ? "Critical" : "Recommended")}.",
            $"Set max_depth={(gap > 0.2 ? "10-12" : gap > 0.1 ? "15-20" : "20-25")}. Try progressively lower values."));
        techniques.Add((priority++, "Increase min_samples_leaf",
            "Requires minimum samples in leaf nodes, smooths predictions and reduces overfitting.",
            $"Set min_samples_leaf={Math.Max(5, nSamples / 200)}."));
        techniques.Add((priority++, "Reduce max_features",
            "Limits features considered for each split, increases tree diversity.",
            "Use max_features='sqrt' or 'log2' instead of 'auto' or None."));
        if (gap > 0.15)
        {
            techniques.Add((priority++, "Reduce n_estimators",
                "Fewer trees may prevent ensemble overfitting. Try reducing tree count.",
                "Reduce from current value by 30-50%. Monitor validation performance."));
        }
    }
    private void GenerateGradientBoostingRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples, double gap)
    {
        int priority = 1;
        techniques.Add((priority++, "Reduce Learning Rate",
            "Lower learning rate makes model learn more slowly, reducing overfitting. Key parameter for boosting.",
            $"Reduce learning_rate to {(gap > 0.2 ? "0.01" : "0.05")}. Compensate by increasing n_estimators."));
        techniques.Add((priority++, "Subsample Training Data",
            "Use only a fraction of training samples for each tree, similar to bagging.",
            "Set subsample=0.7 or 0.8 (use 70-80% of data per iteration)."));
        techniques.Add((priority++, "Limit Tree Depth",
            "Shallow trees work well with boosting. Prevent individual trees from overfitting.",
            $"Set max_depth={(gap > 0.15 ? "3" : "5")}. Boosting works best with shallow trees (3-7 depth)."));
        techniques.Add((priority++, "Early Stopping",
            "Stop boosting when validation error stops decreasing. Built-in regularization.",
            "Use early_stopping_rounds=50 with validation set. Monitors validation metric."));
        techniques.Add((priority++, "Column Subsampling",
            "Use only subset of features for each tree (XGBoost/LightGBM specific).",
            "Set colsample_bytree=0.8 and/or colsample_bylevel=0.8."));
        if (nSamples < 5000)
        {
            techniques.Add((priority++, "Minimum Child Weight",
                "Prevents splits with very few samples. Important for smaller datasets.",
                "Set min_child_weight=5 to 20 depending on dataset size."));
        }
    }
    private void GenerateSVMRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples, int nFeatures)
    {
        int priority = 1;
        techniques.Add((priority++, "Reduce C Parameter",
            "Lower C = stronger regularization. Controls trade-off between margin and training error.",
            $"Try C values: {(nSamples < 1000 ? "0.1, 1.0, 10" : "1.0, 10, 100")}. Use GridSearchCV."));
        techniques.Add((priority++, "Adjust Gamma (RBF Kernel)",
            "Controls influence of single training example. Lower gamma = smoother decision boundary.",
            "Use gamma='scale' (default) or try smaller values like 0.001, 0.01."));
        if (nFeatures > nSamples)
        {
            techniques.Add((priority++, "Switch to Linear Kernel",
                $"With {nFeatures} features and {nSamples} samples, linear kernel may generalize better.",
                "Use kernel='linear' instead of 'rbf'. Faster and less prone to overfitting."));
        }
    }
    private void GenerateDecisionTreeRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples)
    {
        int priority = 1;
        techniques.Add((priority++, "Limit max_depth",
            "Most important regularization for decision trees. Prevents overly deep trees.",
            $"Set max_depth={Math.Max(3, Math.Min(15, nSamples / 50))}. Try values 5-15."));
        techniques.Add((priority++, "Increase min_samples_split",
            "Require more samples before considering a split.",
            $"Set min_samples_split={Math.Max(10, nSamples / 50)}."));
        techniques.Add((priority++, "Increase min_samples_leaf",
            "Require minimum samples in leaf nodes.",
            $"Set min_samples_leaf={Math.Max(5, nSamples / 100)}."));
        techniques.Add((priority++, "Pruning via ccp_alpha",
            "Cost complexity pruning removes subtrees that don't improve validation performance.",
            "Use ccp_alpha > 0 (try 0.01, 0.1, 1.0). Higher values = more aggressive pruning."));
    }
    private void GenerateGenericRegularization(
        List<(int, string, string, string)> techniques,
        int nSamples, int nFeatures, double samplesPerFeature)
    {
        int priority = 1;
        techniques.Add((priority++, "Cross-Validation",
            "Use cross-validation to detect overfitting and tune hyperparameters.",
            "Use 5-fold or 10-fold cross-validation to evaluate model performance."));
        if (samplesPerFeature < 20)
        {
            techniques.Add((priority++, "Feature Selection",
                $"Reduce features ({nFeatures}) to improve sample-to-feature ratio ({samplesPerFeature:F1}).",
                "Use feature importance, correlation analysis, or recursive feature elimination."));
        }
        techniques.Add((priority++, "Ensemble Methods",
            "Combine predictions from multiple models to reduce overfitting.",
            "Use bagging, boosting, or model stacking/blending."));
        if (nSamples < 5000)
        {
            techniques.Add((priority++, "Collect More Data",
                $"Current dataset ({nSamples} samples) may be insufficient. More data reduces overfitting.",
                "Collect additional training examples if possible. Data often more valuable than better algorithms."));
        }
    }
    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"model_type\": \"string\", \"n_samples\": number, \"n_features\": number, " +
               "\"training_score\": number, \"validation_score\": number, ... }";
    }
}
