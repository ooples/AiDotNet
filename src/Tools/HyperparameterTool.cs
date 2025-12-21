using System.Linq;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
namespace AiDotNet.Tools;
/// <summary>
/// A specialized tool that suggests optimal hyperparameter values and ranges for machine learning models
/// based on dataset characteristics and model type.
/// </summary>
/// <remarks>
/// <para>
/// This tool provides AI agents with expert-level hyperparameter tuning guidance. Hyperparameters are the
/// configuration settings that control how a machine learning algorithm learns, distinct from the parameters
/// that the model learns during training. The tool analyzes the dataset size, feature dimensionality, problem
/// complexity, and model type to recommend hyperparameter values that balance model capacity with generalization.
/// It provides both specific recommended values and reasonable ranges for exploration during tuning.
/// </para>
/// <para><b>For Beginners:</b> This tool is like having an expert help you configure the settings on your
/// machine learning model for best performance.
///
/// What are hyperparameters?
/// Think of them as the "settings" or "knobs" you adjust on your model:
/// - **Learning rate**: How fast the model learns (too fast = unstable, too slow = takes forever)
/// - **Max depth**: How complex the model can get (too deep = memorizes noise, too shallow = misses patterns)
/// - **Number of trees/layers**: How many components the model has (more isn't always better)
/// - **Regularization**: How much to penalize complexity (prevents overfitting)
///
/// Why proper hyperparameters matter:
/// - Wrong settings can cause overfitting (memorizing training data, failing on new data)
/// - Wrong settings can cause underfitting (model too simple to capture real patterns)
/// - Good hyperparameters maximize performance and generalization
/// - Manual tuning can take hours or days; this tool provides smart starting points
///
/// Example input (JSON format):
/// <code>
/// {
///   "model_type": "RandomForest",
///   "n_samples": 5000,
///   "n_features": 20,
///   "problem_type": "regression",
///   "data_complexity": "moderate"
/// }
/// </code>
///
/// Example output:
/// "Recommended Hyperparameters for Random Forest:\n\n" +
/// "n_estimators: 200 (range: 100-500)\n" +
/// "  • Number of trees in the forest\n" +
/// "  • 200 trees provide good balance for your dataset size (5,000 samples)\n" +
/// "  • More trees improve stability but have diminishing returns beyond 200\n\n" +
/// "max_depth: 15 (range: 10-20)\n" +
/// "  • Maximum depth of each tree\n" +
/// "  • Depth 15 allows capturing complex patterns without overfitting\n" +
/// "  • Dataset size supports this complexity level\n\n" +
/// "min_samples_split: 10 (range: 5-20)\n" +
/// "  • Minimum samples required to split a node\n" +
/// "  • Higher values prevent overfitting to noise\n"
///
/// These recommendations give you excellent starting points and help you understand what each setting does.
/// </para>
/// </remarks>
public class HyperparameterTool : ToolBase
{
    /// <inheritdoc/>
    public override string Name => "HyperparameterTool";
    /// <inheritdoc/>
    public override string Description =>
        "Suggests optimal hyperparameter values and ranges for machine learning models. " +
        "Input should be a JSON object: { \"model_type\": \"string\", \"n_samples\": number, " +
        "\"n_features\": number, \"problem_type\": \"regression|classification\", " +
        "\"data_complexity\": \"low|moderate|high\" }. " +
        "Returns recommended hyperparameter values with explanations, valid ranges, and tuning guidance.";
    /// <inheritdoc/>
    protected override string ExecuteCore(string input)
    {
        var root = JObject.Parse(input);
        // Extract parameters
        string modelType = TryGetString(root, "model_type", "RandomForest");
        int nSamples = TryGetInt(root, "n_samples", 1000);
        int nFeatures = TryGetInt(root, "n_features", 10);
        string problemType = TryGetString(root, "problem_type", "regression");
        string dataComplexity = TryGetString(root, "data_complexity", "moderate");

        // Validate input parameters
        if (nSamples <= 0)
            return "Error: n_samples must be a positive integer.";
        if (nFeatures <= 0)
            return "Error: n_features must be a positive integer.";
        if (string.IsNullOrWhiteSpace(problemType))
            return "Error: problem_type cannot be empty.";
        if (!new[] { "regression", "classification", "clustering", "timeseries" }.Contains(problemType.ToLowerInvariant()))
            return $"Warning: Unexpected problem_type '{problemType}'. Expected: regression, classification, clustering, or timeseries.";
        if (!new[] { "low", "moderate", "high" }.Contains(dataComplexity.ToLowerInvariant()))
            return $"Warning: Unexpected data_complexity '{dataComplexity}'. Expected: low, moderate, or high.";

        // Normalize dataComplexity to lowercase for consistent handling in helper methods
        dataComplexity = dataComplexity.ToLowerInvariant();

        var recommendations = new System.Text.StringBuilder();
        recommendations.AppendLine("=== HYPERPARAMETER RECOMMENDATIONS ===\n");
        recommendations.AppendLine($"**Model Type:** {modelType}");
        recommendations.AppendLine($"**Dataset:** {nSamples:N0} samples × {nFeatures} features");
        recommendations.AppendLine($"**Problem:** {problemType}");
        recommendations.AppendLine($"**Data Complexity:** {dataComplexity}\n");
        // Generate recommendations based on model type
        switch (modelType.ToLowerInvariant().Replace(" ", ""))
        {
            case "randomforest":
            case "randomforestregression":
            case "randomforestclassification":
                GenerateRandomForestRecommendations(recommendations, nSamples, nFeatures, dataComplexity);
                break;
            case "gradientboosting":
            case "xgboost":
            case "lightgbm":
            case "catboost":
                GenerateGradientBoostingRecommendations(recommendations, nSamples, nFeatures, dataComplexity, modelType);
                break;
            case "neuralnetwork":
            case "deeplearning":
            case "mlp":
                GenerateNeuralNetworkRecommendations(recommendations, nSamples, nFeatures, dataComplexity, problemType);
                break;
            case "svm":
            case "supportvectormachine":
            case "svr":
            case "svc":
                GenerateSVMRecommendations(recommendations, nSamples, nFeatures, dataComplexity);
                break;
            case "linearregression":
            case "logisticregression":
            case "ridge":
            case "lasso":
            case "elasticnet":
                GenerateLinearModelRecommendations(recommendations, nSamples, nFeatures, modelType);
                break;
            case "decisiontree":
            case "cart":
                GenerateDecisionTreeRecommendations(recommendations, nSamples, nFeatures, dataComplexity);
                break;
            case "knn":
            case "knearestneighbors":
                GenerateKNNRecommendations(recommendations, nSamples, nFeatures);
                break;
            default:
                return $"Model type '{modelType}' not recognized. Supported models: RandomForest, GradientBoosting, " +
                       "NeuralNetwork, SVM, LinearRegression, LogisticRegression, DecisionTree, KNN.";
        }
        recommendations.AppendLine("\n**General Tuning Advice:**");
        recommendations.AppendLine("  • Start with recommended values");
        recommendations.AppendLine("  • Use cross-validation to evaluate different settings");
        recommendations.AppendLine("  • Try grid search or random search within suggested ranges");
        recommendations.AppendLine("  • Monitor for overfitting: if training score >> validation score, increase regularization");
        recommendations.AppendLine("  • Consider automated hyperparameter optimization (Optuna, Hyperopt) for extensive tuning");
        return recommendations.ToString();
    }
    /// <inheritdoc/>
    protected override string GetJsonErrorMessage(JsonReaderException ex)
    {
        return $"Error: Invalid JSON format. {ex.Message}\n" +
               "Expected format: { \"model_type\": \"string\", \"n_samples\": number, \"n_features\": number, ... }";
    }
    private void GenerateRandomForestRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures, string complexity)
    {
        // Determine n_estimators based on dataset size
        int nEstimators = nSamples switch
        {
            < 1000 => 100,
            < 10000 => 200,
            _ => 300
        };
        // Determine max_depth based on complexity and dataset size
        int maxDepth = (nSamples, complexity) switch
        {
            ( < 1000, "low") => 8,
            ( < 1000, "moderate") => 12,
            ( < 1000, "high") => 15,
            ( < 10000, "low") => 12,
            ( < 10000, "moderate") => 15,
            ( < 10000, "high") => 20,
            (_, "low") => 15,
            (_, "moderate") => 20,
            _ => 25
        };
        int minSamplesSplit = nSamples switch
        {
            < 1000 => 5,
            < 10000 => 10,
            _ => 20
        };
        sb.AppendLine("**Recommended Hyperparameters:**\n");
        sb.AppendLine($"**n_estimators:** {nEstimators} (range: {nEstimators / 2}-{nEstimators * 2})");
        sb.AppendLine("  • Number of trees in the forest");
        sb.AppendLine($"  • {nEstimators} trees balance performance and training time for {nSamples:N0} samples");
        sb.AppendLine("  • More trees → more stable but slower; fewer trees → faster but less stable");
        sb.AppendLine();
        sb.AppendLine($"**max_depth:** {maxDepth} (range: {Math.Max(5, maxDepth - 5)}-{maxDepth + 5})");
        sb.AppendLine("  • Maximum depth of each tree");
        sb.AppendLine($"  • Depth {maxDepth} suitable for {complexity} complexity data");
        sb.AppendLine("  • Deeper → can learn more complex patterns but risk overfitting");
        sb.AppendLine();
        sb.AppendLine($"**min_samples_split:** {minSamplesSplit} (range: {Math.Max(2, minSamplesSplit / 2)}-{minSamplesSplit * 2})");
        sb.AppendLine("  • Minimum samples required to split an internal node");
        sb.AppendLine("  • Higher values → more conservative splits, less overfitting");
        sb.AppendLine();
        sb.AppendLine($"**max_features:** \"sqrt\" (alternatives: \"log2\", {Math.Max(1, nFeatures / 3)})");
        sb.AppendLine("  • Number of features to consider for each split");
        sb.AppendLine($"  • sqrt({nFeatures}) ≈ {(int)Math.Sqrt(nFeatures)} features provides good randomization");
        sb.AppendLine("  • Lower → more randomization and faster; higher → potentially better splits");
    }
    private void GenerateGradientBoostingRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures,
        string complexity, string modelType)
    {
        double learningRate = nSamples switch
        {
            < 1000 => 0.1,
            < 10000 => 0.05,
            _ => 0.01
        };
        int nEstimators = nSamples switch
        {
            < 1000 => 100,
            < 10000 => 500,
            _ => 1000
        };
        int maxDepth = complexity switch
        {
            "low" => 3,
            "moderate" => 5,
            _ => 7
        };
        sb.AppendLine("**Recommended Hyperparameters:**\n");
        sb.AppendLine($"**learning_rate:** {learningRate} (range: {learningRate / 10:F3}-{learningRate * 2:F2})");
        sb.AppendLine("  • Step size for each boosting iteration");
        sb.AppendLine($"  • {learningRate} is appropriate for {nSamples:N0} samples");
        sb.AppendLine("  • Lower learning rate → better generalization but needs more trees");
        sb.AppendLine("  • Higher learning rate → faster training but risk of overfitting");
        sb.AppendLine();
        sb.AppendLine($"**n_estimators:** {nEstimators} (range: {nEstimators / 2}-{nEstimators * 2})");
        sb.AppendLine("  • Number of boosting iterations");
        sb.AppendLine($"  • Balanced with learning rate {learningRate} for optimal performance");
        sb.AppendLine("  • Use early stopping to find optimal value automatically");
        sb.AppendLine();
        sb.AppendLine($"**max_depth:** {maxDepth} (range: 3-10)");
        sb.AppendLine("  • Maximum depth of each tree (typically shallow for boosting)");
        sb.AppendLine($"  • Depth {maxDepth} suitable for {complexity} complexity");
        sb.AppendLine("  • Gradient boosting works well with shallow trees due to sequential learning");
        sb.AppendLine();
        sb.AppendLine("**subsample:** 0.8 (range: 0.6-1.0)");
        sb.AppendLine("  • Fraction of samples used for each tree");
        sb.AppendLine("  • Subsampling adds randomization and prevents overfitting");
        sb.AppendLine();
        if (modelType.ToLowerInvariant().Contains("xgboost"))
        {
            sb.AppendLine("**colsample_bytree:** 0.8 (range: 0.6-1.0)");
            sb.AppendLine("  • Fraction of features used for each tree (XGBoost specific)");
            sb.AppendLine("  • Adds further randomization similar to Random Forest");
        }
    }
    private void GenerateNeuralNetworkRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures,
        string complexity, string problemType)
    {
        // Determine architecture based on dataset
        int hiddenLayerSize = Math.Max(nFeatures * 2, 64);
        if (nSamples >= 10000) hiddenLayerSize = Math.Max(hiddenLayerSize, 128);
        int numLayers = complexity switch
        {
            "low" => 1,
            "moderate" => 2,
            _ => 3
        };
        double learningRate = 0.001;
        int batchSize = Math.Min(64, Math.Max(16, nSamples / 100));
        sb.AppendLine("**Recommended Architecture:**\n");
        sb.AppendLine($"**hidden_layers:** {numLayers} layers of size [{string.Join(", ", Enumerable.Repeat(hiddenLayerSize, numLayers))}]");
        sb.AppendLine($"  • {numLayers} hidden layer(s) for {complexity} complexity data");
        sb.AppendLine($"  • {hiddenLayerSize} neurons provide sufficient capacity");
        sb.AppendLine();
        sb.AppendLine($"**learning_rate:** {learningRate} (range: 0.0001-0.01)");
        sb.AppendLine("  • Adam optimizer learning rate");
        sb.AppendLine("  • Start with 0.001 and use learning rate scheduling");
        sb.AppendLine();
        sb.AppendLine($"**batch_size:** {batchSize} (range: {batchSize / 2}-{batchSize * 2})");
        sb.AppendLine("  • Number of samples per gradient update");
        sb.AppendLine($"  • {batchSize} suitable for {nSamples:N0} total samples");
        sb.AppendLine();
        sb.AppendLine("**dropout:** 0.2-0.5");
        sb.AppendLine("  • Dropout rate for regularization");
        sb.AppendLine("  • Higher dropout for larger networks or risk of overfitting");
        sb.AppendLine();
        sb.AppendLine("**epochs:** 100-500 with early stopping");
        sb.AppendLine("  • Training iterations over entire dataset");
        sb.AppendLine("  • Use validation loss monitoring to stop when no improvement");
    }
    private void GenerateSVMRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures, string complexity)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");
        sb.AppendLine("**kernel:** 'rbf' (alternatives: 'linear', 'poly')");
        sb.AppendLine("  • Radial Basis Function for non-linear patterns");
        sb.AppendLine("  • Use 'linear' if data appears linearly separable (faster)");
        sb.AppendLine();
        double C = nSamples < 1000 ? 1.0 : 10.0;
        sb.AppendLine($"**C:** {C} (range: 0.1-100)");
        sb.AppendLine("  • Regularization parameter");
        sb.AppendLine("  • Lower C → simpler model, more regularization");
        sb.AppendLine("  • Higher C → more complex model, less regularization");
        sb.AppendLine();
        sb.AppendLine("**gamma:** 'scale' (alternatives: 'auto', 0.001-0.1)");
        sb.AppendLine("  • Kernel coefficient for RBF");
        sb.AppendLine("  • 'scale' = 1 / (n_features * X.var()) - good default");
    }
    private void GenerateLinearModelRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures, string modelType)
    {
        sb.AppendLine("**Recommended Hyperparameters:**\n");
        double alpha = nSamples switch
        {
            < 100 => 10.0,
            < 1000 => 1.0,
            _ => 0.1
        };
        if (modelType.ToLowerInvariant().Contains("ridge") || modelType.ToLowerInvariant().Contains("lasso"))
        {
            sb.AppendLine($"**alpha:** {alpha} (range: {alpha / 100:F4}-{alpha * 100})");
            sb.AppendLine("  • Regularization strength");
            sb.AppendLine("  • Higher α → more regularization, simpler model");
            sb.AppendLine($"  • {alpha} appropriate for {nSamples} samples, {nFeatures} features");
        }
        if (modelType.ToLowerInvariant().Contains("elasticnet"))
        {
            sb.AppendLine($"**alpha:** {alpha}");
            sb.AppendLine("**l1_ratio:** 0.5 (range: 0-1)");
            sb.AppendLine("  • Mix of L1 (Lasso) and L2 (Ridge) regularization");
            sb.AppendLine("  • 0.5 = equal mix; 1.0 = pure Lasso; 0.0 = pure Ridge");
        }
    }
    private void GenerateDecisionTreeRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures, string complexity)
    {
        int maxDepth = complexity switch
        {
            "low" => 5,
            "moderate" => 10,
            _ => 15
        };
        sb.AppendLine("**Recommended Hyperparameters:**\n");
        sb.AppendLine($"**max_depth:** {maxDepth} (range: {maxDepth - 3}-{maxDepth + 5})");
        sb.AppendLine("  • Maximum depth of the tree");
        sb.AppendLine("  • Prevents overfitting by limiting tree growth");
        sb.AppendLine();
        sb.AppendLine("**min_samples_split:** 10 (range: 2-50)");
        sb.AppendLine("  • Minimum samples to split a node");
        sb.AppendLine("  • Higher → more conservative, less overfitting");
    }
    private void GenerateKNNRecommendations(System.Text.StringBuilder sb, int nSamples, int nFeatures)
    {
        int k = (int)Math.Sqrt(nSamples);
        k = Math.Max(3, Math.Min(k, 20));
        sb.AppendLine("**Recommended Hyperparameters:**\n");
        sb.AppendLine($"**n_neighbors:** {k} (range: 3-{k * 2})");
        sb.AppendLine($"  • Number of neighbors to consider");
        sb.AppendLine($"  • Rule of thumb: sqrt(n_samples) ≈ {k}");
        sb.AppendLine("  • Odd numbers avoid ties in classification");
        sb.AppendLine();
        sb.AppendLine("**weights:** 'distance' (alternative: 'uniform')");
        sb.AppendLine("  • Weight function: closer neighbors have more influence");
        sb.AppendLine("  • 'distance' generally better than uniform weighting");
    }
}
