using AiDotNet.Enums;

namespace AiDotNet.Models;

/// <summary>
/// Stores the AI agent's analysis results and recommendations after examining your data and model configuration.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates all the insights, recommendations, and reasoning provided by the AI agent during the
/// model building process. It contains the results of various agent operations such as data analysis, model selection,
/// hyperparameter tuning, and feature analysis. Each property corresponds to a specific agent capability that can be
/// enabled through AgentAssistanceOptions. Properties will only be populated if their corresponding feature was enabled;
/// otherwise, they remain null. This class is returned as part of the AiModelResult and provides valuable
/// insights into why the agent made specific recommendations.
/// </para>
/// <para><b>For Beginners:</b> This class holds all the advice and recommendations the AI agent provides about your
/// machine learning model.
///
/// Think of it as a detailed report from an AI consultant that contains:
/// - **Analysis of your data**: What patterns, issues, or characteristics the AI found
/// - **Model recommendations**: Which model to use and why
/// - **Setting suggestions**: What hyperparameters to use and why those values make sense
/// - **Feature insights**: Which input variables matter most and how to improve them
/// - **Reasoning explanations**: The AI's thought process behind each recommendation
///
/// How it works:
/// - Only the sections you enabled in AgentAssistanceOptions will be filled in
/// - Each property contains detailed, human-readable explanations
/// - You can read these insights to understand the AI's recommendations
/// - The reasoning helps you learn about machine learning best practices
///
/// For example, if you enabled ModelSelection and HyperparameterTuning, you might get:
/// - SuggestedModelType: RandomForest
/// - ModelSelectionReasoning: "Random Forest is recommended because your dataset has 10,000 samples with
///   complex non-linear relationships. It will handle the outliers better than linear models and is less
///   likely to overfit than a deep neural network given your data size."
/// - SuggestedHyperparameters: {"n_trees": 100, "max_depth": 15, "min_samples_split": 10}
/// - TuningReasoning: "These hyperparameters balance model complexity with generalization..."
///
/// This information helps you understand not just what to do, but why it's the right approach for your specific problem.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters and calculations (typically float or double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;double&gt;, Vector&lt;float&gt;).</typeparam>
/// <typeparam name="TOutput">The type of output predictions (e.g., Vector&lt;double&gt;, double[]).</typeparam>
public class AgentRecommendation<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the AI agent's analysis of your training data, including patterns, issues, and characteristics.
    /// </summary>
    /// <value>A detailed text description of data analysis insights, or null if data analysis was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains the agent's comprehensive analysis of your training data when EnableDataAnalysis is turned on.
    /// The analysis includes observations about data distributions, outliers, missing values, feature correlations, class
    /// imbalances, and potential data quality issues. The agent provides this information in natural language with specific
    /// examples and statistics from your data. This property is null if data analysis was not enabled in AgentAssistanceOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This contains what the AI discovered when examining your data.
    ///
    /// What you'll find here:
    /// - Unusual patterns or anomalies in your data
    /// - Statistics about your features and target variable
    /// - Warnings about potential problems (missing data, outliers, etc.)
    /// - Observations about feature relationships and correlations
    /// - Suggestions for data preprocessing or cleaning
    ///
    /// For example, you might see:
    /// "Your dataset contains 5,000 samples with 12 features. The 'price' feature shows significant right skew with
    /// some extreme outliers above $2M. Features 'sqft' and 'bedrooms' are highly correlated (r=0.85). About 8% of
    /// 'year_built' values are missing. Consider log-transforming 'price' to handle the skew, and either remove or
    /// impute the missing values before training."
    ///
    /// This helps you understand your data better and catch issues before they affect your model's performance.
    /// </para>
    /// </remarks>
    public string? DataAnalysis { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's recommended model type for your problem.
    /// </summary>
    /// <value>A ModelType enum value representing the recommended model, or null if model selection was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains the specific model type that the AI agent recommends based on your data characteristics,
    /// problem complexity, and dataset size. The recommendation is made by analyzing factors such as linearity, feature
    /// dimensionality, sample size, presence of outliers, and computational constraints. This property is null if model
    /// selection was not enabled in AgentAssistanceOptions. See the ModelSelectionReasoning property for the explanation
    /// of why this model was chosen.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you which machine learning model the AI thinks you should use.
    ///
    /// The AI chooses based on:
    /// - Your data size (small, medium, or large dataset)
    /// - Complexity of patterns in your data
    /// - Type of relationships (linear, non-linear, etc.)
    /// - Presence of outliers or unusual data points
    ///
    /// For example, the AI might recommend:
    /// - SimpleRegression for small datasets with clear linear relationships
    /// - RandomForest for medium-sized datasets with complex patterns
    /// - NeuralNetworkRegression for large datasets with very complex relationships
    ///
    /// Always read the ModelSelectionReasoning property to understand why this model was chosen for your specific situation.
    /// </para>
    /// </remarks>
    public ModelType? SuggestedModelType { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's detailed explanation for why it recommended the suggested model type.
    /// </summary>
    /// <value>A detailed text explanation of the model selection rationale, or null if model selection was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains the agent's natural language explanation of why it recommended the SuggestedModelType.
    /// The reasoning describes how the model's characteristics align with your data properties, explains what makes
    /// this model a good fit, discusses potential alternatives that were considered, and may warn about limitations
    /// or considerations to keep in mind. This educational explanation helps you understand the decision-making process
    /// rather than just receiving a recommendation blindly. This property is null if model selection was not enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This explains why the AI chose the recommended model - the "why" behind the "what."
    ///
    /// The reasoning typically covers:
    /// - Why this model suits your specific data characteristics
    /// - What advantages this model has for your problem
    /// - What alternatives were considered and why they weren't chosen
    /// - Any limitations or caveats to be aware of
    /// - How this model compares to others for your use case
    ///
    /// For example:
    /// "Random Forest is recommended for your dataset because:
    /// 1. Your 10,000 samples provide enough data for an ensemble method
    /// 2. The non-linear relationships in your features need a flexible model
    /// 3. Random Forest handles the outliers in 'price' better than linear models
    /// 4. It's less prone to overfitting than a single decision tree
    /// 5. It provides feature importance scores which will be valuable for your analysis
    ///
    /// While Neural Networks could also work, Random Forest will likely train faster and be easier to interpret
    /// for your business stakeholders."
    ///
    /// This helps you learn about model selection while getting personalized advice for your situation.
    /// </para>
    /// </remarks>
    public string? ModelSelectionReasoning { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's recommended hyperparameter values for the selected model.
    /// </summary>
    /// <value>A dictionary mapping hyperparameter names to their recommended values, or null if hyperparameter tuning was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains a dictionary of hyperparameter names and their recommended values for your selected model.
    /// The keys are hyperparameter names (like "learning_rate", "max_depth", "n_estimators") and the values are the
    /// suggested settings. These recommendations are tailored to your specific data characteristics and model type.
    /// The agent considers factors like dataset size, feature dimensionality, and problem complexity when suggesting
    /// these values. This property is null if hyperparameter tuning was not enabled in AgentAssistanceOptions. See
    /// TuningReasoning for detailed explanations of why these values were chosen.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the recommended settings for your model - like tuning knobs on a machine.
    ///
    /// Hyperparameters control how your model learns:
    /// - How fast it learns (learning_rate)
    /// - How complex it can get (max_depth, n_layers)
    /// - How many parts it has (n_estimators, n_neurons)
    /// - How it handles data (batch_size, regularization)
    ///
    /// For example, if using Random Forest, you might get:
    /// {
    ///   "n_estimators": 100,        // Use 100 trees
    ///   "max_depth": 15,            // Each tree can be 15 levels deep
    ///   "min_samples_split": 10,    // Need at least 10 samples to split
    ///   "max_features": "sqrt"      // Consider sqrt(n) features per split
    /// }
    ///
    /// These aren't random guesses - the AI chose these values specifically for your data size and characteristics.
    /// Check TuningReasoning to understand why each value was selected.
    /// </para>
    /// </remarks>
    public Dictionary<string, object>? SuggestedHyperparameters { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's detailed explanation for the recommended hyperparameter values.
    /// </summary>
    /// <value>A detailed text explanation of the hyperparameter tuning rationale, or null if hyperparameter tuning was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains the agent's natural language explanation of why it chose the specific hyperparameter values
    /// in SuggestedHyperparameters. For each major hyperparameter, the reasoning explains what the parameter controls,
    /// how it affects model behavior, why the suggested value is appropriate for your data, and what trade-offs were
    /// considered. This educational explanation helps you understand hyperparameter tuning principles rather than just
    /// applying recommended values blindly. This property is null if hyperparameter tuning was not enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This explains why the AI chose these specific hyperparameter values.
    ///
    /// For each important setting, you'll learn:
    /// - What the hyperparameter does and why it matters
    /// - Why this specific value is good for your data
    /// - What happens if you increase or decrease the value
    /// - How it balances different trade-offs (speed vs accuracy, simplicity vs power)
    ///
    /// For example:
    /// "Hyperparameter recommendations for your Random Forest:
    ///
    /// n_estimators=100: Using 100 trees provides a good balance. More trees improve accuracy but with diminishing
    /// returns beyond 100 for your dataset size. Fewer trees might underfit your complex patterns.
    ///
    /// max_depth=15: Limiting tree depth prevents overfitting. Your dataset's complexity needs depth to capture
    /// patterns, but unlimited depth would memorize noise in your 5,000 training samples.
    ///
    /// min_samples_split=10: Requiring 10 samples before splitting helps avoid creating overly specific rules
    /// based on just a few data points, which improves generalization to new data.
    ///
    /// These values are tuned for your specific dataset size and complexity. If you have more data later,
    /// you could increase max_depth to capture more detailed patterns."
    ///
    /// This helps you understand hyperparameter tuning and make informed adjustments if needed.
    /// </para>
    /// </remarks>
    public string? TuningReasoning { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's recommendations about feature selection, importance, and engineering.
    /// </summary>
    /// <value>A detailed text description of feature analysis and recommendations, or null if feature analysis was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains the agent's analysis of your input features when EnableFeatureAnalysis is turned on.
    /// It includes rankings of feature importance, identification of redundant or unhelpful features, suggestions
    /// for new features to create through feature engineering, recommendations for feature transformations, and
    /// insights about feature interactions. The agent explains which features are most predictive, which could be
    /// removed without hurting performance, and how you might improve your feature set. This property is null if
    /// feature analysis was not enabled in AgentAssistanceOptions.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you which input variables (features) matter most and how to improve them.
    ///
    /// What you'll learn:
    /// - Which features are most important for predictions
    /// - Which features aren't helping and could be removed
    /// - Ideas for new features to create (feature engineering)
    /// - Transformations that might improve feature usefulness
    /// - How features interact with each other
    ///
    /// For example, when predicting house prices:
    /// "Feature Analysis:
    ///
    /// Most Important Features:
    /// 1. square_footage (importance: 0.45) - Strongest predictor of price
    /// 2. location_score (importance: 0.28) - Second most influential
    /// 3. year_built (importance: 0.12) - Moderate impact
    ///
    /// Low-Impact Features:
    /// - exterior_color (importance: 0.01) - Consider removing
    /// - HOA_name (importance: 0.00) - Not predictive, remove
    ///
    /// Feature Engineering Suggestions:
    /// - Create 'price_per_sqft' = price / square_footage (useful for normalization)
    /// - Create 'age' = current_year - year_built (more intuitive than year_built)
    /// - Consider binning 'year_built' into decades for non-linear patterns
    ///
    /// The strong correlation between square_footage and bedrooms (r=0.85) suggests multicollinearity.
    /// Consider using just one or creating a composite feature."
    ///
    /// This helps you focus on important features and improve your model's performance through better feature engineering.
    /// </para>
    /// </remarks>
    public string? FeatureRecommendations { get; set; }

    /// <summary>
    /// Gets or sets the complete reasoning trace showing the AI agent's thought process across all operations.
    /// </summary>
    /// <value>A comprehensive text record of the agent's reasoning steps and decisions, or null if no agent operations were performed.</value>
    /// <remarks>
    /// <para>
    /// This property contains a chronological trace of all reasoning steps performed by the AI agent during the model
    /// building process. It combines insights from all enabled agent operations into a single comprehensive narrative,
    /// showing how the agent analyzed your data, considered different options, made decisions, and arrived at its
    /// recommendations. This is particularly useful for understanding the complete picture of the agent's analysis
    /// and for debugging or learning purposes. This property is null if no agent assistance features were enabled.
    /// </para>
    /// <para><b>For Beginners:</b> This is the complete record of everything the AI thought about and decided during
    /// the model building process.
    ///
    /// It's like reading the AI's notebook where it:
    /// - Recorded observations about your data
    /// - Explained its decision-making process
    /// - Showed how different pieces of analysis connect
    /// - Documented why it made each recommendation
    ///
    /// This is the most detailed view of the agent's work. While the other properties give you specific pieces
    /// (data analysis, model choice, hyperparameters, features), this property shows the complete story of how
    /// all those pieces fit together.
    ///
    /// Use this when you want to:
    /// - Understand the full picture of the AI's analysis
    /// - Learn about the machine learning process
    /// - Debug unexpected recommendations
    /// - See how the AI connects different insights
    ///
    /// For example, it might show:
    /// "Step 1: Analyzed data distribution and found right-skewed target variable...
    /// Step 2: Based on skewness and dataset size, considered RandomForest and GradientBoosting...
    /// Step 3: Selected RandomForest because it's more robust to outliers...
    /// Step 4: Tuned hyperparameters considering the dataset size of 5,000 samples...
    /// Step 5: Identified square_footage as the most important feature due to high correlation with target..."
    ///
    /// This comprehensive view helps you learn how an expert approaches machine learning problems.
    /// </para>
    /// </remarks>
    public string? ReasoningTrace { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's recommended compression technique for model deployment.
    /// </summary>
    /// <value>The suggested compression type, or null if compression analysis was not enabled.</value>
    /// <remarks>
    /// <para>
    /// This property contains the specific compression technique that the AI agent recommends
    /// based on the model architecture, deployment constraints, and accuracy requirements.
    /// The recommendation considers factors like model size, layer types, and target platform.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you which compression method to use to make your model smaller.
    ///
    /// Common recommendations include:
    /// - **SparsePruning**: Remove small weights (good for large, dense models)
    /// - **WeightClustering**: Group similar weights (good for moderate compression)
    /// - **DeepCompression**: Combine multiple techniques (best for maximum compression)
    ///
    /// The agent chooses based on:
    /// - Your model's architecture (CNNs vs fully-connected)
    /// - Your accuracy tolerance (how much accuracy loss is acceptable)
    /// - Your deployment target (mobile, cloud, edge devices)
    /// </para>
    /// </remarks>
    public Enums.CompressionType? SuggestedCompressionType { get; set; }

    /// <summary>
    /// Gets or sets the AI agent's detailed explanation for the compression recommendation.
    /// </summary>
    /// <value>A detailed text explanation of the compression selection rationale, or null if not available.</value>
    /// <remarks>
    /// <para>
    /// This property explains why the agent recommended a specific compression technique,
    /// including analysis of the model structure and expected compression-accuracy tradeoffs.
    /// </para>
    /// <para><b>For Beginners:</b> This explains why a particular compression technique was chosen.
    ///
    /// For example:
    /// "Deep Compression (pruning + quantization + Huffman) is recommended because:
    /// 1. Your model has many fully-connected layers with high redundancy
    /// 2. 92% of weights have magnitude less than 0.1 (good pruning candidates)
    /// 3. Expected compression: 35-50x with less than 1% accuracy loss
    /// 4. Target platform (mobile) requires small model size
    ///
    /// Alternative considered: Weight Clustering alone would give 8x compression,
    /// but Deep Compression's multi-stage approach will achieve better results
    /// for your deployment constraints."
    /// </para>
    /// </remarks>
    public string? CompressionReasoning { get; set; }

    /// <summary>
    /// Gets or sets the recommended hyperparameters for the compression technique.
    /// </summary>
    /// <value>A dictionary mapping parameter names to recommended values, or null if not available.</value>
    /// <remarks>
    /// <para>
    /// This property contains the specific hyperparameter values recommended for the suggested
    /// compression technique. These are tuned based on the model characteristics and constraints.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings to use with the recommended compression technique.
    ///
    /// For example, for Deep Compression:
    /// - "pruningSparsity": 0.9 (remove 90% of smallest weights)
    /// - "numClusters": 32 (use 5-bit quantization)
    /// - "huffmanPrecision": 4 (4 decimal places for encoding)
    ///
    /// These values were chosen based on your model's characteristics to balance
    /// compression and accuracy.
    /// </para>
    /// </remarks>
    public Dictionary<string, object>? SuggestedCompressionParameters { get; set; }

    /// <summary>
    /// Gets or sets the expected compression metrics after applying the recommended technique.
    /// </summary>
    /// <value>Predicted compression metrics, or null if not available.</value>
    /// <remarks>
    /// <para>
    /// This property contains the agent's prediction of what compression metrics you can expect
    /// if you apply the recommended compression technique. These are estimates based on model analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This shows what to expect from compression before you apply it.
    ///
    /// Typical information includes:
    /// - Expected compression ratio (e.g., "20-30x smaller")
    /// - Expected accuracy loss (e.g., "less than 1%")
    /// - Expected inference speedup (e.g., "2-3x faster")
    ///
    /// These are estimates - actual results depend on your specific model and data.
    /// </para>
    /// </remarks>
    public string? ExpectedCompressionMetrics { get; set; }

    /// <summary>
    /// Gets or sets the results of auto-applying hyperparameter recommendations to the model.
    /// </summary>
    /// <value>
    /// A HyperparameterApplicationResult containing details about which parameters were applied,
    /// skipped, or failed, or null if auto-apply was not enabled or no hyperparameters were recommended.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property is populated when EnableAutoApplyHyperparameters is true and the model implements
    /// IConfigurableModel&lt;T&gt;. It provides a detailed accounting of the hyperparameter application
    /// process, including which parameters were successfully set, which had no matching property,
    /// and which failed due to type conversion or other errors.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you exactly what happened when the AI agent's
    /// hyperparameter recommendations were applied to your model:
    /// - **Applied**: Parameters that were successfully set (e.g., "NumberOfTrees = 100")
    /// - **Skipped**: Parameters the model doesn't support (e.g., "max_features" on a model that doesn't have it)
    /// - **Failed**: Parameters that couldn't be set due to errors
    /// - **Warnings**: Issues like values outside typical ranges
    ///
    /// Check result.HyperparameterApplicationResult.GetSummary() for a human-readable report.
    /// </para>
    /// </remarks>
    public HyperparameterApplicationResult? HyperparameterApplicationResult { get; set; }
}
