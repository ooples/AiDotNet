namespace AiDotNet.Models;

/// <summary>
/// Configures which types of assistance the AI agent provides during model building.
/// </summary>
/// <remarks>
/// <para>
/// This class allows fine-grained control over which aspects of model building the AI agent assists with.
/// Each property enables or disables a specific capability, allowing you to balance between AI assistance
/// and manual control based on your needs. The class provides three predefined configurations (Default,
/// Minimal, and Comprehensive) as well as the ability to create custom configurations using the fluent
/// builder pattern.
/// </para>
/// <para><b>For Beginners:</b> This class lets you choose exactly what kind of help you want from the AI agent.
///
/// Think of it like choosing which tasks to delegate to an assistant:
/// - **Data Analysis**: Agent examines your data and reports on its characteristics
/// - **Model Selection**: Agent recommends which machine learning algorithm to use
/// - **Hyperparameter Tuning**: Agent suggests optimal settings for your chosen model
/// - **Feature Analysis**: Agent identifies which input variables are most important
/// - **Meta-Learning Advice**: Agent provides guidance on advanced few-shot learning setups
///
/// Common usage patterns:
/// - **Default**: Good starting point, enables basic data analysis and model selection
/// - **Minimal**: Only helps choose a model, gives you full control over everything else
/// - **Comprehensive**: Maximum assistance, agent helps with every aspect
/// - **Custom**: Use the Create() builder to enable only specific features you want
///
/// For example, if you're new to machine learning, you might use Comprehensive to get maximum help.
/// If you're experienced but want validation on model choice, you might use Minimal. If you want help
/// with everything except hyperparameters (which you want to tune yourself), you'd create a custom
/// configuration.
/// </para>
/// </remarks>
public class AgentAssistanceOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether the agent should analyze data characteristics.
    /// </summary>
    /// <value>True to enable data analysis; false to disable. Default is true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the agent examines your dataset and provides insights about its characteristics,
    /// such as distribution patterns, potential outliers, missing values, and data quality issues.
    /// This analysis helps identify potential problems before training begins.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the agent to look at your data and report what it finds.
    ///
    /// What the agent checks:
    /// - How many samples and features you have
    /// - Whether features are on similar scales
    /// - If there are outliers (unusual values)
    /// - Distribution patterns (normal, skewed, etc.)
    /// - Potential data quality issues
    ///
    /// This is useful because:
    /// - You might discover problems before wasting time training
    /// - The agent can warn about imbalanced data or scaling issues
    /// - You learn about your data's characteristics
    ///
    /// Example: The agent might report "Your dataset has 1000 samples with 20 features. Feature 'income'
    /// has much larger values than other features and should be normalized. Detected 5 potential outliers
    /// in feature 'age'."
    /// </para>
    /// </remarks>
    public bool EnableDataAnalysis { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether the agent should suggest model types.
    /// </summary>
    /// <value>True to enable model selection assistance; false to disable. Default is true.</value>
    /// <remarks>
    /// <para>
    /// When enabled and no model has been explicitly configured, the agent analyzes your data characteristics
    /// and recommends which machine learning algorithm would be most appropriate. The recommendation is based
    /// on dataset size, feature count, problem complexity, and expected performance trade-offs.
    /// </para>
    /// <para><b>For Beginners:</b> This lets the agent recommend which type of machine learning model to use.
    ///
    /// Why this matters:
    /// - There are many different ML algorithms (linear regression, random forests, neural networks, etc.)
    /// - Each works best for different types of problems
    /// - Choosing the wrong model can give poor results
    /// - The agent considers your data's characteristics to make a smart recommendation
    ///
    /// What the agent considers:
    /// - How much data you have (small datasets → simpler models)
    /// - Number of features (many features → models that handle high dimensions well)
    /// - Data patterns (linear relationships → linear models, complex patterns → tree-based or neural networks)
    /// - Your need for interpretability vs. accuracy
    ///
    /// For example, with a small dataset of house prices based on size and location, the agent might
    /// recommend linear regression. For a large dataset with complex relationships, it might suggest
    /// a random forest or gradient boosting model.
    /// </para>
    /// </remarks>
    public bool EnableModelSelection { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether the agent should recommend hyperparameter values.
    /// </summary>
    /// <value>True to enable hyperparameter tuning assistance; false to disable. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the agent suggests optimal hyperparameter values for your chosen model based on
    /// dataset characteristics and best practices. Hyperparameters control how the model learns and
    /// can significantly impact performance. This feature is disabled by default as it requires more
    /// computational analysis and may not be necessary for all use cases.
    /// </para>
    /// <para><b>For Beginners:</b> Hyperparameters are settings that control how your model learns.
    ///
    /// Think of hyperparameters like knobs on a radio:
    /// - **Learning Rate**: How quickly the model adjusts (too fast = unstable, too slow = takes forever)
    /// - **Regularization**: How much to penalize complex models (prevents overfitting)
    /// - **Tree Depth**: For tree-based models, how detailed the decisions can get
    /// - **Number of Neurons**: For neural networks, how many processing units to use
    ///
    /// Why enable this:
    /// - Default hyperparameters often aren't optimal for your specific data
    /// - Bad hyperparameters can dramatically hurt performance
    /// - Finding good values usually requires trial and error
    /// - The agent can suggest good starting points based on your data
    ///
    /// Why it's disabled by default:
    /// - It requires more time and analysis
    /// - Many datasets work fine with default settings
    /// - Advanced users often want to tune these themselves
    ///
    /// Enable this when: You want to squeeze out maximum performance or when default settings aren't
    /// working well.
    /// </para>
    /// </remarks>
    public bool EnableHyperparameterTuning { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether the agent should analyze feature importance.
    /// </summary>
    /// <value>True to enable feature analysis; false to disable. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the agent evaluates which input features are most important for making predictions
    /// and may suggest feature selection strategies. This can improve model performance, reduce training
    /// time, and increase interpretability by focusing on the most relevant features.
    /// </para>
    /// <para><b>For Beginners:</b> Features are the input variables you use to make predictions.
    ///
    /// What the agent does:
    /// - Identifies which features actually matter for predictions
    /// - Detects features that add no value (can be removed)
    /// - Finds redundant features (similar information)
    /// - Suggests feature engineering opportunities
    ///
    /// Why this matters:
    /// - More features isn't always better
    /// - Irrelevant features add noise and slow down training
    /// - Understanding which features matter helps you understand your problem
    /// - You might be able to collect less data if you know what's important
    ///
    /// For example, when predicting house prices:
    /// - The agent might find that square footage and location are critical
    /// - The color of the front door probably doesn't matter
    /// - Number of bedrooms might be redundant with square footage
    ///
    /// Why it's disabled by default:
    /// - Feature analysis requires additional computation
    /// - Not all problems benefit from feature selection
    /// - You might want to keep all features even if some are less important
    ///
    /// Enable this when: You have many features and suspect some aren't useful, or you want to understand
    /// what drives your predictions.
    /// </para>
    /// </remarks>
    public bool EnableFeatureAnalysis { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether the agent should provide meta-learning configuration advice.
    /// </summary>
    /// <value>True to enable meta-learning assistance; false to disable. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the agent provides guidance on meta-learning configurations such as task distribution,
    /// adaptation strategies, and inner/outer loop settings. This is an advanced feature primarily useful
    /// when working with few-shot learning scenarios where models must quickly adapt to new tasks.
    /// </para>
    /// <para><b>For Beginners:</b> Meta-learning is an advanced technique where models "learn to learn."
    ///
    /// What is meta-learning:
    /// - Regular learning: Train a model on lots of data for one specific task
    /// - Meta-learning: Train a model on many different tasks so it can quickly learn new tasks with minimal data
    /// - It's like teaching someone how to learn quickly rather than teaching them one specific thing
    ///
    /// When you'd use this:
    /// - You have many related tasks (e.g., image classification for different product categories)
    /// - Each task has only a few examples (few-shot learning)
    /// - You need models that can adapt quickly to new scenarios
    /// - You're working on personalization or rapid adaptation problems
    ///
    /// What the agent helps with:
    /// - How to structure your tasks for meta-learning
    /// - Which meta-learning algorithm to use (MAML, Reptile, etc.)
    /// - How many adaptation steps to use
    /// - Balance between learning speed and final performance
    ///
    /// Why it's disabled by default:
    /// - This is an advanced technique not needed for most projects
    /// - Regular supervised learning is sufficient for most use cases
    /// - Requires understanding of meta-learning concepts
    ///
    /// Only enable this if: You're specifically working on few-shot learning or need models that rapidly
    /// adapt to new tasks.
    /// </para>
    /// </remarks>
    public bool EnableMetaLearningAdvice { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether agent-recommended hyperparameters should be automatically
    /// applied to the model's options before training.
    /// </summary>
    /// <value>True to auto-apply recommended hyperparameters; false to store them for review only. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled and EnableHyperparameterTuning is also enabled, the AI agent's hyperparameter
    /// recommendations are automatically applied to the model's configuration before training begins.
    /// The results of the application (what was applied, skipped, or failed) are available in the
    /// AgentRecommendation.HyperparameterApplicationResult property. This feature requires the model
    /// to implement IConfigurableModel&lt;T&gt;.
    /// </para>
    /// <para><b>For Beginners:</b> When the AI agent suggests settings like "use 100 trees with depth 15",
    /// this option controls whether those settings are automatically applied to your model.
    ///
    /// - **Disabled (default)**: The agent's suggestions are stored for you to review, but your model
    ///   keeps its current settings. You can read the recommendations and apply them manually.
    /// - **Enabled**: The agent's suggestions are automatically applied to your model before training.
    ///   You can check what was applied in the results.
    ///
    /// This is disabled by default because:
    /// - You may want to review recommendations before applying them
    /// - Not all models support all hyperparameters
    /// - You might want to selectively apply only some recommendations
    ///
    /// Enable this when: You trust the agent's recommendations and want a fully automated workflow.
    /// </para>
    /// </remarks>
    public bool EnableAutoApplyHyperparameters { get; set; } = false;

    /// <summary>
    /// Gets a predefined configuration with data analysis and model selection enabled.
    /// </summary>
    /// <value>An AgentAssistanceOptions instance with default settings suitable for most users.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a balanced preset that enables the most commonly useful features.
    ///
    /// What's enabled:
    /// - ✓ Data analysis (helps you understand your data)
    /// - ✓ Model selection (recommends which algorithm to use)
    /// - ✗ Hyperparameter tuning (uses default settings)
    /// - ✗ Feature analysis (keeps all features)
    /// - ✗ Meta-learning advice (not needed for standard problems)
    ///
    /// Use this when: You're getting started and want helpful guidance without overwhelming detail.
    /// It's a good balance between AI assistance and keeping things simple.
    /// </para>
    /// </remarks>
    public static AgentAssistanceOptions Default => new AgentAssistanceOptions
    {
        EnableDataAnalysis = true,
        EnableModelSelection = true,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Gets a predefined configuration with only model selection enabled.
    /// </summary>
    /// <value>An AgentAssistanceOptions instance with minimal assistance suitable for experienced users.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This preset gives you just one piece of help: choosing which model to use.
    ///
    /// What's enabled:
    /// - ✗ Data analysis (you'll analyze data yourself)
    /// - ✓ Model selection (agent recommends an algorithm)
    /// - ✗ Hyperparameter tuning (you'll tune manually)
    /// - ✗ Feature analysis (you'll handle features yourself)
    /// - ✗ Meta-learning advice (not needed)
    ///
    /// Use this when: You're experienced with ML and only want validation on which model type to use.
    /// Everything else you'll handle yourself. This gives you maximum control with one helpful suggestion.
    /// </para>
    /// </remarks>
    public static AgentAssistanceOptions Minimal => new AgentAssistanceOptions
    {
        EnableDataAnalysis = false,
        EnableModelSelection = true,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Gets a predefined configuration with all assistance features enabled.
    /// </summary>
    /// <value>An AgentAssistanceOptions instance with maximum assistance suitable for getting the best results.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This preset turns on every available AI assistance feature for maximum help.
    ///
    /// What's enabled:
    /// - ✓ Data analysis (understands your data)
    /// - ✓ Model selection (picks the best algorithm)
    /// - ✓ Hyperparameter tuning (optimizes settings)
    /// - ✓ Feature analysis (identifies important variables)
    /// - ✓ Meta-learning advice (if using meta-learning)
    ///
    /// Use this when:
    /// - You want the AI to help with every decision
    /// - You're working on an important project and want maximum performance
    /// - You want to learn what the AI recommends for each aspect
    /// - You're new to ML and want comprehensive guidance
    ///
    /// Note: This will make model building take longer because the agent does more analysis, but you'll
    /// get more insights and potentially better results.
    /// </para>
    /// </remarks>
    public static AgentAssistanceOptions Comprehensive => new AgentAssistanceOptions
    {
        EnableDataAnalysis = true,
        EnableModelSelection = true,
        EnableHyperparameterTuning = true,
        EnableFeatureAnalysis = true,
        EnableMetaLearningAdvice = true,
        EnableAutoApplyHyperparameters = true
    };

    /// <summary>
    /// Creates a new fluent builder for custom agent assistance configuration.
    /// </summary>
    /// <returns>A new AgentAssistanceOptionsBuilder instance for fluent configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to create a custom configuration by picking exactly which features you want.
    ///
    /// Example of creating a custom configuration:
    /// <code>
    /// var options = AgentAssistanceOptions.Create()
    ///     .EnableDataAnalysis()
    ///     .EnableModelSelection()
    ///     .EnableFeatureAnalysis()
    ///     .Build();
    /// </code>
    ///
    /// This gives you precise control - maybe you want data analysis and model selection but not hyperparameter
    /// tuning. The fluent builder makes it easy to enable exactly what you want.
    /// </para>
    /// </remarks>
    public static AgentAssistanceOptionsBuilder Create() => new AgentAssistanceOptionsBuilder();

    /// <summary>
    /// Creates a deep copy of this options instance.
    /// </summary>
    /// <returns>A new AgentAssistanceOptions instance with the same property values.</returns>
    internal AgentAssistanceOptions Clone() => new AgentAssistanceOptions
    {
        EnableDataAnalysis = this.EnableDataAnalysis,
        EnableModelSelection = this.EnableModelSelection,
        EnableHyperparameterTuning = this.EnableHyperparameterTuning,
        EnableFeatureAnalysis = this.EnableFeatureAnalysis,
        EnableMetaLearningAdvice = this.EnableMetaLearningAdvice,
        EnableAutoApplyHyperparameters = this.EnableAutoApplyHyperparameters
    };
}
