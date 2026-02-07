namespace AiDotNet.Models;

/// <summary>
/// Provides a fluent interface for configuring which AI agent assistance features should be enabled during model building.
/// </summary>
/// <remarks>
/// <para>
/// This builder class provides a chainable, fluent API for configuring agent assistance options. It allows you to
/// selectively enable or disable specific agent capabilities such as data analysis, model selection recommendations,
/// hyperparameter tuning suggestions, feature analysis, and meta-learning advice. By default, all features are disabled,
/// requiring explicit opt-in for each capability you want to use. This design gives you fine-grained control over which
/// AI assistance features are active and helps manage API costs by only enabling the features you need.
/// </para>
/// <para><b>For Beginners:</b> This class helps you choose which AI assistance features you want to use when building
/// your machine learning models.
///
/// Think of it as a configuration menu where you can turn on or off different AI helper features:
/// - **Data Analysis**: AI examines your data and points out patterns, issues, or characteristics
/// - **Model Selection**: AI recommends which type of model would work best for your problem
/// - **Hyperparameter Tuning**: AI suggests optimal settings for your chosen model
/// - **Feature Analysis**: AI analyzes which input features are most important
/// - **Meta-Learning Advice**: AI provides general best practices based on similar problems
///
/// Why use a builder:
/// - You can chain method calls together for clean, readable code
/// - You only pay for (and wait for) the AI features you actually use
/// - You can easily enable/disable features without changing lots of code
/// - All features start disabled for safety and cost control
///
/// For example, if you're building a house price prediction model and want AI help choosing the model
/// and tuning it, but don't need data analysis, you would enable just ModelSelection and HyperparameterTuning.
/// This keeps your API costs lower and makes the process faster.
/// </para>
/// </remarks>
public class AgentAssistanceOptionsBuilder
{
    private readonly AgentAssistanceOptions _options = new AgentAssistanceOptions
    {
        // Start with nothing enabled
        EnableDataAnalysis = false,
        EnableModelSelection = false,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Enables AI-powered data analysis that examines your dataset for patterns, anomalies, and characteristics.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When enabled, the AI agent will analyze your training data to identify characteristics such as data distributions,
    /// potential outliers, missing values, correlations between features, and other statistical properties. The agent
    /// provides insights about data quality issues and suggests preprocessing steps that might improve model performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the AI to examine your data and give you insights about it.
    ///
    /// What the AI looks for:
    /// - Unusual patterns or outliers (data points that don't fit the pattern)
    /// - Missing or invalid values
    /// - How your features relate to each other
    /// - Whether your data is balanced or skewed
    /// - Potential data quality issues
    ///
    /// This is useful when:
    /// - You're working with unfamiliar data
    /// - You want to catch data quality issues early
    /// - You need guidance on data preprocessing
    /// - You want to understand your data better before modeling
    ///
    /// For example, if you're predicting house prices, the AI might notice that 20% of your price values are
    /// missing, or that there's a very strong correlation between square footage and price.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptionsBuilder EnableDataAnalysis()
    {
        _options.EnableDataAnalysis = true;
        return this;
    }

    /// <summary>
    /// Enables AI-powered model selection recommendations based on your data characteristics and problem type.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When enabled, the AI agent will analyze your data and problem characteristics to recommend the most appropriate
    /// model type. The agent considers factors such as dataset size, feature dimensionality, linearity of relationships,
    /// presence of outliers, and computational constraints. It provides a recommended model type along with reasoning
    /// explaining why that model is a good fit for your specific use case.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the AI to suggest which type of model would work best for your problem.
    ///
    /// How it helps:
    /// - Recommends a specific model type (like Random Forest, Neural Network, etc.)
    /// - Explains why that model is a good choice for your data
    /// - Considers your data's characteristics (size, complexity, patterns)
    /// - Saves you from trying many models manually
    ///
    /// This is useful when:
    /// - You're not sure which model to use
    /// - You want expert guidance on model selection
    /// - You're new to machine learning
    /// - You want to avoid obviously poor model choices
    ///
    /// For example, if you have a small dataset with clear linear relationships, the AI might recommend
    /// Simple Regression and explain that more complex models would likely overfit with so little data.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptionsBuilder EnableModelSelection()
    {
        _options.EnableModelSelection = true;
        return this;
    }

    /// <summary>
    /// Enables AI-powered hyperparameter tuning suggestions to optimize model configuration.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When enabled, the AI agent will suggest optimal hyperparameter values for your selected model based on
    /// your data characteristics and problem type. Hyperparameters are settings that control how the model learns,
    /// such as learning rate, tree depth, number of neurons, etc. The agent provides specific recommended values
    /// along with explanations of how each parameter affects model performance and why those values are appropriate.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the AI to suggest the best settings for your chosen model.
    ///
    /// What are hyperparameters:
    /// - Settings that control how your model learns (like learning speed, complexity, etc.)
    /// - Different from the parameters the model learns during training
    /// - Can dramatically affect how well your model performs
    /// - Usually require experimentation to find good values
    ///
    /// How the AI helps:
    /// - Recommends specific values for each important setting
    /// - Explains what each setting does and why that value makes sense
    /// - Considers your data size, complexity, and characteristics
    /// - Saves you hours of manual tuning experiments
    ///
    /// This is useful when:
    /// - You're using a complex model with many settings
    /// - You don't know where to start with hyperparameter values
    /// - You want to get better results without extensive manual tuning
    /// - You want to understand what each setting does
    ///
    /// For example, if you're using a Random Forest, the AI might suggest using 100 trees with a max depth
    /// of 15, and explain that this balances model accuracy with avoiding overfitting for your dataset size.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptionsBuilder EnableHyperparameterTuning()
    {
        _options.EnableHyperparameterTuning = true;
        return this;
    }

    /// <summary>
    /// Enables AI-powered feature analysis to identify important variables and suggest feature engineering improvements.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When enabled, the AI agent will analyze your input features to identify which ones are most important for
    /// predictions, detect redundant or highly correlated features, and suggest feature engineering techniques that
    /// might improve model performance. This can include recommendations for creating new features, transforming
    /// existing ones, or removing unhelpful features that add noise without predictive value.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the AI to examine your input variables (features) and give advice about them.
    ///
    /// What the AI analyzes:
    /// - Which features are most important for making predictions
    /// - Which features might be redundant or unhelpful
    /// - How features relate to each other
    /// - Whether creating new combined features would help
    ///
    /// What you get:
    /// - Ranking of feature importance
    /// - Suggestions for new features to create (feature engineering)
    /// - Identification of features that could be removed
    /// - Recommendations for transforming features
    ///
    /// This is useful when:
    /// - You have many features and want to know which matter most
    /// - You suspect some features aren't helping
    /// - You want ideas for improving your feature set
    /// - You're trying to simplify your model without losing accuracy
    ///
    /// For example, when predicting house prices, the AI might tell you that square footage and location
    /// are very important, number of bathrooms is somewhat important, but exterior color doesn't matter at all.
    /// It might also suggest creating a new feature: "price per square foot" which could improve predictions.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptionsBuilder EnableFeatureAnalysis()
    {
        _options.EnableFeatureAnalysis = true;
        return this;
    }

    /// <summary>
    /// Enables AI-powered meta-learning advice that provides best practices and general recommendations.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// When enabled, the AI agent will provide high-level advice based on machine learning best practices and
    /// experiences with similar problems. This includes recommendations about data preprocessing, validation strategies,
    /// potential pitfalls to avoid, and general approaches that tend to work well for problems similar to yours.
    /// Meta-learning advice draws on patterns learned from many different machine learning projects.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the AI to share general wisdom and best practices for your type of problem.
    ///
    /// What you get:
    /// - General advice about your machine learning approach
    /// - Common mistakes to avoid for your type of problem
    /// - Recommended validation and testing strategies
    /// - Best practices from similar successful projects
    /// - Warnings about potential issues specific to your problem type
    ///
    /// This is useful when:
    /// - You're tackling a new type of machine learning problem
    /// - You want to avoid common beginner mistakes
    /// - You need guidance on overall strategy, not just technical details
    /// - You want to benefit from experience with similar problems
    ///
    /// For example, when building a time series forecasting model, the AI might advise:
    /// - Always split data chronologically, never randomly
    /// - Watch for seasonal patterns that repeat yearly
    /// - Consider testing on multiple time periods, not just one
    /// - Be careful about data leakage from future to past
    ///
    /// This higher-level guidance complements the more specific technical recommendations from other features.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptionsBuilder EnableMetaLearningAdvice()
    {
        _options.EnableMetaLearningAdvice = true;
        return this;
    }

    /// <summary>
    /// Disables AI-powered data analysis.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// This method turns off data analysis assistance, useful if you want to disable a previously enabled feature
    /// or ensure it's explicitly disabled even if it was enabled by default in a template configuration.
    /// </remarks>
    public AgentAssistanceOptionsBuilder DisableDataAnalysis()
    {
        _options.EnableDataAnalysis = false;
        return this;
    }

    /// <summary>
    /// Disables AI-powered model selection recommendations.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// This method turns off model selection assistance, useful if you already know which model you want to use
    /// or want to reduce API calls and costs by skipping this feature.
    /// </remarks>
    public AgentAssistanceOptionsBuilder DisableModelSelection()
    {
        _options.EnableModelSelection = false;
        return this;
    }

    /// <summary>
    /// Disables AI-powered hyperparameter tuning suggestions.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// This method turns off hyperparameter tuning assistance, useful if you want to use default hyperparameters
    /// or have your own tuning strategy you prefer to follow.
    /// </remarks>
    public AgentAssistanceOptionsBuilder DisableHyperparameterTuning()
    {
        _options.EnableHyperparameterTuning = false;
        return this;
    }

    /// <summary>
    /// Disables AI-powered feature analysis.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// This method turns off feature analysis assistance, useful if you've already performed feature engineering
    /// or want to skip this analysis to reduce processing time and API costs.
    /// </remarks>
    public AgentAssistanceOptionsBuilder DisableFeatureAnalysis()
    {
        _options.EnableFeatureAnalysis = false;
        return this;
    }

    /// <summary>
    /// Disables AI-powered meta-learning advice.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// This method turns off meta-learning advice, useful if you're experienced with this type of problem
    /// or want to focus only on specific technical recommendations rather than general guidance.
    /// </remarks>
    public AgentAssistanceOptionsBuilder DisableMetaLearningAdvice()
    {
        _options.EnableMetaLearningAdvice = false;
        return this;
    }

    /// <summary>
    /// Enables automatic application of agent-recommended hyperparameters to the model before training.
    /// Also enables hyperparameter tuning if not already enabled, since auto-apply requires recommendations.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells the AI agent to not only recommend hyperparameters
    /// but also automatically apply them to your model. This creates a fully automated workflow
    /// where the agent analyzes your data, suggests settings, and configures your model.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptionsBuilder EnableAutoApplyHyperparameters()
    {
        _options.EnableAutoApplyHyperparameters = true;
        _options.EnableHyperparameterTuning = true;
        return this;
    }

    /// <summary>
    /// Disables automatic application of agent-recommended hyperparameters.
    /// </summary>
    /// <returns>The current builder instance for method chaining.</returns>
    public AgentAssistanceOptionsBuilder DisableAutoApplyHyperparameters()
    {
        _options.EnableAutoApplyHyperparameters = false;
        return this;
    }

    /// <summary>
    /// Builds and returns the configured AgentAssistanceOptions instance.
    /// </summary>
    /// <returns>An AgentAssistanceOptions instance with all configured settings.</returns>
    /// <remarks>
    /// <para>
    /// This method finalizes the configuration and returns the AgentAssistanceOptions object that can be
    /// passed to the AiModelBuilder. You typically call this method at the end of your fluent chain.
    /// </para>
    /// <para><b>For Beginners:</b> This finalizes your configuration choices and creates the options object.
    ///
    /// You call this at the end of your configuration chain, like:
    /// <code>
    /// var options = new AgentAssistanceOptionsBuilder()
    ///     .EnableDataAnalysis()
    ///     .EnableModelSelection()
    ///     .Build();
    /// </code>
    ///
    /// The returned options object is then passed to ConfigureAgentAssistance() in your model builder.
    /// Returns a fresh copy to prevent external mutation of builder state.
    /// </para>
    /// </remarks>
    public AgentAssistanceOptions Build() => _options.Clone();

    /// <summary>
    /// Implicitly converts the builder to an AgentAssistanceOptions instance.
    /// </summary>
    /// <param name="builder">The builder to convert.</param>
    /// <returns>The configured AgentAssistanceOptions instance.</returns>
    /// <remarks>
    /// <para>
    /// This implicit conversion operator allows you to use the builder directly where an AgentAssistanceOptions
    /// is expected, without explicitly calling Build(). This makes the API more flexible and the code slightly cleaner.
    /// </para>
    /// <para><b>For Beginners:</b> This allows you to skip calling Build() in some cases.
    ///
    /// Because of this operator, both of these work the same:
    /// <code>
    /// // Explicit Build() call
    /// var options = new AgentAssistanceOptionsBuilder()
    ///     .EnableDataAnalysis()
    ///     .Build();
    ///
    /// // Implicit conversion (no Build() needed)
    /// AgentAssistanceOptions options = new AgentAssistanceOptionsBuilder()
    ///     .EnableDataAnalysis();
    /// </code>
    ///
    /// C# automatically calls this conversion when needed, making your code slightly cleaner.
    /// Returns a fresh copy to prevent external mutation of builder state.
    /// </para>
    /// </remarks>
    public static implicit operator AgentAssistanceOptions(AgentAssistanceOptionsBuilder builder)
        => builder._options.Clone();
}
