namespace AiDotNet.Agents;

/// <summary>
/// Options for customizing what the agent helps with during building.
/// </summary>
public class AgentAssistanceOptions
{
    /// <summary>
    /// Enable agent to analyze data characteristics.
    /// </summary>
    public bool EnableDataAnalysis { get; set; } = true;

    /// <summary>
    /// Enable agent to suggest model types if not already configured.
    /// </summary>
    public bool EnableModelSelection { get; set; } = true;

    /// <summary>
    /// Enable agent to recommend hyperparameter values.
    /// </summary>
    public bool EnableHyperparameterTuning { get; set; } = false;

    /// <summary>
    /// Enable agent to analyze feature importance and suggest feature selection.
    /// </summary>
    public bool EnableFeatureAnalysis { get; set; } = false;

    /// <summary>
    /// Enable agent to provide advice for meta-learning configurations.
    /// </summary>
    public bool EnableMetaLearningAdvice { get; set; } = false;

    /// <summary>
    /// Default options: data analysis and model selection enabled.
    /// </summary>
    public static AgentAssistanceOptions Default => new AgentAssistanceOptions
    {
        EnableDataAnalysis = true,
        EnableModelSelection = true,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Minimal options: only model selection enabled.
    /// </summary>
    public static AgentAssistanceOptions Minimal => new AgentAssistanceOptions
    {
        EnableDataAnalysis = false,
        EnableModelSelection = true,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Comprehensive options: everything enabled.
    /// </summary>
    public static AgentAssistanceOptions Comprehensive => new AgentAssistanceOptions
    {
        EnableDataAnalysis = true,
        EnableModelSelection = true,
        EnableHyperparameterTuning = true,
        EnableFeatureAnalysis = true,
        EnableMetaLearningAdvice = true
    };

    /// <summary>
    /// Creates a new options builder for fluent configuration.
    /// </summary>
    public static AgentAssistanceOptionsBuilder Create() => new AgentAssistanceOptionsBuilder();
}
