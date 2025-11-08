using AiDotNet.Enums;

namespace AiDotNet.Agents;

/// <summary>
/// Contains the agent's recommendations after analyzing the data and configuration.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class AgentRecommendation<T, TInput, TOutput>
{
    /// <summary>
    /// Data analysis insights from the agent.
    /// </summary>
    public string? DataAnalysis { get; set; }

    /// <summary>
    /// Suggested model type from the ModelType enum.
    /// </summary>
    public ModelType? SuggestedModelType { get; set; }

    /// <summary>
    /// Agent's reasoning for model selection.
    /// </summary>
    public string? ModelSelectionReasoning { get; set; }

    /// <summary>
    /// Suggested hyperparameter values.
    /// </summary>
    public Dictionary<string, object>? SuggestedHyperparameters { get; set; }

    /// <summary>
    /// Agent's reasoning for hyperparameter tuning.
    /// </summary>
    public string? TuningReasoning { get; set; }

    /// <summary>
    /// Feature selection and importance recommendations.
    /// </summary>
    public string? FeatureRecommendations { get; set; }

    /// <summary>
    /// Complete reasoning trace from all agent operations.
    /// </summary>
    public string? ReasoningTrace { get; set; }
}
