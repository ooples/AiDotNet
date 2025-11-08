namespace AiDotNet.Models;

/// <summary>
/// Fluent builder for agent assistance options.
/// </summary>
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

    public AgentAssistanceOptionsBuilder EnableDataAnalysis()
    {
        _options.EnableDataAnalysis = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableModelSelection()
    {
        _options.EnableModelSelection = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableHyperparameterTuning()
    {
        _options.EnableHyperparameterTuning = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableFeatureAnalysis()
    {
        _options.EnableFeatureAnalysis = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableMetaLearningAdvice()
    {
        _options.EnableMetaLearningAdvice = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableDataAnalysis()
    {
        _options.EnableDataAnalysis = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableModelSelection()
    {
        _options.EnableModelSelection = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableHyperparameterTuning()
    {
        _options.EnableHyperparameterTuning = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableFeatureAnalysis()
    {
        _options.EnableFeatureAnalysis = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableMetaLearningAdvice()
    {
        _options.EnableMetaLearningAdvice = false;
        return this;
    }

    public AgentAssistanceOptions Build() => _options;

    public static implicit operator AgentAssistanceOptions(AgentAssistanceOptionsBuilder builder)
        => builder._options;
}
