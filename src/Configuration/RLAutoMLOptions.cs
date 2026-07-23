using AiDotNet.AutoML;
using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running AutoML over reinforcement learning agents and hyperparameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This options class is designed for use with <c>AiModelBuilder</c> when training RL agents.
/// It supports the AiDotNet facade pattern by providing sensible defaults while allowing customization.
/// </para>
/// <para><b>For Beginners:</b> RL AutoML tries a few different RL agent settings, measures which one earns
/// the most reward, and then trains the best configuration for your full training budget.</para>
/// </remarks>
public class RLAutoMLOptions<T>
{
    /// <summary>
    /// Gets or sets the number of training episodes to run per AutoML trial.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each trial trains briefly to estimate how good that configuration is.
    /// Lower values are faster but noisier; higher values are slower but more reliable.
    /// </remarks>
    public int TrainingEpisodesPerTrial { get; set; } = 50;

    /// <summary>
    /// Gets or sets the number of evaluation episodes to run per AutoML trial (no learning).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> After a trial is trained, the agent is evaluated without exploration to measure performance.
    /// </remarks>
    public int EvaluationEpisodesPerTrial { get; set; } = 10;

    /// <summary>
    /// Gets or sets an optional maximum step count per episode override for AutoML trials.
    /// </summary>
    /// <remarks>
    /// If null, the configured <see cref="RLTrainingOptions{T}.MaxStepsPerEpisode"/> is used.
    /// </remarks>
    public int? MaxStepsPerEpisodeOverride { get; set; }

    /// <summary>
    /// Gets or sets the allowed agent types for RL AutoML.
    /// </summary>
    /// <remarks>
    /// If null or empty, AiDotNet selects a default set based on whether the environment action space is discrete or continuous.
    /// </remarks>
    public List<RLAutoMLAgentType>? CandidateAgents { get; set; }

    /// <summary>
    /// Gets or sets optional hyperparameter search-space overrides.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This dictionary is merged into the built-in search spaces. Unknown keys are ignored by agents that don't support them.
    /// </para>
    /// <para><b>For Beginners:</b> Use this if you want to constrain what AutoML is allowed to try
    /// (for example, a smaller learning rate range).</para>
    /// </remarks>
    public Dictionary<string, ParameterRange> SearchSpaceOverrides { get; set; } = new(StringComparer.Ordinal);
}

