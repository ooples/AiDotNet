using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running AutoML through the AiDotNet facade.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
/// <remarks>
/// <para>
/// This options class is designed for use with <c>PredictionModelBuilder</c>.
/// It follows the AiDotNet facade pattern: users provide minimal configuration, and the library supplies
/// industry-standard defaults internally.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML is an automatic "model picker + tuner".
/// You choose a budget, and AutoML tries different models/settings to find a strong performer.
/// </para>
/// </remarks>
public class AutoMLOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the compute budget for the AutoML run.
    /// </summary>
    public AutoMLBudgetOptions Budget { get; set; } = new();

    /// <summary>
    /// Gets or sets an optional task family override.
    /// </summary>
    /// <remarks>
    /// If null, AutoML infers the task family from the training labels and model/task configuration.
    /// </remarks>
    public AutoMLTaskFamily? TaskFamilyOverride { get; set; }

    /// <summary>
    /// Gets or sets an optional optimization metric override.
    /// </summary>
    /// <remarks>
    /// If null, AutoML chooses an industry-standard default metric for the inferred task family.
    /// </remarks>
    public MetricType? OptimizationMetricOverride { get; set; }

    /// <summary>
    /// Gets or sets the search strategy used for hyperparameter/model exploration.
    /// </summary>
    public AutoMLSearchStrategy SearchStrategy { get; set; } = AutoMLSearchStrategy.RandomSearch;

    /// <summary>
    /// Gets or sets multi-fidelity options used when <see cref="SearchStrategy"/> is <see cref="AutoMLSearchStrategy.MultiFidelity"/>.
    /// </summary>
    /// <remarks>
    /// If null, sensible defaults are used.
    /// </remarks>
    public AutoMLMultiFidelityOptions? MultiFidelity { get; set; }

    /// <summary>
    /// Gets or sets optional ensembling options applied after the AutoML search completes.
    /// </summary>
    /// <remarks>
    /// If null, the facade uses sensible defaults based on the selected budget preset.
    /// </remarks>
    public AutoMLEnsembleOptions? Ensembling { get; set; }

    /// <summary>
    /// Gets or sets reinforcement-learning specific AutoML options.
    /// </summary>
    /// <remarks>
    /// This is used when <see cref="TaskFamilyOverride"/> is set to <see cref="AutoMLTaskFamily.ReinforcementLearning"/>.
    /// If null, sensible defaults are used.
    /// </remarks>
    public RLAutoMLOptions<T>? ReinforcementLearning { get; set; }
}
