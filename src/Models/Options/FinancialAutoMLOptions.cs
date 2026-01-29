using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Finance.AutoML;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for financial AutoML runs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinancialAutoML selects among finance-specific models based on the chosen domain.
/// It follows the facade pattern: you provide a minimal setup and the library supplies defaults.
/// </para>
/// <para>
/// <b>For Beginners:</b> Use this class to say "I want AutoML for forecasting" or
/// "I want AutoML for risk." Then provide your architecture and budget.
/// </para>
/// </remarks>
public class FinancialAutoMLOptions<T>
{
    /// <summary>
    /// Gets or sets the finance domain to search.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Forecasting predicts future values, while Risk estimates
    /// exposure to losses.
    /// </para>
    /// </remarks>
    public FinancialDomain Domain { get; set; } = FinancialDomain.Forecasting;

    /// <summary>
    /// Gets or sets the user-provided neural network architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You provide the architecture so the library does not
    /// expose internal design details. AutoML will reuse this architecture
    /// across candidate finance models.
    /// </para>
    /// </remarks>
    public NeuralNetworkArchitecture<T>? Architecture { get; set; }

    /// <summary>
    /// Gets or sets the compute budget for the AutoML run.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The budget controls how long AutoML searches.
    /// Choose a smaller preset for quick results or a larger one for deeper search.
    /// </para>
    /// </remarks>
    public AutoMLBudgetOptions Budget { get; set; } = new();

    /// <summary>
    /// Gets or sets an optional optimization metric override.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you do not set this, AutoML picks a sensible
    /// default metric for the selected domain.
    /// </para>
    /// </remarks>
    public MetricType? OptimizationMetricOverride { get; set; }

    /// <summary>
    /// Gets or sets the AutoML search strategy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Random search is a reliable default. Other strategies
    /// can be added later without changing your model code.
    /// </para>
    /// </remarks>
    public AutoMLSearchStrategy SearchStrategy { get; set; } = AutoMLSearchStrategy.RandomSearch;

    /// <summary>
    /// Gets or sets an optional list of candidate models to consider.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you leave this empty, AutoML will choose a curated
    /// set of finance models for the selected domain.
    /// </para>
    /// </remarks>
    public List<ModelType>? CandidateModels { get; set; }

    /// <summary>
    /// Gets or sets optional cross-validation settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cross-validation gives more reliable scores but takes longer.
    /// Leave this null for faster runs.
    /// </para>
    /// </remarks>
    public CrossValidationOptions? CrossValidation { get; set; }

    /// <summary>
    /// Validates the options and throws if required values are missing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This catches missing inputs (like architecture)
    /// before AutoML starts.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (Architecture is null)
            throw new ArgumentNullException(nameof(Architecture), "Architecture must be provided.");
    }
}
