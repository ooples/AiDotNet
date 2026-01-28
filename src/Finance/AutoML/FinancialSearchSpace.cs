using AiDotNet.AutoML;
using AiDotNet.Enums;

namespace AiDotNet.Finance.AutoML;

/// <summary>
/// Provides default AutoML search spaces for finance models.
/// </summary>
/// <remarks>
/// <para>
/// The search space defines which hyperparameters AutoML is allowed to explore.
/// This implementation keeps the default space intentionally small to preserve the
/// facade pattern and avoid exposing sensitive configuration details.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML can either just pick a model or also tune settings.
/// This class tells AutoML which settings it is allowed to tune.
/// </para>
/// </remarks>
public sealed class FinancialSearchSpace
{
    private readonly FinancialDomain _domain;

    /// <summary>
    /// Initializes a new search space provider for the chosen finance domain.
    /// </summary>
    /// <param name="domain">The finance domain.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different tasks (forecasting vs risk) use different models,
    /// so the search space is tied to the domain.
    /// </para>
    /// </remarks>
    public FinancialSearchSpace(FinancialDomain domain)
    {
        _domain = domain;
    }

    /// <summary>
    /// Gets the default search space for a specific model type.
    /// </summary>
    /// <param name="modelType">The model type to configure.</param>
    /// <returns>Dictionary of parameter ranges for AutoML sampling.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returning an empty dictionary means AutoML will only
    /// choose between models and not tune internal settings.
    /// </para>
    /// </remarks>
    public Dictionary<string, ParameterRange> GetSearchSpace(ModelType modelType)
    {
        _ = _domain;
        return new Dictionary<string, ParameterRange>(StringComparer.Ordinal);
    }
}
