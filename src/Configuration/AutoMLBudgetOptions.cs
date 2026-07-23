using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options that control AutoML compute budgets.
/// </summary>
/// <remarks>
/// <para>
/// AutoML budgets define how long AutoML is allowed to search and how many trials it can run.
/// </para>
/// <para>
/// <b>For Beginners:</b> This controls how much time AutoML is allowed to spend searching for a good model.
/// If you want faster results, choose a smaller preset or set a shorter time limit / fewer trials.
/// </para>
/// </remarks>
public class AutoMLBudgetOptions
{
    /// <summary>
    /// Gets or sets the budget preset used when explicit overrides are not provided.
    /// </summary>
    public AutoMLBudgetPreset Preset { get; set; } = AutoMLBudgetPreset.Standard;

    /// <summary>
    /// Gets or sets an optional time limit override for the AutoML run.
    /// </summary>
    /// <remarks>
    /// If null, the preset default is used.
    /// </remarks>
    public TimeSpan? TimeLimitOverride { get; set; }

    /// <summary>
    /// Gets or sets an optional trial limit override for the AutoML run.
    /// </summary>
    /// <remarks>
    /// If null, the preset default is used.
    /// </remarks>
    public int? TrialLimitOverride { get; set; }
}

