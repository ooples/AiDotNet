using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for AutoML ensembling.
/// </summary>
/// <remarks>
/// <para>
/// Ensembling retrains a small set of top-performing trials on the full training data and combines their
/// predictions into a single "ensemble" model. This can improve accuracy and reduce variance.
/// </para>
/// <para>
/// <b>For Beginners:</b> An ensemble is like asking multiple experts and averaging their answers.
/// It often performs better than relying on a single model.
/// </para>
/// </remarks>
public sealed class AutoMLEnsembleOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether AutoML should attempt to build an ensemble after the search.
    /// </summary>
    public bool Enabled { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of top trials to include in the ensemble.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Values less than 2 disable ensembling (an ensemble requires at least two models).
    /// </para>
    /// </remarks>
    public int MaxModelCount { get; set; } = 3;

    /// <summary>
    /// Gets or sets the policy that determines whether the ensemble replaces the best single model.
    /// </summary>
    public AutoMLFinalModelSelectionPolicy FinalSelectionPolicy { get; set; } = AutoMLFinalModelSelectionPolicy.UseEnsembleIfBetter;
}
