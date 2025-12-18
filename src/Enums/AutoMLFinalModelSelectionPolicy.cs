namespace AiDotNet.Enums;

/// <summary>
/// Defines how AutoML chooses the final model to return after the search completes.
/// </summary>
/// <remarks>
/// <para>
/// AutoML can return the single best trial, or it can optionally build an ensemble from top trials
/// and return that instead.
/// </para>
/// <para>
/// <b>For Beginners:</b> An ensemble combines multiple models to often improve accuracy and stability.
/// This setting controls whether AutoML should return the best single model or an ensemble.
/// </para>
/// </remarks>
public enum AutoMLFinalModelSelectionPolicy
{
    /// <summary>
    /// Always return the best single trial model.
    /// </summary>
    BestSingleModel,

    /// <summary>
    /// Build an ensemble and return it if it scores better than the best single model.
    /// </summary>
    UseEnsembleIfBetter,

    /// <summary>
    /// Always return an ensemble when enough successful trials exist.
    /// </summary>
    AlwaysUseEnsemble
}
