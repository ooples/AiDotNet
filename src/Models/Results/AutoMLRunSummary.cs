using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a redacted (safe-to-share) summary of an AutoML run.
/// </summary>
/// <remarks>
/// <para>
/// This summary is intended for facade outputs (for example, <c>PredictionModelResult</c>) and avoids exposing
/// hyperparameters or sensitive model details. It provides transparency into the AutoML process while protecting
/// proprietary implementation choices.
/// </para>
/// <para><b>For Beginners:</b> AutoML is an automatic process that tries multiple model attempts ("trials")
/// and keeps the best one. This class tells you how that search went, including:
/// - How many trials ran
/// - The best score found
/// - A per-trial outcome history (without exposing secret settings)
/// </para>
/// </remarks>
public sealed class AutoMLRunSummary
{
    /// <summary>
    /// Gets or sets the search strategy used for the AutoML run, if known.
    /// </summary>
    public AutoMLSearchStrategy? SearchStrategy { get; set; }

    /// <summary>
    /// Gets or sets the time limit used for the AutoML search.
    /// </summary>
    public TimeSpan TimeLimit { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of trials allowed for the AutoML search.
    /// </summary>
    public int TrialLimit { get; set; }

    /// <summary>
    /// Gets or sets the optimization metric used to rank trials.
    /// </summary>
    public MetricType OptimizationMetric { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether higher metric values are better.
    /// </summary>
    public bool MaximizeMetric { get; set; }

    /// <summary>
    /// Gets or sets the best score achieved during the AutoML search.
    /// </summary>
    public double BestScore { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether AutoML selected an ensemble as the final model.
    /// </summary>
    public bool UsedEnsemble { get; set; }

    /// <summary>
    /// Gets or sets the number of models in the selected ensemble, when applicable.
    /// </summary>
    public int? EnsembleSize { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the AutoML search started.
    /// </summary>
    public DateTimeOffset SearchStartedUtc { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the AutoML search ended.
    /// </summary>
    public DateTimeOffset SearchEndedUtc { get; set; }

    /// <summary>
    /// Gets or sets a redacted list of trial summaries.
    /// </summary>
    public List<AutoMLTrialSummary> Trials { get; set; } = new();
}
