using System;

namespace AiDotNet.Models.Results;

/// <summary>
/// Represents a redacted (safe-to-share) summary of a single AutoML trial.
/// </summary>
/// <remarks>
/// <para>
/// This summary intentionally excludes hyperparameter values, model weights, and other implementation details.
/// It is designed to support the AiDotNet facade pattern, where users can review outcomes without gaining access
/// to proprietary configuration details.
/// </para>
/// <para><b>For Beginners:</b> AutoML tries many different "trials" (model attempts).
/// This class records what happened for one trial:
/// - Did it succeed?
/// - How long did it take?
/// - What score did it achieve?
/// </para>
/// </remarks>
public sealed class AutoMLTrialSummary
{
    /// <summary>
    /// Gets or sets the unique identifier for the trial.
    /// </summary>
    public int TrialId { get; set; }

    /// <summary>
    /// Gets or sets the score achieved by this trial.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets the duration of the trial.
    /// </summary>
    public TimeSpan Duration { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the trial completed (UTC).
    /// </summary>
    public DateTime CompletedUtc { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the trial completed successfully.
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets an error message when the trial fails.
    /// </summary>
    public string? ErrorMessage { get; set; }
}

