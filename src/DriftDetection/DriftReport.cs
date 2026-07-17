namespace AiDotNet.DriftDetection;

/// <summary>
/// Which distribution shift a <see cref="DriftReport"/> attributes drift to.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A model can go stale for two different reasons, and the fix differs:
/// <list type="bullet">
/// <item><description><b>Concept drift</b> — the relationship between inputs and the answer changed
/// (P(y|x)); the model's predictions are getting <i>wrong</i>. Usually means retrain.</description></item>
/// <item><description><b>Covariate drift</b> — the inputs themselves shifted (P(x)); the model may not
/// be wrong yet, but it is now operating outside the data it learned from. Usually means investigate the
/// data pipeline before accuracy degrades.</description></item>
/// </list>
/// Reporting <i>which</i> is happening — not just <i>that</i> something is — is what makes this more
/// actionable than a single error-stream detector.</para>
/// </remarks>
public enum DriftSource
{
    /// <summary>No drift detected on either lens.</summary>
    None = 0,

    /// <summary>Concept drift only: the error stream drifted (P(y|x) changed).</summary>
    Concept = 1,

    /// <summary>Covariate drift only: the prediction distribution shifted (P(x) changed).</summary>
    Covariate = 2,

    /// <summary>Both concept and covariate drift detected.</summary>
    Both = 3,
}

/// <summary>
/// The attributed drift assessment produced by a <see cref="DriftMonitor{T}"/> over a checked stream.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After training, the model's errors and predictions on held-out data are
/// streamed through a drift monitor calibrated on the training data. This report says whether that
/// held-out stream already looks like it is drifting away from what the model learned, and — the useful
/// part — whether the drift is in the errors (concept) or the inputs (covariate).</para>
/// </remarks>
public sealed class DriftReport
{
    /// <summary>Gets whether any drift (concept or covariate) was detected.</summary>
    public bool DriftDetected { get; init; }

    /// <summary>Gets whether a warning-level (pre-drift) signal was raised on either lens.</summary>
    public bool WarningDetected { get; init; }

    /// <summary>Gets which shift the detected drift is attributed to.</summary>
    public DriftSource Source { get; init; } = DriftSource.None;

    /// <summary>
    /// Gets the index in the checked stream at which drift was first flagged, or <c>-1</c> when none was.
    /// </summary>
    public int FirstDriftIndex { get; init; } = -1;

    /// <summary>Gets the number of observations streamed through the monitor during the check.</summary>
    public int ObservationsChecked { get; init; }
}
