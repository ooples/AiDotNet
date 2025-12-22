namespace AiDotNet.Interfaces;

/// <summary>
/// Tracks cumulative privacy loss across federated learning rounds.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Differential privacy has a finite "budget" (epsilon, delta).
/// Each training round spends some of that budget. A privacy accountant keeps track
/// of what was spent so you can report guarantees and enforce limits.
/// </remarks>
public interface IPrivacyAccountant
{
    /// <summary>
    /// Records a single privacy event (typically one federated learning round).
    /// </summary>
    /// <param name="epsilon">The epsilon value used for this event.</param>
    /// <param name="delta">The delta value used for this event.</param>
    /// <param name="samplingRate">The client participation rate for this event (0.0 to 1.0).</param>
    void AddRound(double epsilon, double delta, double samplingRate);

    /// <summary>
    /// Gets the total epsilon consumed so far according to this accountant.
    /// </summary>
    double GetTotalEpsilonConsumed();

    /// <summary>
    /// Gets the total delta consumed so far according to this accountant.
    /// </summary>
    double GetTotalDeltaConsumed();

    /// <summary>
    /// Gets a reported epsilon value at a given target delta (if supported by the accountant).
    /// </summary>
    /// <param name="targetDelta">The delta to report epsilon for.</param>
    double GetEpsilonAtDelta(double targetDelta);

    /// <summary>
    /// Gets the name of this privacy accountant implementation.
    /// </summary>
    string GetAccountantName();
}

