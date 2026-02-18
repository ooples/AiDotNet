namespace AiDotNet.FederatedLearning.BackdoorDefense;

/// <summary>
/// Interface for detecting backdoor attacks in federated learning updates.
/// </summary>
/// <remarks>
/// <para>
/// Backdoor attacks are a stealthy form of poisoning where a malicious client injects a
/// "trigger pattern" into the model. The model behaves normally on clean inputs but produces
/// attacker-chosen outputs when the trigger is present. Unlike untargeted poisoning (handled
/// by Byzantine-robust aggregators), backdoor attacks are targeted and can evade statistical
/// anomaly detection.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine a stop sign recognition model. A backdoor attack might make
/// the model correctly identify all normal stop signs, but misclassify any stop sign with a
/// small yellow sticker as a speed limit sign. Standard defenses (like Krum or Bulyan) may
/// not catch this because the poisoned update looks statistically normal overall — it only
/// misbehaves on the specific trigger pattern.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IBackdoorDetector<T>
{
    /// <summary>
    /// Analyzes client updates and returns a suspicion score for each client.
    /// </summary>
    /// <param name="clientUpdates">Dictionary of client ID to model update (gradient or parameter delta).</param>
    /// <param name="globalModel">The current global model parameters.</param>
    /// <returns>Dictionary of client ID to suspicion score (0.0 = clean, 1.0 = highly suspicious).</returns>
    Dictionary<int, double> DetectSuspiciousUpdates(Dictionary<int, Vector<T>> clientUpdates, Vector<T> globalModel);

    /// <summary>
    /// Filters out suspected backdoor updates, returning only clean updates.
    /// </summary>
    /// <param name="clientUpdates">All client updates.</param>
    /// <param name="globalModel">The current global model.</param>
    /// <param name="suspicionThreshold">Threshold above which clients are filtered out.</param>
    /// <returns>Filtered dictionary containing only non-suspicious updates.</returns>
    Dictionary<int, Vector<T>> FilterMaliciousUpdates(Dictionary<int, Vector<T>> clientUpdates,
        Vector<T> globalModel, double suspicionThreshold);

    /// <summary>
    /// Gets the detector name for logging.
    /// </summary>
    string DetectorName { get; }
}
