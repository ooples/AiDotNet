using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.DriftDetection;

/// <summary>
/// Detects concept drift in federated learning by monitoring client model updates over time.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In production FL, data doesn't stay static. Customer behavior changes,
/// new fraud patterns emerge, market conditions shift. This interface monitors each client's training
/// behavior and flags when their data distribution has changed significantly. The system can then
/// adapt (e.g., increase that client's training epochs, adjust aggregation weights, or trigger
/// selective retraining).</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var detector = new StatisticalDriftDetector&lt;double&gt;(options);
/// // After each FL round:
/// var report = detector.DetectDrift(round, clientModels, globalModel, clientMetrics);
/// if (report.GlobalDriftDetected) { /* adapt training */ }
/// </code>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface IFederatedDriftDetector<T>
{
    /// <summary>
    /// Detects drift across all federated clients for the current round.
    /// </summary>
    /// <param name="round">Current FL round number.</param>
    /// <param name="clientModels">Current model updates from each client.</param>
    /// <param name="globalModel">Current global model parameters.</param>
    /// <param name="clientMetrics">Per-client metrics (e.g., loss, accuracy) from this round.
    /// Key: clientId, Value: metric value (typically loss).</param>
    /// <returns>Drift report with per-client scores and global drift assessment.</returns>
    DriftReport DetectDrift(
        int round,
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, double> clientMetrics);

    /// <summary>
    /// Gets recommended aggregation weight adjustments based on detected drift.
    /// </summary>
    /// <param name="originalWeights">Original client aggregation weights.</param>
    /// <param name="driftReport">Latest drift report from <see cref="DetectDrift"/>.</param>
    /// <returns>Adjusted weights (stable clients get more weight, drifting clients get less).</returns>
    Dictionary<int, double> GetAdaptiveWeights(
        Dictionary<int, double> originalWeights,
        DriftReport driftReport);

    /// <summary>
    /// Resets drift detection state for all clients.
    /// Call this after a global model reset or after handling drift.
    /// </summary>
    void Reset();

    /// <summary>
    /// Gets the name of the drift detection method.
    /// </summary>
    string MethodName { get; }
}
