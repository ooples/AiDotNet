using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Evaluates how much each client contributed to the federated global model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In a group project, some students do more work than others.
/// This interface measures each client's "fair share" of the global model improvement.
/// High-contribution clients provided valuable data, while low-contribution clients may be
/// free-riding (benefiting without contributing). This is essential for fair compensation
/// and for detecting poisoned or low-quality updates.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var evaluator = new DataShapleyEvaluator&lt;double&gt;(options);
/// var scores = evaluator.EvaluateContributions(clientModels, globalModel, clientHistories);
/// // scores[clientId] = contribution score (higher = more valuable)
/// </code>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface IClientContributionEvaluator<T>
{
    /// <summary>
    /// Evaluates the contribution of each client to the global model.
    /// </summary>
    /// <param name="clientModels">Current model updates from each client. Key: clientId, Value: model update tensor.</param>
    /// <param name="globalModel">The current global model parameters.</param>
    /// <param name="clientHistories">Historical model updates per client per round.</param>
    /// <returns>Dictionary mapping each client ID to their contribution score (higher = more valuable).</returns>
    Dictionary<int, double> EvaluateContributions(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, List<Tensor<T>>> clientHistories);

    /// <summary>
    /// Identifies free-rider clients whose contribution falls below the threshold.
    /// </summary>
    /// <param name="contributionScores">Contribution scores from <see cref="EvaluateContributions"/>.</param>
    /// <returns>Set of client IDs identified as free-riders.</returns>
    HashSet<int> IdentifyFreeRiders(Dictionary<int, double> contributionScores);

    /// <summary>
    /// Gets the name of this contribution evaluation method.
    /// </summary>
    string MethodName { get; }
}
