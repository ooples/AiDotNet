using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Defines and enforces fairness constraints during federated learning aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A fairness constraint ensures the global model works well for ALL
/// client groups, not just the majority. Without it, a model trained across 10 urban hospitals
/// and 2 rural hospitals might be great for urban patients but poor for rural ones. Fairness
/// constraints rebalance the aggregation to protect underrepresented groups.</para>
///
/// <para><b>Common fairness measures:</b></para>
/// <list type="bullet">
/// <item><description><b>Demographic Parity:</b> Positive prediction rates should be equal across groups.</description></item>
/// <item><description><b>Equalized Odds:</b> Error rates should be equal across groups.</description></item>
/// <item><description><b>Minimax:</b> Minimize the worst-case group performance.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface IFairnessConstraint<T>
{
    /// <summary>
    /// Evaluates the current fairness metric across client groups.
    /// </summary>
    /// <param name="clientModels">Current model updates from each client.</param>
    /// <param name="globalModel">The current global model.</param>
    /// <param name="clientGroups">Mapping of client IDs to their group assignments.</param>
    /// <returns>Fairness violation score: 0 = perfectly fair, higher = more unfair.</returns>
    double EvaluateFairness(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, int> clientGroups);

    /// <summary>
    /// Adjusts aggregation weights to enforce fairness constraints.
    /// </summary>
    /// <param name="originalWeights">Original aggregation weights (e.g., from FedAvg).</param>
    /// <param name="clientModels">Current model updates from each client.</param>
    /// <param name="globalModel">The current global model.</param>
    /// <param name="clientGroups">Mapping of client IDs to their group assignments.</param>
    /// <returns>Adjusted weights that incorporate fairness corrections.</returns>
    Dictionary<int, double> AdjustWeights(
        Dictionary<int, double> originalWeights,
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, int> clientGroups);

    /// <summary>
    /// Gets whether the current model satisfies the fairness constraint.
    /// </summary>
    /// <param name="fairnessViolation">The violation score from <see cref="EvaluateFairness"/>.</param>
    /// <returns>True if the constraint is satisfied (violation below threshold).</returns>
    bool IsSatisfied(double fairnessViolation);

    /// <summary>
    /// Gets the name of this fairness constraint.
    /// </summary>
    string ConstraintName { get; }
}
