using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Core interface for federated unlearning: removes a client's contribution from the global model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This interface defines the "forget" operation for federated learning.
/// When a client exercises their GDPR right to be forgotten, you call <see cref="Unlearn"/> with
/// their client ID. The unlearner modifies the global model to remove that client's influence
/// and returns a certificate proving it was done.</para>
///
/// <para><b>Usage:</b></para>
/// <code>
/// var unlearner = new GradientAscentUnlearner&lt;double&gt;(options);
/// var certificate = unlearner.Unlearn(clientId, globalModel, clientHistories);
/// // certificate.Verified == true means unlearning was successful
/// // certificate.MembershipInferenceScore close to 0.5 means no memorization
/// </code>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public interface IFederatedUnlearner<T>
{
    /// <summary>
    /// Removes a client's contribution from the global model.
    /// </summary>
    /// <param name="targetClientId">ID of the client to unlearn.</param>
    /// <param name="globalModel">Current global model parameters.</param>
    /// <param name="clientHistories">Historical model updates per client per round.
    /// Key: clientId, Value: list of model updates (one per round they participated).</param>
    /// <returns>Unlearning certificate with the modified model and verification metrics.</returns>
    (Tensor<T> UnlearnedModel, UnlearningCertificate Certificate) Unlearn(
        int targetClientId,
        Tensor<T> globalModel,
        Dictionary<int, List<Tensor<T>>> clientHistories);

    /// <summary>
    /// Gets the name of this unlearning method.
    /// </summary>
    string MethodName { get; }
}
